"""[A0.5] Live-readiness signal gate.

Parses the YAML frontmatter of docs/phase8/strategy_evidence_v1.md and
validates all strategy thresholds before allowing exchange_mode: live.

Exit codes (when used as __main__):
    0 — all checks passed
    2 — gate failed (one or more thresholds not met)
    1 — error (evidence pack missing / malformed)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from deployment.monitoring.alerter import TradingAlerter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (mirror config/deployment.yaml live_signal_gate section)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "min_sharpe_net": 0.5,
    "min_dsr": 0.0,
    "min_bootstrap_ci_lower": 0.0,
    "max_permutation_p": 0.05,
    "max_regime_dd": 0.30,
}

_DEFAULT_MAX_EVIDENCE_AGE_DAYS: float = 30.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalGateResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_pack_path: Optional[Path] = None
    evidence_pack_age_days: float = 0.0


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class LiveSignalGate:
    """Validate strategy evidence pack before allowing live trading.

    Parameters
    ----------
    evidence_pack:
        Path to ``strategy_evidence_v1.md`` (YAML frontmatter required).
    thresholds:
        Override dict; missing keys fall back to ``_DEFAULT_THRESHOLDS``.
    max_evidence_age_days:
        Evidence pack must have been generated within this many days.
    alerter:
        Optional :class:`TradingAlerter` — called on gate failure.
    _now:
        Override "now" datetime (UTC) for testing.
    """

    def __init__(
        self,
        evidence_pack: Path,
        thresholds: Optional[Dict[str, Any]] = None,
        max_evidence_age_days: float = _DEFAULT_MAX_EVIDENCE_AGE_DAYS,
        alerter: Optional["TradingAlerter"] = None,
        _now: Optional[datetime] = None,
    ) -> None:
        self.evidence_pack = Path(evidence_pack)
        self.thresholds = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.max_evidence_age_days = max_evidence_age_days
        self.alerter = alerter
        self._now = _now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> SignalGateResult:
        """Run all gate checks and return a :class:`SignalGateResult`."""
        failures: List[str] = []

        # 1. Evidence pack exists
        if not self.evidence_pack.exists():
            return SignalGateResult(
                passed=False,
                failures=[f"Evidence pack not found: {self.evidence_pack}"],
                evidence_pack_path=self.evidence_pack,
            )

        # 2. Parse frontmatter
        try:
            frontmatter = _parse_frontmatter(self.evidence_pack)
        except ValueError as exc:
            return SignalGateResult(
                passed=False,
                failures=[f"Evidence pack parse error: {exc}"],
                evidence_pack_path=self.evidence_pack,
            )

        metrics = frontmatter.get("metrics", {})

        # 3. Age check
        age_days = self._compute_age_days(frontmatter)
        if age_days > self.max_evidence_age_days:
            failures.append(
                f"Evidence pack is {age_days:.1f} days old "
                f"(max {self.max_evidence_age_days:.0f} days)"
            )

        # 4. Net Sharpe
        net_sharpe = _get_float(metrics, "net_sharpe")
        if net_sharpe is None:
            failures.append("metrics.net_sharpe missing")
        elif net_sharpe <= self.thresholds["min_sharpe_net"]:
            failures.append(
                f"net_sharpe {net_sharpe:.4f} ≤ threshold {self.thresholds['min_sharpe_net']}"
            )

        # 5. DSR
        dsr = _get_float(metrics, "dsr")
        if dsr is None:
            failures.append("metrics.dsr missing")
        elif dsr <= self.thresholds["min_dsr"]:
            failures.append(
                f"dsr {dsr:.4f} ≤ threshold {self.thresholds['min_dsr']}"
            )

        # 6. Bootstrap CI lower
        ci_lower = _get_float(metrics, "bootstrap_ci_lower")
        if ci_lower is None:
            failures.append("metrics.bootstrap_ci_lower missing")
        elif ci_lower <= self.thresholds["min_bootstrap_ci_lower"]:
            failures.append(
                f"bootstrap_ci_lower {ci_lower:.4f} ≤ threshold "
                f"{self.thresholds['min_bootstrap_ci_lower']}"
            )

        # 7. Permutation p-value
        perm_p = _get_float(metrics, "permutation_p")
        if perm_p is None:
            failures.append("metrics.permutation_p missing")
        elif perm_p >= self.thresholds["max_permutation_p"]:
            failures.append(
                f"permutation_p {perm_p:.4f} ≥ threshold {self.thresholds['max_permutation_p']}"
            )

        # 8. Max regime DD (crisis)
        max_regime_dd_cfg = self.thresholds["max_regime_dd"]
        regime_dd = metrics.get("max_regime_dd", {})
        if not isinstance(regime_dd, dict):
            failures.append("metrics.max_regime_dd must be a mapping of regime→float")
        else:
            for regime, dd_val in regime_dd.items():
                try:
                    dd_float = float(dd_val)
                except (TypeError, ValueError):
                    failures.append(f"max_regime_dd.{regime} is not numeric")
                    continue
                if dd_float >= max_regime_dd_cfg:
                    failures.append(
                        f"max_regime_dd.{regime} {dd_float:.4f} ≥ threshold {max_regime_dd_cfg}"
                    )

        passed = len(failures) == 0
        result = SignalGateResult(
            passed=passed,
            failures=failures,
            metrics=metrics,
            evidence_pack_path=self.evidence_pack,
            evidence_pack_age_days=age_days,
        )

        if not passed and self.alerter is not None:
            self.alerter.send_alert(
                f"Live signal gate FAILED: {len(failures)} check(s) failed",
                level="CRITICAL",
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_age_days(self, frontmatter: Dict[str, Any]) -> float:
        generated_at_raw = frontmatter.get("generated_at")
        if not generated_at_raw:
            return float("inf")
        try:
            generated_at = datetime.fromisoformat(str(generated_at_raw).rstrip("Z"))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            now = self._now if self._now is not None else datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            delta = now - generated_at
            return delta.total_seconds() / 86400.0
        except Exception:
            return float("inf")


# ---------------------------------------------------------------------------
# YAML frontmatter parser (no external deps)
# ---------------------------------------------------------------------------

def _parse_frontmatter(path: Path) -> Dict[str, Any]:
    """Extract and parse YAML frontmatter between ``---`` delimiters."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        raise ValueError("No YAML frontmatter found (expected '---' on line 1)")

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Frontmatter closing '---' not found")

    yaml_text = "\n".join(lines[1:end_idx])
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(yaml_text)
    except Exception as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Frontmatter must be a YAML mapping")

    return data


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    val = d.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI entry point (used by check_signal_gate.py and go_live_checklist Z1)
# ---------------------------------------------------------------------------

def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="[A0.5] Check live-readiness signal gate"
    )
    parser.add_argument(
        "--evidence-pack",
        default="docs/phase8/strategy_evidence_v1.md",
        help="Path to strategy_evidence_v1.md",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to deployment.yaml (reads live_signal_gate section)",
    )
    args = parser.parse_args(argv)

    # Load config if provided
    thresholds: Dict[str, Any] = {}
    max_age_days = _DEFAULT_MAX_EVIDENCE_AGE_DAYS
    evidence_pack_path = Path(args.evidence_pack)

    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config) as f:
                cfg = yaml.safe_load(f) or {}
            gate_cfg = cfg.get("live_signal_gate", {})
            if "evidence_pack" in gate_cfg:
                evidence_pack_path = Path(gate_cfg["evidence_pack"])
            max_age_days = float(gate_cfg.get("max_evidence_age_days", max_age_days))
            thresholds = gate_cfg.get("thresholds", {})
        except Exception as exc:
            print(f"ERROR: Failed to load config {args.config}: {exc}", file=sys.stderr)
            return 1

    gate = LiveSignalGate(
        evidence_pack=evidence_pack_path,
        thresholds=thresholds,
        max_evidence_age_days=max_age_days,
    )
    result = gate.check()

    if result.passed:
        print("✅ Signal gate PASSED — live mode authorized")
        print(f"   Evidence pack: {result.evidence_pack_path}")
        print(f"   Age: {result.evidence_pack_age_days:.1f} days")
        for k, v in result.metrics.items():
            print(f"   {k}: {v}")
        return 0
    else:
        print("❌ Signal gate FAILED — live mode NOT authorized")
        for failure in result.failures:
            print(f"   • {failure}")
        return 2


if __name__ == "__main__":
    sys.exit(_cli_main())
