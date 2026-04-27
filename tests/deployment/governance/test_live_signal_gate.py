"""[A0.5] Tests for deployment.governance.live_signal_gate.

8 cases required by phase8-restructured spec:
  1. All thresholds passed → gate passes
  2. net_sharpe below threshold → gate fails
  3. DSR below threshold → gate fails
  4. bootstrap_ci_lower below threshold → gate fails
  5. permutation_p above threshold → gate fails
  6. max_regime_dd crisis above threshold → gate fails
  7. Evidence pack age expired → gate fails
  8. Evidence pack file missing → gate fails

Plus supplementary cases:
  9.  Multiple simultaneous failures → all listed in result.failures
  10. Missing metrics key → failure with helpful message
  11. Alerter called on failure
  12. Malformed frontmatter → error result
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deployment.governance.live_signal_gate import LiveSignalGate, SignalGateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pack(
    tmp_path: Path,
    *,
    net_sharpe: float = 0.65,
    gross_sharpe: float = 0.80,
    dsr: float = 0.18,
    bootstrap_ci_lower: float = 0.12,
    bootstrap_ci_upper: float = 1.05,
    permutation_p: float = 0.018,
    crisis_dd: float = 0.21,
    trend_dd: float = 0.08,
    range_dd: float = 0.04,
    age_days: float = 0.0,
    generated_at: str | None = None,
    filename: str = "evidence.md",
) -> Path:
    now = datetime.now(timezone.utc) - timedelta(days=age_days)
    ts = generated_at or now.isoformat()

    content = textwrap.dedent(f"""\
        ---
        generated_at: {ts}
        walk_forward_period: "2025-04 to 2026-04"
        metrics:
          net_sharpe: {net_sharpe}
          gross_sharpe: {gross_sharpe}
          dsr: {dsr}
          bootstrap_ci_lower: {bootstrap_ci_lower}
          bootstrap_ci_upper: {bootstrap_ci_upper}
          permutation_p: {permutation_p}
          max_regime_dd:
            trend: {trend_dd}
            range: {range_dd}
            crisis: {crisis_dd}
        ---

        # Strategy Evidence Pack
        Body text here.
    """)
    p = tmp_path / filename
    p.write_text(content)
    return p


def _gate(pack: Path, **kwargs) -> LiveSignalGate:
    return LiveSignalGate(evidence_pack=pack, **kwargs)


# ---------------------------------------------------------------------------
# Case 1: All thresholds pass
# ---------------------------------------------------------------------------

def test_gate_passes_all_thresholds(tmp_path):
    pack = _make_pack(tmp_path)
    result = _gate(pack).check()
    assert result.passed is True
    assert result.failures == []
    assert result.evidence_pack_path == pack
    assert result.evidence_pack_age_days < 1.0


# ---------------------------------------------------------------------------
# Case 2: net_sharpe too low
# ---------------------------------------------------------------------------

def test_gate_fails_net_sharpe(tmp_path):
    pack = _make_pack(tmp_path, net_sharpe=0.40)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("net_sharpe" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 3: DSR too low
# ---------------------------------------------------------------------------

def test_gate_fails_dsr(tmp_path):
    pack = _make_pack(tmp_path, dsr=-0.05)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("dsr" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 4: bootstrap CI lower too low
# ---------------------------------------------------------------------------

def test_gate_fails_bootstrap_ci_lower(tmp_path):
    pack = _make_pack(tmp_path, bootstrap_ci_lower=-0.10)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("bootstrap_ci_lower" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 5: permutation p too high
# ---------------------------------------------------------------------------

def test_gate_fails_permutation_p(tmp_path):
    pack = _make_pack(tmp_path, permutation_p=0.12)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("permutation_p" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 6: regime DD too high (crisis)
# ---------------------------------------------------------------------------

def test_gate_fails_regime_dd(tmp_path):
    pack = _make_pack(tmp_path, crisis_dd=0.35)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("max_regime_dd" in f and "crisis" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 7: Evidence pack too old
# ---------------------------------------------------------------------------

def test_gate_fails_age_expired(tmp_path):
    pack = _make_pack(tmp_path, age_days=31.0)
    result = _gate(pack, max_evidence_age_days=30.0).check()
    assert result.passed is False
    assert any("days old" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 8: Evidence pack file missing
# ---------------------------------------------------------------------------

def test_gate_fails_missing_file(tmp_path):
    missing = tmp_path / "nonexistent.md"
    result = _gate(missing).check()
    assert result.passed is False
    assert any("not found" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Case 9: Multiple simultaneous failures reported
# ---------------------------------------------------------------------------

def test_gate_reports_multiple_failures(tmp_path):
    pack = _make_pack(tmp_path, net_sharpe=0.1, dsr=-1.0, permutation_p=0.99)
    result = _gate(pack).check()
    assert result.passed is False
    # At least 3 distinct failures
    assert len(result.failures) >= 3


# ---------------------------------------------------------------------------
# Case 10: Missing metrics key
# ---------------------------------------------------------------------------

def test_gate_fails_missing_metric_key(tmp_path):
    content = textwrap.dedent("""\
        ---
        generated_at: 2026-04-27T10:00:00+00:00
        walk_forward_period: "2025-04 to 2026-04"
        metrics:
          gross_sharpe: 0.80
        ---
        Body.
    """)
    pack = tmp_path / "sparse.md"
    pack.write_text(content)
    result = _gate(pack).check()
    assert result.passed is False
    # Should flag multiple missing keys
    missing_flags = [f for f in result.failures if "missing" in f]
    assert len(missing_flags) >= 2


# ---------------------------------------------------------------------------
# Case 11: Alerter called on failure
# ---------------------------------------------------------------------------

def test_gate_calls_alerter_on_failure(tmp_path):
    pack = _make_pack(tmp_path, net_sharpe=0.1)
    alerter = MagicMock()
    result = _gate(pack, alerter=alerter).check()
    assert result.passed is False
    alerter.send_alert.assert_called_once()
    call_kwargs = alerter.send_alert.call_args
    assert call_kwargs[1].get("level") == "CRITICAL" or (
        len(call_kwargs[0]) >= 2 and call_kwargs[0][1] == "CRITICAL"
    )


# ---------------------------------------------------------------------------
# Case 12: Malformed frontmatter
# ---------------------------------------------------------------------------

def test_gate_fails_malformed_frontmatter(tmp_path):
    content = "No frontmatter at all\njust body text\n"
    pack = tmp_path / "bad.md"
    pack.write_text(content)
    result = _gate(pack).check()
    assert result.passed is False
    assert any("frontmatter" in f.lower() or "parse" in f.lower() for f in result.failures)


# ---------------------------------------------------------------------------
# Custom thresholds override
# ---------------------------------------------------------------------------

def test_gate_custom_thresholds_pass(tmp_path):
    # Lower bar: net_sharpe threshold 0.3 instead of 0.5
    pack = _make_pack(tmp_path, net_sharpe=0.40)
    result = _gate(pack, thresholds={"min_sharpe_net": 0.3}).check()
    assert result.passed is True


def test_gate_custom_thresholds_fail(tmp_path):
    # Raise bar: require net_sharpe > 0.8
    pack = _make_pack(tmp_path, net_sharpe=0.65)
    result = _gate(pack, thresholds={"min_sharpe_net": 0.8}).check()
    assert result.passed is False


# ---------------------------------------------------------------------------
# Age boundary: exactly at limit should fail (> not >=)
# ---------------------------------------------------------------------------

def test_gate_age_exactly_at_limit_fails(tmp_path):
    pack = _make_pack(tmp_path, age_days=30.01)
    result = _gate(pack, max_evidence_age_days=30.0).check()
    assert result.passed is False


def test_gate_age_under_limit_passes_age_check(tmp_path):
    pack = _make_pack(tmp_path, age_days=5.0)
    # All other metrics good, only age check matters here
    result = _gate(pack, max_evidence_age_days=30.0).check()
    assert result.passed is True
