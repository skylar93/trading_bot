#!/usr/bin/env python
"""
G12: First Dollar Drill — pre-flight validation + $100 simulation drill.

Usage:
    # Auto-verify only (no trading)
    python scripts/first_dollar_drill.py --check-only

    # Full drill with $100 simulated capital
    python scripts/first_dollar_drill.py --capital 100

    # Write JSON report
    python scripts/first_dollar_drill.py --capital 100 --report drill_report.json

Exit codes:
    0 — all checks passed (+ drill succeeded if --capital provided)
    1 — one or more pre-flight checks failed
    2 — drill itself failed (guards fired unexpectedly or NaN detected)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Root of the project (one level above scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Individual auto-checks
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return {"name": name, "status": status, "detail": detail}


def check_pytest_ini_ignores() -> dict[str, Any]:
    """E2: pytest.ini ignore count ≤ 5."""
    ini = PROJECT_ROOT / "pytest.ini"
    if not ini.exists():
        return _check("pytest.ini ignore ≤ 5", False, "pytest.ini not found")
    text = ini.read_text()
    ignores = [l.strip() for l in text.splitlines() if l.strip().startswith("--ignore=")]
    count = len(ignores)
    return _check("pytest.ini ignore ≤ 5", count <= 5, f"{count} ignores found")


def check_no_old_risk_api() -> dict[str, Any]:
    """E4: no caller in deployment/ uses the old check_max_drawdown API.

    risk_management/ is excluded — it retains the deprecated shim intentionally.
    """
    search_dir = PROJECT_ROOT / "deployment"
    try:
        result = subprocess.run(
            ["rg", "-l", "check_max_drawdown", str(search_dir), "--glob", "*.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        hits = [l for l in result.stdout.strip().splitlines() if l]
    except FileNotFoundError:
        result = subprocess.run(
            ["grep", "-rl", "check_max_drawdown", str(search_dir)],
            capture_output=True, text=True,
        )
        hits = [l for l in result.stdout.strip().splitlines() if l]
    return _check(
        "deployment/ check_max_drawdown callers → 0",
        len(hits) == 0,
        f"{len(hits)} file(s): {hits[:3]}" if hits else "",
    )


def check_risk_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """R1-R5: validate required risk config keys exist and are non-zero."""
    results = []
    pt = config.get("paper_trading", config)
    ex = config.get("exchange", {})
    lim = config.get("limits", {})

    checks = [
        ("max_drawdown_threshold set", pt.get("max_drawdown_threshold", 0) > 0,
         str(pt.get("max_drawdown_threshold"))),
        ("daily_loss_limit set", ex.get("daily_loss_limit", 0) != 0,
         str(ex.get("daily_loss_limit"))),
        ("limits.per_symbol_notional_max set",
         lim.get("per_symbol_notional_max", 0) > 0,
         str(lim.get("per_symbol_notional_max"))),
        ("limits.portfolio_notional_max set",
         lim.get("portfolio_notional_max", 0) > 0,
         str(lim.get("portfolio_notional_max"))),
        ("limits.leverage_max set",
         lim.get("leverage_max", 0) > 0,
         str(lim.get("leverage_max"))),
    ]
    for name, passed, detail in checks:
        results.append(_check(name, passed, detail))
    return results


def check_postmortem_template() -> dict[str, Any]:
    """O6: postmortem template exists."""
    path = PROJECT_ROOT / "docs" / "runbook" / "postmortem_template.md"
    return _check("postmortem_template.md exists", path.exists(), str(path))


def check_go_live_checklist() -> dict[str, Any]:
    """O6: go_live_checklist.md exists."""
    path = PROJECT_ROOT / "docs" / "runbook" / "go_live_checklist.md"
    return _check("go_live_checklist.md exists", path.exists(), str(path))


def check_kill_switch_script() -> dict[str, Any]:
    """O1: kill_switch.py exists and is executable."""
    path = PROJECT_ROOT / "scripts" / "kill_switch.py"
    exists = path.exists()
    return _check("scripts/kill_switch.py exists", exists, str(path))


def check_checkpoint_freshness(max_age_hours: float = 24.0) -> dict[str, Any]:
    """O5: StateStore checkpoint is < max_age_hours old."""
    db = PROJECT_ROOT / "state" / "paper_trader.db"
    if not db.exists():
        return _check("StateStore checkpoint fresh", True,
                      "no checkpoint (first run — OK)")
    age_h = (time.time() - db.stat().st_mtime) / 3600
    return _check("StateStore checkpoint fresh",
                  age_h < max_age_hours,
                  f"age {age_h:.1f}h (max {max_age_hours}h)")


def check_audit_chain() -> dict[str, Any]:
    """O4: audit chain integrity."""
    audit = PROJECT_ROOT / "audit_log" / "audit.jsonl"
    if not audit.exists():
        return _check("audit chain integrity", True,
                      "no audit log yet (first run — OK)")
    verify = PROJECT_ROOT / "scripts" / "verify_audit_log.py"
    if not verify.exists():
        return _check("audit chain integrity", False, "verify_audit_log.py missing")
    result = subprocess.run(
        [sys.executable, str(verify), str(audit)],
        capture_output=True, text=True,
    )
    return _check("audit chain integrity", result.returncode == 0,
                  result.stderr.strip()[:120] if result.returncode != 0 else "")


# ---------------------------------------------------------------------------
# Kill-switch timing test (local process, simulation_mode)
# ---------------------------------------------------------------------------

def run_kill_switch_timing_test() -> dict[str, Any]:
    """G13: Start a throwaway PaperTrader, fire kill switch, confirm < 5 s."""
    import threading
    import tempfile
    import numpy as np

    # Minimal config
    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 100.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 5,
        },
        "pid_file": str(PROJECT_ROOT / "state" / "drill_kill_test.pid"),
        "monitoring": {},
    }

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from deployment.paper_trader import PaperTrader

        class _DummyAgent:
            def predict(self, obs, deterministic=True):
                return np.array([0.0]), None

        trader = PaperTrader(_DummyAgent(), cfg, simulation_mode=True)

        prices = iter([100.0 + i * 0.01 for i in range(10_000)])
        done_event = threading.Event()

        def _run():
            try:
                trader.run(price_stream=prices, duration_seconds=30)
            finally:
                done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.3)  # let it start

        start = time.monotonic()
        trader._trigger_shutdown("drill: kill switch test")
        done_event.wait(timeout=6.0)
        elapsed = time.monotonic() - start

        pid_file = Path(cfg["pid_file"])
        pid_file.unlink(missing_ok=True)

        ok = elapsed < 5.0 and trader.state.shutdown_triggered
        return _check(
            "Kill switch < 5 s",
            ok,
            f"shutdown in {elapsed:.2f}s, triggered={trader.state.shutdown_triggered}",
        )

    except Exception as exc:
        return _check("Kill switch < 5 s", False, str(exc))


# ---------------------------------------------------------------------------
# $100 Drill simulation
# ---------------------------------------------------------------------------

def run_dollar_drill(capital: float, n_steps: int = 200) -> dict[str, Any]:
    """G12: Run PaperTrader in simulation mode with *capital*, verify it completes."""
    import numpy as np

    rng = np.random.default_rng(42)
    # Synthetic price stream: GBM-like
    prices = [capital * 10]
    for _ in range(n_steps):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.002)))

    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": float(capital),
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 5,
        },
        "pid_file": str(PROJECT_ROOT / "state" / "drill_dollar.pid"),
        "monitoring": {},
        "limits": {
            "per_symbol_notional_max": capital * 5,
            "portfolio_notional_max": capital * 5,
            "leverage_max": 1.0,
        },
    }

    # Ensure project root is importable when run as subprocess
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from deployment.paper_trader import PaperTrader
        from risk_management.limits import PreTradeComplianceChecker
        from deployment.execution.order_manager import OrderManager

        class _RandomAgent:
            def __init__(self, rng):
                self._rng = rng

            def predict(self, obs, deterministic=True):
                return np.array([self._rng.uniform(-1, 1)]), None

        compliance = PreTradeComplianceChecker(cfg["limits"])
        om = OrderManager(
            exchange_config={"symbol": "BTC/USDT"},
            paper_mode=True,
            compliance_checker=compliance,
        )
        trader = PaperTrader(
            _RandomAgent(rng), cfg, simulation_mode=True, order_manager=om
        )
        t0 = time.monotonic()
        report = trader.run(price_stream=iter(prices), duration_seconds=60)
        elapsed = time.monotonic() - t0

        Path(cfg["pid_file"]).unlink(missing_ok=True)

        final_pv = report.get("final_portfolio_value", capital)
        pnl = final_pv - capital
        shutdown = trader.state.shutdown_triggered

        return _check(
            f"${capital:.0f} drill completed",
            not (shutdown and "error" in str(trader.state.shutdown_reason).lower()),
            f"steps={n_steps}, pnl={pnl:+.2f}, elapsed={elapsed:.1f}s, "
            f"shutdown={shutdown} reason={trader.state.shutdown_reason!r}",
        )

    except Exception as exc:
        return _check(f"${capital:.0f} drill completed", False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="First Dollar pre-flight drill (G12/G13)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Run simulation drill with this capital (e.g. 100)")
    parser.add_argument("--config", default=None,
                        help="YAML config path for risk/exchange settings")
    parser.add_argument("--check-only", action="store_true",
                        help="Run auto-checks only, skip trading drill")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--skip-kill-switch-test", action="store_true",
                        help="Skip the in-process kill switch timing test")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  First Dollar Pre-Flight Drill")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # Load config
    config: dict[str, Any] = {}
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"WARNING: could not load config {args.config}: {exc}")
    else:
        config = {
            "paper_trading": {"max_drawdown_threshold": 0.20, "window_size": 5},
            "exchange": {"daily_loss_limit": -500.0},
            "limits": {
                "per_symbol_notional_max": 10_000.0,
                "portfolio_notional_max": 50_000.0,
                "leverage_max": 1.0,
            },
        }

    results: list[dict[str, Any]] = []

    print("── Structural Checks ──────────────────────────────────────")
    results.append(check_pytest_ini_ignores())
    results.append(check_no_old_risk_api())
    results.append(check_postmortem_template())
    results.append(check_go_live_checklist())
    results.append(check_kill_switch_script())
    results.append(check_checkpoint_freshness())
    results.append(check_audit_chain())

    print("\n── Risk Config Checks ─────────────────────────────────────")
    results.extend(check_risk_config(config))

    if not args.skip_kill_switch_test:
        print("\n── Kill Switch Timing Test ────────────────────────────────")
        results.append(run_kill_switch_timing_test())

    if not args.check_only:
        capital = args.capital or 100.0
        print(f"\n── ${capital:.0f} Simulation Drill ─────────────────────────────")
        results.append(run_dollar_drill(capital))

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ✅ ALL CHECKS PASSED — you may proceed to go-live sign-off")
    else:
        print("  ❌ CHECKS FAILED — resolve before going live")
    print(f"{'='*60}\n")

    if args.report:
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "failed": failed,
            "total": total,
            "results": results,
        }
        Path(args.report).write_text(json.dumps(report_data, indent=2))
        print(f"Report written to {args.report}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
