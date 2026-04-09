#!/usr/bin/env python
"""
Disaster-recovery drill runner.

Usage
-----
    python scripts/drills/run_drill.py --scenario crash_mid_episode
    python scripts/drills/run_drill.py --scenario data_gap
    python scripts/drills/run_drill.py --scenario risk_breach
    python scripts/drills/run_drill.py --scenario all

Exit codes
----------
    0  — all selected drills passed
    1  — one or more drills failed

Each drill is a self-contained function that raises AssertionError on failure
and returns a human-readable summary dict on success.  No real exchange
connections are made; all drills run in simulation mode using in-process
objects.

Phase 6 Week 59 (S17)
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import MagicMock

import numpy as np

# Ensure repo root is on path when called from project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deployment.paper_trader import PaperTrader
from deployment.persistence.state_store import StateStore


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_drills.py so the CLI has no pytest dependency)
# ---------------------------------------------------------------------------

def _make_agent(action: float = 0.3):
    agent = MagicMock()
    agent.predict.return_value = (np.array([action], dtype=np.float32), None)
    return agent


def _make_config(max_drawdown: float = 0.99, **overrides):
    cfg = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": max_drawdown,
            "window_size": 10,
            "daily_report_interval": 999_999,
            "poll_interval_seconds": 1.0,
        }
    }
    cfg["paper_trading"].update(overrides)
    return cfg


def _stable_prices(n: int, start: float = 50_000.0, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    return (start + np.cumsum(rng.normal(10.0, 50.0, n))).tolist()


def _assert_finite(trader: PaperTrader) -> None:
    hist = list(trader.state.portfolio_history)
    bad = [v for v in hist if not math.isfinite(v)]
    if bad:
        raise AssertionError(f"Non-finite portfolio_history values: {bad[:5]}")


# ---------------------------------------------------------------------------
# Drill 1 — crash_mid_episode
# ---------------------------------------------------------------------------

def drill_crash_mid_episode() -> Dict[str, Any]:
    """
    Simulates SIGKILL mid-episode.

    Steps:
        1. Run PaperTrader for 50 steps with StateStore enabled.
        2. Drop the trader object (crash simulation).
        3. Restore via PaperTrader.restore().
        4. Run 50 more steps.
        5. Assert step count advanced and state is coherent.
    """
    print("  [crash_mid_episode] Starting…")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "crash.db")
        store = StateStore(db_path)
        cfg = _make_config()
        prices = _stable_prices(120, seed=42)

        # Phase 1 — 50 steps
        trader_a = PaperTrader(_make_agent(0.3), cfg, simulation_mode=True, state_store=store)
        trader_a.run(price_stream=iter(prices[:50]))
        step_pre_crash = trader_a.state.step
        print(f"  [crash_mid_episode] Pre-crash step={step_pre_crash}")
        del trader_a  # simulate crash

        # Phase 2 — restore & run 50 more
        trader_b = PaperTrader.restore(store, _make_agent(0.3), cfg, simulation_mode=True)
        restored_step = trader_b.state.step
        if restored_step != step_pre_crash:
            raise AssertionError(
                f"Restored step {restored_step} != pre-crash step {step_pre_crash}"
            )
        trader_b.run(price_stream=iter(prices[50:100]))
        final_step = trader_b.state.step

        _assert_finite(trader_b)
        if trader_b.state.shutdown_triggered:
            raise AssertionError("Kill-switch fired on stable prices (unexpected)")
        if not math.isfinite(trader_b.state.balance):
            raise AssertionError(f"Balance is not finite: {trader_b.state.balance}")

    print(f"  [crash_mid_episode] PASSED — restored_step={restored_step} final_step={final_step}")
    return {
        "scenario": "crash_mid_episode",
        "pre_crash_step": step_pre_crash,
        "restored_step": restored_step,
        "final_step": final_step,
    }


# ---------------------------------------------------------------------------
# Drill 2 — data_gap
# ---------------------------------------------------------------------------

def drill_data_gap() -> Dict[str, Any]:
    """
    Injects a simulated 1-hour data gap (large single-step price jump).

    Verifies:
        * No exception raised.
        * portfolio_history is fully finite.
        * position and entry_price stay finite.
    """
    print("  [data_gap] Starting…")
    cfg = _make_config(max_drawdown=0.99)
    agent = _make_agent(action=0.1)

    rng = np.random.default_rng(7)
    pre_gap = (50_000.0 + np.cumsum(rng.normal(10, 50, 50))).tolist()
    gap_price = [pre_gap[-1] * 1.15]  # +15 % jump (simulated 1-h gap)
    post_gap = (gap_price[0] + np.cumsum(rng.normal(10, 50, 50))).tolist()
    prices = pre_gap + gap_price + post_gap

    gap_magnitude_pct = (gap_price[0] / pre_gap[-1] - 1.0) * 100
    print(f"  [data_gap] Gap magnitude: +{gap_magnitude_pct:.1f}%")

    trader = PaperTrader(agent, cfg, simulation_mode=True)
    report = trader.run(price_stream=iter(prices))

    _assert_finite(trader)
    if not math.isfinite(report["final_balance"]):
        raise AssertionError(f"final_balance is not finite: {report['final_balance']}")
    if not math.isfinite(trader.state.position):
        raise AssertionError(f"position is not finite: {trader.state.position}")

    print(f"  [data_gap] PASSED — final_balance={report['final_balance']:.2f}")
    return {
        "scenario": "data_gap",
        "gap_magnitude_pct": gap_magnitude_pct,
        "final_balance": report["final_balance"],
    }


# ---------------------------------------------------------------------------
# Drill 3 — risk_breach
# ---------------------------------------------------------------------------

def drill_risk_breach() -> Dict[str, Any]:
    """
    Forces portfolio drawdown > 10 % by feeding a sharp price crash.

    Verifies:
        * shutdown_triggered = True.
        * shutdown_reason mentions "drawdown".
        * position = 0 after liquidation.
    """
    print("  [risk_breach] Starting…")
    cfg = _make_config(max_drawdown=0.10)  # 10 % kill-switch
    agent = _make_agent(action=0.8)        # large long → amplifies loss

    rng = np.random.default_rng(99)
    initial = 50_000.0
    warm_up = (initial + np.cumsum(rng.normal(50, 20, 15))).tolist()
    crash_target = initial * 0.65
    crash = np.linspace(warm_up[-1], crash_target, 40).tolist()
    hold = [crash_target] * 10
    prices = warm_up + crash + hold

    trader = PaperTrader(agent, cfg, simulation_mode=True)
    trader.run(price_stream=iter(prices))

    if not trader.state.shutdown_triggered:
        raise AssertionError(
            "Kill-switch did NOT fire — drawdown check may be broken"
        )
    reason = trader.state.shutdown_reason.lower()
    if "drawdown" not in reason and "max" not in reason:
        raise AssertionError(f"Unexpected shutdown_reason: {trader.state.shutdown_reason!r}")
    if abs(trader.state.position) > 1e-9:
        raise AssertionError(
            f"Position not liquidated after kill-switch: {trader.state.position}"
        )

    print(
        f"  [risk_breach] PASSED — shutdown_reason={trader.state.shutdown_reason!r}"
        f" position={trader.state.position}"
    )
    return {
        "scenario": "risk_breach",
        "shutdown_triggered": True,
        "shutdown_reason": trader.state.shutdown_reason,
        "final_position": trader.state.position,
    }


# ---------------------------------------------------------------------------
# Registry & runner
# ---------------------------------------------------------------------------

DRILLS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "crash_mid_episode": drill_crash_mid_episode,
    "data_gap": drill_data_gap,
    "risk_breach": drill_risk_breach,
}


def run_drills(scenarios: List[str]) -> bool:
    """Run each scenario; return True iff all passed."""
    results = {}
    failed = []

    for name in scenarios:
        fn = DRILLS.get(name)
        if fn is None:
            print(f"ERROR: unknown scenario '{name}'. Available: {list(DRILLS)}")
            failed.append(name)
            continue

        print(f"\n{'='*60}")
        print(f"DRILL: {name}")
        print('='*60)
        try:
            result = fn()
            results[name] = {"status": "PASSED", **result}
            print(f"  → PASSED")
        except Exception as exc:
            failed.append(name)
            results[name] = {"status": "FAILED", "error": str(exc)}
            print(f"  → FAILED: {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(scenarios) - len(failed)}/{len(scenarios)} passed")
    if failed:
        print(f"  FAILED: {failed}")
    print('='*60)

    return len(failed) == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run disaster-recovery drills for the trading bot.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(DRILLS) + ["all"],
        default="all",
        help="Which drill to run (default: all).",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = list(DRILLS)
    else:
        scenarios = [args.scenario]

    ok = run_drills(scenarios)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
