"""
Phase 6 Week 59 (S17): Disaster-recovery drill tests.

Three scenarios exercised as pytest tests so they can be run in CI (nightly
job) and also called from scripts/drills/run_drill.py.

Scenarios
---------
1. crash_mid_episode  — SIGKILL simulation: run 50 steps, drop trader object,
                        restore from StateStore, run 50 more; verify step
                        count and shutdown state are coherent.

2. data_gap           — 1-hour gap in price feed: inject a large price jump
                        simulating a 1-hour data gap; trader must survive
                        (no crash, no NaN state, portfolio_history stays finite).

3. risk_breach        — Force drawdown > 10%: feed prices that crash the
                        portfolio; kill-switch must fire (shutdown_triggered=True)
                        and position must be liquidated.

Each test is self-contained: uses only in-process objects (no processes, no
networking, no files beyond a tmp SQLite DB for crash_mid_episode).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.paper_trader import PaperTrader, TradingState
from deployment.persistence.state_store import StateStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_agent(action: float = 0.3):
    """Deterministic stub agent that always returns the same action."""
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


def _stable_prices(n: int, start: float = 50_000.0, seed: int = 42) -> list[float]:
    """Mild upward-trending prices — won't trigger a 10 % drawdown."""
    rng = np.random.default_rng(seed)
    return (start + np.cumsum(rng.normal(10.0, 50.0, n))).tolist()


def _assert_all_finite(trader: PaperTrader) -> None:
    """Assert that portfolio_history contains no NaN or inf."""
    hist = list(trader.state.portfolio_history)
    bad = [v for v in hist if not math.isfinite(v)]
    assert not bad, f"Non-finite values in portfolio_history: {bad[:5]}"


# ---------------------------------------------------------------------------
# Drill 1 — crash_mid_episode
# ---------------------------------------------------------------------------

class TestDrillCrashMidEpisode:
    """SIGKILL simulation: drop trader mid-run, restore, resume."""

    def test_restore_resumes_at_correct_step(self, tmp_path):
        db_path = str(tmp_path / "test_crash.db")
        store = StateStore(db_path)
        cfg = _make_config()
        agent = _make_agent(action=0.3)

        prices = _stable_prices(120, seed=42)

        # --- Phase 1: run 50 steps ---
        trader_a = PaperTrader(
            agent, cfg, simulation_mode=True, state_store=store
        )
        trader_a.run(price_stream=iter(prices[:50]))
        step_after_phase1 = trader_a.state.step
        assert step_after_phase1 > 0, "Trader must have advanced steps"

        # --- Simulate crash: drop the object ---
        del trader_a

        # --- Phase 2: restore from StateStore and run 50 more steps ---
        agent2 = _make_agent(action=0.3)
        trader_b = PaperTrader.restore(
            store, agent2, cfg, simulation_mode=True
        )
        restored_step = trader_b.state.step
        assert restored_step == step_after_phase1, (
            f"Restored step {restored_step} != pre-crash step {step_after_phase1}"
        )

        trader_b.run(price_stream=iter(prices[50:100]))
        final_step = trader_b.state.step
        assert final_step > restored_step, "Trader must advance after restore"

        # --- State sanity ---
        _assert_all_finite(trader_b)
        assert not trader_b.state.shutdown_triggered, (
            "Kill-switch must NOT fire on stable prices"
        )
        assert math.isfinite(trader_b.state.balance)

    def test_empty_store_yields_fresh_trader(self, tmp_path):
        """restore() on an empty StateStore returns a fresh trader at step 0."""
        db_path = str(tmp_path / "empty.db")
        store = StateStore(db_path)
        cfg = _make_config()
        trader = PaperTrader.restore(store, _make_agent(), cfg, simulation_mode=True)
        assert trader.state.step == 0


# ---------------------------------------------------------------------------
# Drill 2 — data_gap
# ---------------------------------------------------------------------------

class TestDrillDataGap:
    """
    Inject a 1-hour price gap: prices jump by a factor of ~2× in one step.
    The trader must survive (no exception, no NaN state, finite portfolio).
    """

    def test_large_gap_does_not_crash(self):
        cfg = _make_config(max_drawdown=0.99)
        agent = _make_agent(action=0.1)  # tiny position so gap doesn't kill the portfolio

        # Build a price sequence:  50 normal prices, 1 big gap, 50 normal prices
        rng = np.random.default_rng(7)
        pre_gap = (50_000.0 + np.cumsum(rng.normal(10, 50, 50))).tolist()
        gap_price = [pre_gap[-1] * 1.15]   # +15 % jump (1-hour gap in crypto)
        post_gap = (gap_price[0] + np.cumsum(rng.normal(10, 50, 50))).tolist()
        prices = pre_gap + gap_price + post_gap

        trader = PaperTrader(agent, cfg, simulation_mode=True)
        report = trader.run(price_stream=iter(prices))

        # Must complete without exception
        assert report is not None
        _assert_all_finite(trader)
        assert math.isfinite(report["final_balance"]), "Balance must be finite after gap"

    def test_gap_does_not_corrupt_position(self):
        """After a price gap, position and entry_price stay finite."""
        cfg = _make_config(max_drawdown=0.99)
        agent = _make_agent(action=0.5)

        rng = np.random.default_rng(13)
        pre = (50_000.0 + np.cumsum(rng.normal(10, 30, 30))).tolist()
        gap = [pre[-1] * 0.80]  # -20 % crash (downward gap)
        post = (gap[0] + np.cumsum(rng.normal(5, 30, 30))).tolist()
        prices = pre + gap + post

        trader = PaperTrader(agent, cfg, simulation_mode=True)
        trader.run(price_stream=iter(prices))

        assert math.isfinite(trader.state.position)
        assert math.isfinite(trader.state.entry_price)
        assert math.isfinite(trader.state.balance)


# ---------------------------------------------------------------------------
# Drill 3 — risk_breach
# ---------------------------------------------------------------------------

class TestDrillRiskBreach:
    """
    Force portfolio drawdown > 10 %.  Kill-switch must fire:
    * shutdown_triggered = True
    * position liquidated (position ≈ 0)
    """

    def _make_crashing_prices(self, initial: float = 50_000.0, n_pre: int = 15) -> list[float]:
        """
        Warm-up N prices near initial, then a sharp crash that guarantees
        a >10 % drop in portfolio value for a long-biased agent.

        The agent holds a long position from the warm-up phase.  When price
        collapses to 70 % of initial the portfolio (cash + pos * price) drops
        well past 10 %.
        """
        rng = np.random.default_rng(99)
        warm_up = (initial + np.cumsum(rng.normal(50, 20, n_pre))).tolist()
        # Aggressive crash: 35 % drop from peak
        crash_target = initial * 0.65
        crash = np.linspace(warm_up[-1], crash_target, 40).tolist()
        # Hold at low price so drawdown check fires multiple times
        hold = [crash_target] * 10
        return warm_up + crash + hold

    def test_kill_switch_fires_on_deep_drawdown(self):
        """Kill-switch must fire when drawdown >= 10 %."""
        # max_drawdown_threshold = 0.10 → fires at 10 % loss
        cfg = _make_config(max_drawdown=0.10)
        agent = _make_agent(action=0.8)  # large long position → amplifies loss

        prices = self._make_crashing_prices()

        trader = PaperTrader(agent, cfg, simulation_mode=True)
        trader.run(price_stream=iter(prices))

        assert trader.state.shutdown_triggered, (
            "Kill-switch must fire when drawdown >= 10 %"
        )
        assert "drawdown" in trader.state.shutdown_reason.lower() or \
               "max drawdown" in trader.state.shutdown_reason.lower(), (
            f"Unexpected shutdown_reason: {trader.state.shutdown_reason!r}"
        )

    def test_position_liquidated_after_kill_switch(self):
        """After kill-switch, position must be zero (trader liquidated)."""
        cfg = _make_config(max_drawdown=0.10)
        agent = _make_agent(action=0.8)

        prices = self._make_crashing_prices()
        trader = PaperTrader(agent, cfg, simulation_mode=True)
        trader.run(price_stream=iter(prices))

        # The _trigger_shutdown() method calls _execute_sell(1.0, ...) which
        # liquidates the full position.
        assert trader.state.position == pytest.approx(0.0, abs=1e-9), (
            f"Position must be zero after liquidation, got {trader.state.position}"
        )

    def test_no_further_trades_after_shutdown(self):
        """No trades should be recorded after the kill-switch step."""
        cfg = _make_config(max_drawdown=0.10)
        agent = _make_agent(action=0.8)

        prices = self._make_crashing_prices()
        trader = PaperTrader(agent, cfg, simulation_mode=True)
        trader.run(price_stream=iter(prices))

        assert trader.state.shutdown_triggered
        # Find the shutdown trade (liquidation sell) and confirm no trades after
        shutdown_idx = None
        for i, t in enumerate(trader.state.trades):
            if t.side == "sell":
                shutdown_idx = i
                break
        # After the shutdown sell there must be no more buy trades
        post_shutdown_buys = [
            t for t in trader.state.trades[shutdown_idx + 1:]
            if t.side == "buy"
        ] if shutdown_idx is not None else []
        assert not post_shutdown_buys, (
            f"Found {len(post_shutdown_buys)} buy trades after shutdown"
        )
