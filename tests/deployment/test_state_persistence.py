"""
Phase 6 Week 56 (S4): Idempotent restart tests for PaperTrader StateStore.

Scenario:
    1. Run PaperTrader for 50 steps with a fixed deterministic agent.
    2. Drop the trader object (simulating crash).
    3. PaperTrader.restore() from the SQLite StateStore.
    4. Run 50 more steps over the same continuation prices.
    5. Compare against a baseline trader that ran 100 steps fresh; cash,
       position, portfolio_value, peak_value, num_trades and shutdown state
       must match exactly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import math
import numpy as np
import pytest

from deployment.paper_trader import PaperTrader, TradingState
from deployment.persistence.state_store import StateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(action: float = 0.3):
    agent = MagicMock()
    agent.predict.return_value = (np.array([action], dtype=np.float32), None)
    return agent


def _make_config(**overrides):
    cfg = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.99,  # don't shut down during the test
            "window_size": 10,
            "daily_report_interval": 999_999,
            "poll_interval_seconds": 1.0,
        }
    }
    cfg["paper_trading"].update(overrides)
    return cfg


def _prices(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    # mild upward drift so the trader actually buys and accumulates
    return (50_000.0 + np.cumsum(rng.normal(50.0, 100.0, n))).tolist()


def _snapshot(trader: PaperTrader) -> dict:
    return {
        "cash": round(trader.state.balance, 6),
        "position": round(trader.state.position, 8),
        "entry_price": round(trader.state.entry_price, 6),
        "peak_value": round(trader.state.peak_portfolio_value, 6),
        "portfolio_value": round(trader.state.portfolio_value, 6),
        "num_trades": len(trader.state.trades),
        "step": trader.state.step,
        "shutdown": trader.state.shutdown_triggered,
    }


# ---------------------------------------------------------------------------
# StateStore unit
# ---------------------------------------------------------------------------

def test_state_store_roundtrip(tmp_path: Path):
    store = StateStore(str(tmp_path / "s.db"))
    assert store.load_latest() is None

    snap = {
        "symbol": "BTC/USDT",
        "position": 0.123,
        "entry_price": 50_000.0,
        "cash": 4_321.0,
        "current_price": 51_000.0,
        "peak_value": 10_500.0,
        "equity": 4_321.0 + 0.123 * 51_000.0,
        "step": 7,
        "shutdown_triggered": False,
        "shutdown_reason": "",
        "portfolio_history": [10_000.0, 10_100.0],
        "trades": [],
        "orders": [],
    }
    store.save_snapshot(snap)
    out = store.load_latest()
    assert out["position"] == pytest.approx(0.123)
    assert out["cash"] == pytest.approx(4_321.0)
    assert out["step"] == 7
    assert out["portfolio_history"] == [10_000.0, 10_100.0]

    store.clear()
    assert store.load_latest() is None
    store.close()


def test_state_store_rejects_nonfinite(tmp_path: Path):
    store = StateStore(str(tmp_path / "s.db"))
    with pytest.raises(ValueError):
        store.save_snapshot({"cash": float("nan"), "equity": 0.0})
    store.close()


# ---------------------------------------------------------------------------
# TradingState round-trip
# ---------------------------------------------------------------------------

def test_trading_state_to_from_dict():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    for p in _prices(30):
        trader._update_price(p)
        trader._execute_action(np.array([0.3]), p)
        trader.state.step += 1

    d = trader.state.to_dict(symbol="BTC/USDT")
    restored = TradingState.from_dict(d)
    assert restored.balance == pytest.approx(trader.state.balance)
    assert restored.position == pytest.approx(trader.state.position)
    assert restored.peak_portfolio_value == pytest.approx(
        trader.state.peak_portfolio_value
    )
    assert restored.step == trader.state.step
    assert len(restored.trades) == len(trader.state.trades)


# ---------------------------------------------------------------------------
# End-to-end idempotent restart
# ---------------------------------------------------------------------------

def test_paper_trader_idempotent_restart(tmp_path: Path):
    prices = _prices(100)
    cfg = _make_config()

    # ---- Baseline: full 100-step run without persistence -----------------
    baseline = PaperTrader(_make_agent(), cfg, simulation_mode=True)
    baseline.run(price_stream=iter(prices))
    baseline_snap = _snapshot(baseline)

    # ---- Split run: 50 steps -> persist -> drop -> restore -> 50 more ----
    db_path = tmp_path / "trader.db"
    store_a = StateStore(str(db_path))
    trader_a = PaperTrader(
        _make_agent(), cfg, simulation_mode=True, state_store=store_a
    )
    trader_a.run(price_stream=iter(prices[:50]))
    mid_step = trader_a.state.step
    assert mid_step == 50
    store_a.close()

    # Simulate crash
    del trader_a

    # Restore in a new process / object
    store_b = StateStore(str(db_path))
    trader_b = PaperTrader.restore(
        state_store=store_b,
        agent=_make_agent(),
        config=cfg,
        simulation_mode=True,
    )
    assert trader_b.state.step == 50
    trader_b.run(price_stream=iter(prices[50:]))
    split_snap = _snapshot(trader_b)
    store_b.close()

    # ---- Compare ---------------------------------------------------------
    assert split_snap["step"] == baseline_snap["step"] == 100
    assert split_snap["num_trades"] == baseline_snap["num_trades"]
    assert split_snap["cash"] == pytest.approx(baseline_snap["cash"], rel=1e-9, abs=1e-6)
    assert split_snap["position"] == pytest.approx(
        baseline_snap["position"], rel=1e-9, abs=1e-10
    )
    assert split_snap["peak_value"] == pytest.approx(
        baseline_snap["peak_value"], rel=1e-9, abs=1e-6
    )
    assert split_snap["portfolio_value"] == pytest.approx(
        baseline_snap["portfolio_value"], rel=1e-9, abs=1e-6
    )
    assert split_snap["shutdown"] == baseline_snap["shutdown"]


def test_paper_trader_persistence_disabled_by_default():
    """Backwards-compat: no persistence config => no StateStore created."""
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    assert trader.state_store is None


def test_paper_trader_persistence_block_creates_store(tmp_path: Path):
    cfg = _make_config()
    cfg["persistence"] = {
        "enabled": True,
        "db_path": str(tmp_path / "auto.db"),
        "checkpoint_every_n_steps": 1,
    }
    trader = PaperTrader(_make_agent(), cfg, simulation_mode=True)
    assert trader.state_store is not None
    trader.run(price_stream=iter(_prices(15)))
    assert (tmp_path / "auto.db").exists()
    snap = trader.state_store.load_latest()
    assert snap is not None
    assert snap["step"] == 15
    trader.state_store.close()
