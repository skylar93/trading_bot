"""
Week 16 Tests: PaperTrader (deployment/paper_trader.py)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.paper_trader import PaperTrader, TradingState, Trade
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_agent(action=0.3):
    """Minimal agent stub that always returns the same action."""
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
            "max_drawdown_threshold": 0.20,
            "window_size": 10,
            "daily_report_interval": 999999,
            "poll_interval_seconds": 1.0,
        }
    }
    cfg["paper_trading"].update(overrides)
    return cfg


def _prices(n=50, start=50_000.0, drift=100.0, seed=42):
    rng = np.random.default_rng(seed)
    return (start + np.cumsum(rng.normal(drift, 200, n))).tolist()


# ---------------------------------------------------------------------------
# Test 1: Initialization
# ---------------------------------------------------------------------------

def test_paper_trader_initialization():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    assert trader.initial_balance == 10_000.0
    assert trader.trading_fee == 0.001
    assert trader.simulation_mode is True
    assert trader.state.balance == 10_000.0
    assert trader.state.position == 0.0


# ---------------------------------------------------------------------------
# Test 2: Simulation mode skips CCXT
# ---------------------------------------------------------------------------

def test_simulation_mode_no_ccxt():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    assert trader._exchange is None


# ---------------------------------------------------------------------------
# Test 3: Action execution – buy
# ---------------------------------------------------------------------------

def test_action_execution_buy():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    initial_balance = trader.state.balance
    trader._update_price(50_000.0)
    trader._execute_action(np.array([0.5]), 50_000.0)
    assert trader.state.position > 0
    assert trader.state.balance < initial_balance


# ---------------------------------------------------------------------------
# Test 4: Action execution – sell reduces position
# ---------------------------------------------------------------------------

def test_action_execution_sell():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_action(np.array([0.8]), 50_000.0)
    pos_after_buy = trader.state.position
    assert pos_after_buy > 0

    trader._update_price(51_000.0)
    trader._execute_action(np.array([-0.8]), 51_000.0)
    assert trader.state.position < pos_after_buy


# ---------------------------------------------------------------------------
# Test 5: Hold (deadband)
# ---------------------------------------------------------------------------

def test_action_execution_hold():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_action(np.array([0.02]), 50_000.0)  # within deadband
    assert trader.state.position == 0.0
    assert trader.state.balance == trader.initial_balance


# ---------------------------------------------------------------------------
# Test 6: Max position size is respected
# ---------------------------------------------------------------------------

def test_position_limits_enforced():
    cfg = _make_config(max_position_size=0.1)
    trader = PaperTrader(_make_agent(), cfg, simulation_mode=True)
    trader._update_price(50_000.0)
    # Even with action=1.0, only 10% of balance should be spent
    trader._execute_buy(1.0, 50_000.0)
    spent = trader.initial_balance - trader.state.balance
    # fee included; spent ≤ 10% * balance * (1 + fee)
    assert spent <= trader.initial_balance * 0.1 * 1.01 + 1e-6


# ---------------------------------------------------------------------------
# Test 7: Max drawdown triggers shutdown
# ---------------------------------------------------------------------------

def test_max_drawdown_shutdown():
    cfg = _make_config(max_drawdown_threshold=0.10)
    trader = PaperTrader(_make_agent(), cfg, simulation_mode=True)
    # Manually inject a portfolio history that shows 15% drawdown
    trader.state.peak_portfolio_value = 10_000.0
    trader.state.portfolio_history = [10_000.0, 8_500.0]  # 15% drop
    trader.state._current_price = 49_000.0
    trader._check_risk(49_000.0)
    assert trader.state.shutdown_triggered is True
    assert "drawdown" in trader.state.shutdown_reason.lower()


# ---------------------------------------------------------------------------
# Test 8: Trading fee deducted from balance
# ---------------------------------------------------------------------------

def test_trading_fee_deducted():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_buy(0.1, 50_000.0)  # spend 10% of balance
    expected_spend = trader.initial_balance * 0.1
    expected_fee = expected_spend * trader.trading_fee
    # balance should be roughly initial - spend - fee
    loss = trader.initial_balance - trader.state.balance
    assert abs(loss - (expected_spend + expected_fee)) < 1e-3


# ---------------------------------------------------------------------------
# Test 9: P&L tracked across buy-sell round trip
# ---------------------------------------------------------------------------

def test_pnl_tracking():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_buy(0.5, 50_000.0)  # spend 50% so fee won't overflow
    assert trader.state.position > 0, "Buy should have executed"
    # Sell at higher price
    trader._update_price(55_000.0)
    trader._execute_sell(1.0, 55_000.0)
    sell_trades = [t for t in trader.state.trades if t.side == "sell"]
    assert len(sell_trades) == 1
    assert sell_trades[0].pnl > 0


# ---------------------------------------------------------------------------
# Test 10: generate_report – structure
# ---------------------------------------------------------------------------

def test_generate_report_structure():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader.run(price_stream=iter(_prices(30)))
    report = trader.generate_report()
    required_keys = {
        "total_return", "sharpe_ratio", "max_drawdown",
        "num_trades", "final_balance", "final_portfolio_value",
        "shutdown_triggered", "win_rate", "avg_trade_pnl", "total_fees",
    }
    assert required_keys.issubset(report.keys())


# ---------------------------------------------------------------------------
# Test 11: generate_report – metric ranges
# ---------------------------------------------------------------------------

def test_generate_report_metrics():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader.run(price_stream=iter(_prices(40)))
    report = trader.generate_report()
    assert 0.0 <= report["max_drawdown"] <= 1.0
    assert 0.0 <= report["win_rate"] <= 1.0
    assert report["num_trades"] >= 0
    assert report["total_fees"] >= 0.0


# ---------------------------------------------------------------------------
# Test 12: Checkpoint save / load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_save_load():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader.run(price_stream=iter(_prices(20)))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    trader.save_checkpoint(path)

    # Load into new trader and compare key state
    trader2 = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader2.load_checkpoint(path)
    assert abs(trader2.state.balance - trader.state.balance) < 1e-6
    assert trader2.state.position == pytest.approx(trader.state.position)
    assert len(trader2.state.trades) == len(trader.state.trades)
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 13: run() returns report dict
# ---------------------------------------------------------------------------

def test_run_with_price_stream():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    report = trader.run(price_stream=iter(_prices(30)))
    assert isinstance(report, dict)
    assert "total_return" in report


# ---------------------------------------------------------------------------
# Test 14: Sell with no position is a no-op
# ---------------------------------------------------------------------------

def test_sell_with_no_position_is_noop():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_sell(1.0, 50_000.0)
    assert trader.state.balance == trader.initial_balance
    assert len(trader.state.trades) == 0


# ---------------------------------------------------------------------------
# Test 15: Shutdown liquidates position
# ---------------------------------------------------------------------------

def test_shutdown_liquidates_position():
    trader = PaperTrader(_make_agent(), _make_config(), simulation_mode=True)
    trader._update_price(50_000.0)
    trader._execute_buy(0.5, 50_000.0)
    assert trader.state.position > 0

    trader._trigger_shutdown("test shutdown")
    assert trader.state.shutdown_triggered is True
    assert trader.state.position == 0.0


# ---------------------------------------------------------------------------
# Test 16: Context manager cleans up
# ---------------------------------------------------------------------------

def test_context_manager():
    with PaperTrader(
        _make_agent(), _make_config(), simulation_mode=True
    ) as trader:
        assert trader.initial_balance == 10_000.0
    # Should exit without error


# ---------------------------------------------------------------------------
# Test 17: MLflow manager receives step metrics
# ---------------------------------------------------------------------------

def test_mlflow_logging():
    mock_mlflow = MagicMock()
    trader = PaperTrader(
        _make_agent(), _make_config(), mlflow_manager=mock_mlflow, simulation_mode=True
    )
    trader.run(price_stream=iter(_prices(15)))
    # log_metric should have been called with portfolio_value at least once
    calls = [c for c in mock_mlflow.log_metric.call_args_list
              if c.args and c.args[0] == "portfolio_value"]
    assert len(calls) >= 1
