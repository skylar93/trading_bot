"""
Integration Test with Agents: uses a BacktestManager(settings) 
with multiple agent names (Dummy, MeanReversion, etc.), checks 
logs for errors, ensures valid trades and final portfolio. 
Tests the agent-backtest pipeline from the manager's perspective.
Comprehensive Integration Tests for Backtesting with Different Agents
---------------------------------------------------------------------

This test suite checks:
1) That each agent returns valid numeric actions during the backtest
2) That the backtest logs do not contain "Error getting action..." messages
3) That we get a valid trades list (with numeric amounts, prices, etc.)
4) That final results have a non-empty portfolio_values array
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gymnasium as gym
import re

from agents.strategies.agent_factory import create_agent
from deployment.web_interface.utils.backtest import BacktestManager
from training.backtesting.risk_manager import RiskConfig

#####################
# TEST DATA HELPERS #
#####################

def create_test_data(periods: int = 100) -> pd.DataFrame:
    """
    Create synthetic market data (OHLCV) with some trend + mean reversion noise.
    """
    dates = pd.date_range(start=datetime.now(), periods=periods, freq='1min')
    data = pd.DataFrame(index=dates)
    
    # Generate sample price data
    base_price = 100.0
    noise = np.random.normal(0, 0.001, periods)
    trend = np.linspace(0, 0.1, periods)  # slight upward trend
    mean_rev = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.05
    prices = base_price * (1 + noise + trend + mean_rev)
    
    data["$open"] = prices * (1 + np.random.normal(0, 0.0002, periods))
    data["$high"] = data["$open"] * (1 + abs(np.random.normal(0, 0.001, periods)))
    data["$low"] = data["$open"] * (1 - abs(np.random.normal(0, 0.001, periods)))
    data["$close"] = prices
    data["$volume"] = np.random.lognormal(10, 1, periods)
    
    return data

def create_test_settings(agent_name: str) -> dict:
    """
    Create test BacktestManager settings.
    If agent is Dummy, we skip observation_space & action_space config.
    """
    if agent_name == "Dummy":
        return {
            "agent_name": agent_name,
            # agent_config is empty or minimal for Dummy
            "agent_config": {},
            "max_position_size": 100,
            "stop_loss": 10,
            "initial_balance": 10000,
            "trading_fee": 0.001
        }
    else:
        return {
            "agent_name": agent_name,
            "agent_config": {
                "observation_space": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(20, 5), dtype=np.float32
                ),
                "action_space": gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(1,), dtype=np.float32
                ),
                "learning_rate": 0.001,
                "batch_size": 64,
                "rsi_window": 14,
                "bb_window": 20,
                "momentum_window": 10
            },
            "max_position_size": 100,
            "stop_loss": 10,
            "initial_balance": 10000,
            "trading_fee": 0.001
        }

###########################
# ACTUAL TESTS START HERE #
###########################

@pytest.mark.parametrize("agent_name", ["Dummy", "MeanReversion", "Momentum", "PPO"])
def test_backtest_integration(agent_name, caplog):
    """
    Single test function (parametrized) to test each agent in a real backtest scenario.
    
    Steps:
    1) Create data & settings
    2) Initialize BacktestManager
    3) Run backtest
    4) Check logs to ensure no "Error getting action..." lines
    5) Check results structure (portfolio_values, trades, etc.)
    6) Check trades for numeric, valid action ([-1, 1]) 
       and ensure trade amount is numeric, with sign matching trade type
    """
    # Skip PPO test if we're using real agents
    if agent_name == "PPO":
        try:
            from agents.strategies.agent_factory import USE_REAL_AGENTS
            if USE_REAL_AGENTS:
                pytest.skip(f"Skipping backtest integration test with {agent_name} agent")
        except ImportError:
            pass
    
    # (1) Create test data & settings
    data = create_test_data(120)  # 120 bars
    settings = create_test_settings(agent_name)
    
    # (2) Initialize Manager
    manager = BacktestManager(settings)
    
    # (3) Run the backtest
    with caplog.at_level("ERROR"):
        results = manager.run_backtest(data)
    
    # (4) Check logs for "Error getting action..." or "Error getting action from strategy"
    error_logs = []
    for record in caplog.records:
        if "Error getting action" in record.message:
            error_logs.append(record.message)
    
    # If any such error logs exist, we fail
    assert not error_logs, (
        f"Found 'Error getting action...' logs for agent '{agent_name}':\n" 
        + "\n".join(error_logs)
    )
    
    # (5) Check results structure
    assert "portfolio_values" in results, f"Missing portfolio_values in results for {agent_name}"
    assert len(results["portfolio_values"]) > 0, f"portfolio_values is empty for {agent_name}"
    assert "trades" in results, f"Missing trades in results for {agent_name}"
    assert isinstance(results["trades"], list), f"trades is not a list for {agent_name}"
    
    # (6) Validate trades content
    for trade in results["trades"]:
        # Minimal fields check
        assert "action" in trade, f"No 'action' field in trade: {trade}"
        assert "amount" in trade, f"No 'amount' field in trade: {trade}"
        assert "price" in trade, f"No 'price' field in trade: {trade}"
        assert "type" in trade, f"No 'type' field in trade: {trade}"
        
        # Check action is numeric & in [-1,1]
        assert isinstance(trade["action"], (int, float, np.ndarray)), \
            f"Trade action is not numeric or array: {trade['action']}"
        action_val = float(trade["action"])  # convert if it's ndarray
        assert -1.0 <= action_val <= 1.0, f"Agent action {action_val} not in [-1,1]"
        
        # Check amount, price are numeric
        assert isinstance(trade["amount"], (int, float)), f"Trade amount not numeric: {trade['amount']}"
        assert isinstance(trade["price"], (int, float)), f"Trade price not numeric: {trade['price']}"
        assert trade["price"] > 0, f"Trade price not positive: {trade['price']}"
        
        # Validate trade amount sign matches trade type
        amt = trade["amount"]
        if amt > 0:
            # Positive amount should be a buy
            assert trade["type"] == "buy", f"Positive amount but type not 'buy': {trade}"
        elif amt < 0:
            # Negative amount should be a sell
            assert trade["type"] == "sell", f"Negative amount but type not 'sell': {trade}"
        else:
            # Zero amount trades should have a valid reason
            valid_reasons = (
                "trade_size_too_small", 
                "insufficient_funds", 
                "price_not_available",
                "risk_check_failed",
                "position_limit_exceeded"
            )
            assert trade.get("reason") in valid_reasons, \
                f"Zero-amount trade with no valid reason: {trade}"
    
    # Also check final portfolio is > 0
    final_value = results["portfolio_values"][-1]
    assert final_value > 0, f"Final portfolio value is not positive for {agent_name}"

@pytest.mark.parametrize("agent_name", ["Dummy", "MeanReversion"])
def test_backtest_with_nan_data(agent_name, caplog):
    """
    Test how backtest handles NaN in the data. Agents should not crash nor produce weird actions.
    """
    data = create_test_data(50)
    # Insert NaN in the middle
    data.iloc[20:25, data.columns.get_loc("$close")] = np.nan
    
    settings = create_test_settings(agent_name)
    manager = BacktestManager(settings)
    
    with caplog.at_level("ERROR"):
        results = manager.run_backtest(data)
    
    # Again, check for "Error getting action..." logs
    error_logs = [r.message for r in caplog.records if "Error getting action" in r.message]
    assert not error_logs, f"NaN data triggered action error logs: {error_logs}"
    
    # Basic result checks
    assert "portfolio_values" in results
    assert len(results["portfolio_values"]) > 0
    assert "trades" in results
    assert isinstance(results["trades"], list)

    # Check trades for numeric action
    for trade in results["trades"]:
        action_val = float(trade["action"])
        assert -1.0 <= action_val <= 1.0