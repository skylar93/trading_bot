import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.backtesting.base_backtester import BaseBacktester
from training.evaluation import TradingMetrics
from pathlib import Path


class DummyAgent:
    """Dummy agent for testing"""

    def get_action(self, state):
        """Random action between -1 and 1"""
        return np.random.uniform(-1, 1)


@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    # Create date range
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")

    # Generate price data
    base_price = 100
    returns = np.random.normal(0, 0.01, size=100)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create DataFrame with $ prefixed columns
    data = pd.DataFrame(
        {
            "$open": prices * (1 + np.random.uniform(-0.001, 0.001, 100)),
            "$high": prices * (1 + np.random.uniform(0.001, 0.002, 100)),
            "$low": prices * (1 - np.random.uniform(0.001, 0.002, 100)),
            "$close": prices,
            "$volume": np.random.uniform(1000, 2000, 100),
        },
        index=dates,
    )

    # Ensure high is highest and low is lowest
    data["$high"] = data[["$open", "$high", "$low", "$close"]].max(axis=1)
    data["$low"] = data[["$open", "$high", "$low", "$close"]].min(axis=1)

    return data


def test_backtester_initialization(sample_data):
    """Test Backtester initialization"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=sample_data
    )
    assert backtester.cash == 10000.0
    assert len(backtester.positions) == 0
    assert len(backtester.trades) == 0
    assert backtester.portfolio_history == [10000.0]


def test_trade_execution(sample_data):
    """Test trade execution"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0
    )

    # Test buy action
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=0.5,  # Buy 50% of possible position
        price_data={'default': 101.0},  # Use close price
        asset='default'
    )

    assert result['type'] == 'buy'
    assert result['amount'] > 0  # Buy should have positive amount
    assert backtester.cash < 10000.0  # Balance should decrease after buy
    assert backtester.positions['default'] > 0  # Should have positive position

    # Test sell action
    result = backtester.execute_trade(
        timestamp=sample_data.index[1],
        action=-0.5,  # Sell 50% of position
        price_data={'default': 102.0},  # Use close price
        asset='default'
    )

    assert result['type'] == 'sell'
    assert result['amount'] < 0  # Sell should have negative amount
    assert backtester.positions.get('default', 0) >= 0  # Position should never go negative


def test_backtest_run(sample_data):
    """Test running backtest"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=sample_data
    )
    agent = DummyAgent()

    results = backtester.run(agent, window_size=5)

    assert isinstance(results, dict)
    assert "metrics" in results
    assert "trades" in results
    assert "portfolio_values" in results

    # Check metrics
    metrics = results["metrics"]
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics

    # Check trades
    trades = results["trades"]
    if trades:  # If any trades were made
        first_trade = trades[0]
        assert "timestamp" in first_trade
        assert "type" in first_trade
        assert "price" in first_trade
        assert "amount" in first_trade

    # Check portfolio values
    assert len(results["portfolio_values"]) > 0
    assert isinstance(results["portfolio_values"][0], float)


def test_metrics_calculation():
    """Test trading metrics calculation"""
    # Test Sharpe Ratio
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    sharpe = TradingMetrics.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)

    # Test Maximum Drawdown
    portfolio_values = np.array([100, 95, 105, 90, 100, 110])
    mdd, start_idx, end_idx = TradingMetrics.calculate_maximum_drawdown(
        portfolio_values
    )
    assert isinstance(mdd, float)
    assert mdd <= 0  # Drawdown should be negative
    assert start_idx < end_idx

    # Test Win Rate
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 75}, {"pnl": 25}]
    win_rate = TradingMetrics.calculate_win_rate(trades)
    assert win_rate == 75.0  # 3 out of 4 trades are profitable

    # Test Profit/Loss Ratio
    pnl_ratio = TradingMetrics.calculate_profit_loss_ratio(trades)
    assert pnl_ratio > 0
    assert isinstance(pnl_ratio, float)


def test_backtester_save_results(sample_data, tmp_path):
    """Test saving backtest results"""
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=sample_data
    )
    agent = DummyAgent()

    # Run backtest
    results = backtester.run(agent, window_size=5)

    # Create save directory
    save_dir = tmp_path / "backtest_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save results manually since save_results is not part of the core functionality
    metrics_df = pd.DataFrame([results["metrics"]])
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        trades_df.to_csv(save_dir / "trades.csv", index=False)

    # Create portfolio values DataFrame
    portfolio_df = pd.DataFrame(
        {
            "timestamp": results["timestamps"],
            "value": results["portfolio_values"],
        }
    )
    portfolio_df.to_csv(save_dir / "portfolio_values.csv", index=False)

    # Verify file contents
    assert (save_dir / "metrics.csv").exists()
    metrics_df = pd.read_csv(save_dir / "metrics.csv")
    assert not metrics_df.empty
    assert "total_return" in metrics_df.columns

    if results["trades"]:
        assert (save_dir / "trades.csv").exists()
        trades_df = pd.read_csv(save_dir / "trades.csv")
        assert not trades_df.empty
        assert "type" in trades_df.columns
        assert "price" in trades_df.columns
        assert "amount" in trades_df.columns

    # Verify portfolio values
    assert (save_dir / "portfolio_values.csv").exists()
    portfolio_df = pd.read_csv(save_dir / "portfolio_values.csv")
    assert not portfolio_df.empty
    assert len(portfolio_df) == len(
        results["timestamps"]
    )  # Verify lengths match
    assert "timestamp" in portfolio_df.columns
    assert "value" in portfolio_df.columns


def test_edge_cases(sample_data):
    """Test edge cases and error handling"""
    # Test insufficient balance first
    backtester = BaseBacktester(
        initial_capital=1.0,  # Set very low initial capital
        trading_fee=0.001,
        max_position=1.0
    )
    
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=1.0,
        price_data={'default': 101.0},
        asset='default'
    )
    
    assert result['success']  # Should succeed but with zero amount
    assert result['amount'] == 0  # No trade executed
    assert result['reason'] == 'trade size too small'
    assert len(backtester.positions) == 0

    # Test zero action
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0
    )
    
    result = backtester.execute_trade(
        timestamp=sample_data.index[0],
        action=0,
        price_data={'default': 101.0},
        asset='default'
    )
    
    assert result['success']  # Should succeed but skip
    assert result['reason'] == 'trade size too small'
    assert len(backtester.positions) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
