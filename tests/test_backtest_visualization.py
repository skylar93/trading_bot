"""
Test integration between backtesting and visualization
"""

import pytest
import pandas as pd
import numpy as np
from training.utils.backtesting.backtest_engine import BacktestEngine
from training.utils.visualization.visualization import TradingVisualizer
from training.utils.backtesting.performance_metrics import (
    calculate_returns_metrics,
)


def generate_sample_data(n_days: int = 100):
    """Generate sample price data"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days)

    # Generate slightly upward trending prices with noise
    btc_prices = 100 * (1 + np.random.normal(0.001, 0.02, n_days).cumsum())
    eth_prices = 50 * (1 + np.random.normal(0.001, 0.02, n_days).cumsum())

    prices_data = []
    for i in range(n_days):
        prices_data.append(
            {
                "timestamp": dates[i],
                "prices": {"BTC": btc_prices[i], "ETH": eth_prices[i]},
            }
        )
    return prices_data


def generate_sample_actions(n_days: int = 100):
    """Generate sample trading actions"""
    np.random.seed(42)
    actions_data = []

    # Simple strategy: alternate between BTC and ETH
    for i in range(n_days):
        if i % 20 < 10:  # Switch every 10 days
            actions_data.append({"BTC": 0.3, "ETH": 0.0})
        else:
            actions_data.append({"BTC": 0.0, "ETH": 0.3})
    return actions_data


def test_backtest_visualization():
    """Test full backtesting workflow with visualization"""
    # Initialize components
    engine = BacktestEngine(initial_capital=10000)
    visualizer = TradingVisualizer()

    # Generate sample data
    prices_data = generate_sample_data()
    actions_data = generate_sample_actions()

    # Run backtest
    portfolio_values = []
    trades_list = []
    returns_list = []

    for price_data, actions in zip(prices_data, actions_data):
        state = engine.update(
            timestamp=price_data["timestamp"],
            prices=price_data["prices"],
            actions=actions,
        )
        portfolio_values.append(state["portfolio_value"])
        if state["trades"]:
            trades_list.append(state["trades"])

    # Calculate returns
    returns = pd.Series(portfolio_values).pct_change().dropna().values

    # Calculate metrics
    metrics = calculate_returns_metrics(pd.Series(returns))

    # Generate visualization
    visualizer.plot_portfolio_performance(
        portfolio_values=np.array(portfolio_values),
        returns=returns,
        trades=trades_list,
        metrics=metrics,
        save_path="training_viz/backtest_results.png",
    )

    # Basic assertions
    assert len(portfolio_values) == len(prices_data)
    assert len(returns) == len(prices_data) - 1
    assert len(trades_list) > 0
    assert isinstance(metrics, dict)
    assert "sharpe_ratio" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
