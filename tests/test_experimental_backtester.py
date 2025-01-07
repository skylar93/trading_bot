import pytest
import pandas as pd
import numpy as np
from training.utils.backtesting.experimental_backtester import ExperimentalBacktester
from risk.risk_manager import RiskConfig

@pytest.fixture
def risk_config():
    return RiskConfig(
        max_position_size=0.1,  # 10% limit
        stop_loss_pct=0.02,
        max_drawdown_pct=0.15,
        daily_trade_limit=10,
        var_confidence_level=0.95,
        portfolio_var_limit=0.02,
        max_correlation=0.7,
        min_trade_size=0.01,
        max_leverage=1.0
    )

@pytest.fixture
def test_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "$open": np.linspace(100, 110, 100),
        "$high": np.linspace(101, 111, 100),
        "$low": np.linspace(99, 109, 100),
        "$close": np.linspace(100, 110, 100),
        "$volume": np.random.rand(100) * 1000
    }, index=dates)
    return df

def test_position_size_limit(test_data, risk_config):
    """Test that position size never exceeds the configured limit."""
    backtester = ExperimentalBacktester(test_data, risk_config)
    
    # Try to open a position that would exceed the limit
    result = backtester.execute_trade(
        timestamp=test_data.index[0],
        action=1.0,  # Try to use 100% of portfolio
        price_data={"$close": test_data["$close"].iloc[0]}
    )
    
    # Calculate actual position size as percentage
    portfolio_value = result["portfolio_value"]
    position_value = result["size"] * result["price"]
    position_size_pct = position_value / portfolio_value
    
    # Should be strictly less than the limit
    assert position_size_pct <= risk_config.max_position_size
    # Should be close to but not exactly at the limit (due to 0.99 buffer)
    assert position_size_pct > risk_config.max_position_size * 0.98

def test_partial_position_closure(test_data, risk_config):
    """Test partial position closure and PnL calculation."""
    backtester = ExperimentalBacktester(test_data, risk_config)
    
    # Open initial position
    entry_price = test_data["$close"].iloc[0]
    buy_result = backtester.execute_trade(
        timestamp=test_data.index[0],
        action=0.5,  # Use 50% of portfolio
        price_data={"$close": entry_price}
    )
    initial_position = buy_result["position"]
    
    # Close half the position at a profit
    exit_price = entry_price * 1.1  # 10% price increase
    sell_result = backtester.execute_trade(
        timestamp=test_data.index[1],
        action=-0.25,  # Close half of the position
        price_data={"$close": exit_price}
    )
    
    # Verify position size reduced
    assert sell_result["position"] == pytest.approx(initial_position / 2, rel=0.01)
    
    # Verify PnL calculation
    expected_pnl = (exit_price - entry_price) * sell_result["size"] - \
                  (exit_price * sell_result["size"] * backtester.trading_fee)
    assert sell_result["pnl"] == pytest.approx(expected_pnl, rel=0.01)
    
    # Verify revenue calculation
    expected_revenue = exit_price * sell_result["size"] * (1 - backtester.trading_fee)
    assert sell_result["revenue"] == pytest.approx(expected_revenue, rel=0.01)

def test_position_entry_price_tracking(test_data, risk_config):
    """Test that entry prices are correctly tracked for multiple trades."""
    backtester = ExperimentalBacktester(test_data, risk_config)
    
    # First buy at 100
    price1 = 100.0
    result1 = backtester.execute_trade(
        timestamp=test_data.index[0],
        action=0.3,  # Use 30% of portfolio
        price_data={"$close": price1}
    )
    
    # Second buy at 105
    price2 = 105.0
    result2 = backtester.execute_trade(
        timestamp=test_data.index[1],
        action=0.2,  # Use additional 20% of portfolio
        price_data={"$close": price2}
    )
    
    # Get the weighted average entry price
    total_size = result1["size"] + result2["size"]
    expected_entry_price = (result1["size"] * price1 + result2["size"] * price2) / total_size
    
    # Verify entry price tracking
    assert backtester.entry_prices["default"] == pytest.approx(expected_entry_price, rel=0.01)

def test_complete_position_closure(test_data, risk_config):
    """Test complete position closure and cleanup."""
    backtester = ExperimentalBacktester(test_data, risk_config)
    
    # Open position
    buy_result = backtester.execute_trade(
        timestamp=test_data.index[0],
        action=0.5,
        price_data={"$close": 100.0}
    )
    
    # Close entire position
    sell_result = backtester.execute_trade(
        timestamp=test_data.index[1],
        action=-0.5,
        price_data={"$close": 110.0}
    )
    
    # Verify position is fully closed
    assert sell_result["position"] == 0
    assert "default" not in backtester.positions
    assert "default" not in backtester.entry_prices 