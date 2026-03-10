"""
Test the weekend close functionality in RiskManager and BaseBacktester.
This tests whether positions are automatically closed on Friday at the end of trading.

Test scenarios:
1. Weekend close enabled - positions should close on Friday
2. Weekend close disabled - positions should remain open on Friday
"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.risk_manager import RiskManager, RiskConfig
from training.backtesting.base_backtester import BaseBacktester


def create_test_data():
    """Create test data with timestamps spanning Monday to Monday"""
    # Create a range from Monday to next Monday
    index = pd.date_range(start='2023-01-02', end='2023-01-09', freq='D')  # Monday to Monday
    
    # Stable prices
    prices = [100] * len(index)
    
    df = pd.DataFrame({
        '$open': prices,
        '$high': [p * 1.02 for p in prices],
        '$low': [p * 0.98 for p in prices],
        '$close': prices,
        '$volume': [1000000] * len(prices)
    }, index=index)
    
    return df


def test_weekend_close_enabled():
    """Test that positions are closed on Friday when weekend close is enabled"""
    # Create test data
    data = create_test_data()
    
    # Create risk config with weekend close enabled
    risk_config = RiskConfig(
        max_position_size=1.0,
        close_positions_on_friday=True
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy on Monday
    monday = pd.Timestamp('2023-01-02 16:00:00')  # Monday
    backtester.update(
        timestamp=monday,
        prices={'default': 100},
        actions={'default': 0.5}  # 50% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Thursday - position should still exist
    thursday = pd.Timestamp('2023-01-05 16:00:00')  # Thursday
    backtester.update(
        timestamp=thursday,
        prices={'default': 100},
        actions={'default': 0.5}  # maintain position
    )
    
    # Verify position still exists
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Friday before market close - position should still exist
    friday_morning = pd.Timestamp('2023-01-06 10:00:00')  # Friday morning
    backtester.update(
        timestamp=friday_morning,
        prices={'default': 100},
        actions={'default': 0.5}  # maintain position
    )
    
    # Position should still exist
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Friday after market close - position should be closed
    friday_close = pd.Timestamp('2023-01-06 16:01:00')  # Friday after 16:00
    result = backtester.update(
        timestamp=friday_close,
        prices={'default': 100},
        actions={'default': 0.5}  # try to maintain position
    )
    
    # Verify position was closed
    assert 'default' not in backtester.positions or abs(backtester.positions['default']['units']) < 1e-8


def test_weekend_close_disabled():
    """Test that positions remain open on Friday when weekend close is disabled"""
    # Create test data
    data = create_test_data()
    
    # Create risk config with weekend close disabled
    risk_config = RiskConfig(
        max_position_size=1.0,
        close_positions_on_friday=False  # Disabled
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy on Monday
    monday = pd.Timestamp('2023-01-02 16:00:00')  # Monday
    backtester.update(
        timestamp=monday,
        prices={'default': 100},
        actions={'default': 0.5}  # 50% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Friday after market close - position should still exist
    friday_close = pd.Timestamp('2023-01-06 16:01:00')  # Friday after 16:00
    result = backtester.update(
        timestamp=friday_close,
        prices={'default': 100},
        actions={'default': 0.5}  # maintain position
    )
    
    # Verify position still exists
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0


def test_multi_asset_weekend_close():
    """Test weekend close with multiple assets"""
    # Create a range from Monday to next Monday
    index = pd.date_range(start='2023-01-02', end='2023-01-09', freq='D')  # Monday to Monday
    
    # Stable prices for both assets
    btc_prices = [100] * len(index)
    eth_prices = [50] * len(index)
    
    # Create risk config with weekend close enabled
    risk_config = RiskConfig(
        max_position_size=1.0,
        close_positions_on_friday=True
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        risk_config=risk_config
    )
    
    # Buy both assets on Monday
    monday = pd.Timestamp('2023-01-02 16:00:00')  # Monday
    backtester.update(
        timestamp=monday,
        prices={'BTC': 100, 'ETH': 50},
        actions={'BTC': 0.3, 'ETH': 0.3}  # 30% in each
    )
    
    # Verify we have positions
    assert 'BTC' in backtester.positions
    assert 'ETH' in backtester.positions
    
    # Friday after market close - positions should be closed
    friday_close = pd.Timestamp('2023-01-06 16:01:00')  # Friday after 16:00
    result = backtester.update(
        timestamp=friday_close,
        prices={'BTC': 100, 'ETH': 50},
        actions={'BTC': 0.3, 'ETH': 0.3}  # try to maintain positions
    )
    
    # Verify positions were closed
    assert 'BTC' not in backtester.positions or abs(backtester.positions['BTC']['units']) < 1e-8
    assert 'ETH' not in backtester.positions or abs(backtester.positions['ETH']['units']) < 1e-8


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 