"""
Test the maximum holding period functionality in RiskManager and BaseBacktester.
This tests whether positions are automatically closed after exceeding the max holding period.

Test scenarios:
1. Max holding period exceeded - positions should close
2. Max holding period not exceeded - positions should remain open
3. Multiple assets with different holding periods
"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.risk_manager import RiskManager, RiskConfig
from training.backtesting.base_backtester import BaseBacktester


def create_test_data(days=10):
    """Create test data with timestamps spanning multiple days"""
    # Create a range of days
    index = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Stable prices
    prices = [100] * days
    
    df = pd.DataFrame({
        '$open': prices,
        '$high': [p * 1.02 for p in prices],
        '$low': [p * 0.98 for p in prices],
        '$close': prices,
        '$volume': [1000000] * days
    }, index=index)
    
    return df


def test_max_holding_period_exceeded():
    """Test that positions are closed when max holding period is exceeded"""
    # Create test data
    data = create_test_data(days=10)
    
    # Create risk config with max holding period of 5 days
    risk_config = RiskConfig(
        max_position_size=1.0,
        max_holding_period_days=5
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy on day 1
    day1 = data.index[0]
    backtester.update(
        timestamp=day1,
        prices={'default': 100},
        actions={'default': 0.5}  # 50% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Verify risk manager is tracking position start time
    assert 'default' in backtester.risk_manager.position_start_times
    assert backtester.risk_manager.position_start_times['default'] == day1
    
    # Day 4 - position should still exist (not yet exceeded 5 days)
    day4 = data.index[3]
    backtester.update(
        timestamp=day4,
        prices={'default': 100},
        actions={'default': 0.5}  # maintain position
    )
    
    # Position should still exist
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Day 6 - position should be closed (exceeded 5 days)
    day6 = data.index[5]
    result = backtester.update(
        timestamp=day6,
        prices={'default': 100},
        actions={'default': 0.5}  # try to maintain position
    )
    
    # Verify position was closed
    assert 'default' not in backtester.positions or abs(backtester.positions['default']['units']) < 1e-8


def test_max_holding_period_disabled():
    """Test that positions remain open when max holding period is disabled"""
    # Create test data
    data = create_test_data(days=10)
    
    # Create risk config with max holding period disabled (0 days)
    risk_config = RiskConfig(
        max_position_size=1.0,
        max_holding_period_days=0  # Disabled
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy on day 1
    day1 = data.index[0]
    backtester.update(
        timestamp=day1,
        prices={'default': 100},
        actions={'default': 0.5}  # 50% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Day 9 - position should still exist (no max holding period)
    day9 = data.index[8]
    result = backtester.update(
        timestamp=day9,
        prices={'default': 100},
        actions={'default': 0.5}  # maintain position
    )
    
    # Position should still exist
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0


def test_multi_asset_holding_periods():
    """Test max holding period functionality with multiple assets."""
    # Create test data for multiple assets
    index = pd.date_range(start='2023-01-01', periods=10, freq='1d')
    data = pd.DataFrame(index=index)
    
    # Setup with max holding period of 5 days
    risk_config = RiskConfig(
        max_holding_period_days=5,
        enable_stop_loss=False,  # Disable stop loss
        max_drawdown_pct=1.0,    # Disable max drawdown (100% drawdown allowed)
        max_position_size=0.5    # Increase max position size to allow 30% positions
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        risk_config=risk_config
    )
    
    # Buy BTC on day 1
    day1 = data.index[0]
    backtester.update(
        timestamp=day1,
        prices={'BTC': 100},
        actions={'BTC': 0.3}  # 30% position
    )
    
    # Buy ETH on day 3
    day3 = data.index[2]
    backtester.update(
        timestamp=day3,
        prices={'BTC': 100, 'ETH': 50},
        actions={'BTC': 0.3, 'ETH': 0.3}  # add ETH position
    )
    
    # Verify we have both positions
    assert 'BTC' in backtester.positions
    assert 'ETH' in backtester.positions
    
    # Verify risk manager is tracking both position start times
    assert 'BTC' in backtester.risk_manager.position_start_times
    assert 'ETH' in backtester.risk_manager.position_start_times
    assert backtester.risk_manager.position_start_times['BTC'] == day1
    assert backtester.risk_manager.position_start_times['ETH'] == day3
    
    # Day 6 - BTC position should be closed (exceeded 5 days), ETH should remain
    day6 = data.index[5]
    result = backtester.update(
        timestamp=day6,
        prices={'BTC': 100, 'ETH': 50},
        actions={'BTC': 0.3, 'ETH': 0.3}  # try to maintain positions
    )
    
    # Verify BTC position was closed but ETH remains
    assert 'BTC' not in backtester.positions or abs(backtester.positions.get('BTC', {}).get('units', 0)) < 1e-8
    assert 'ETH' in backtester.positions
    assert backtester.positions['ETH']['units'] > 0
    
    # Day 8 - ETH position should also be closed (exceeded 5 days from day 3)
    day8 = data.index[7]
    result = backtester.update(
        timestamp=day8,
        prices={'BTC': 100, 'ETH': 50},
        actions={'BTC': 0.3, 'ETH': 0.3}  # try to maintain/reopen positions
    )
    
    # Verify ETH position was also closed
    assert 'ETH' not in backtester.positions or abs(backtester.positions.get('ETH', {}).get('units', 0)) < 1e-8
    
    # If reopening happened, BTC might have a new position with new start time
    if 'BTC' in backtester.positions and backtester.positions['BTC']['units'] > 0:
        assert backtester.risk_manager.position_start_times['BTC'] > day1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 