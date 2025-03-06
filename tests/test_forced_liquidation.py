"""
Test the forced liquidation functionality in RiskManager and BaseBacktester.
This tests whether the drawdown-triggered liquidation works as expected.

Test scenarios:
1. Forced liquidation when drawdown exceeds threshold
2. Verification that positions are closed after liquidation
3. Partial fills during liquidation
"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.risk_manager import RiskManager, RiskConfig
from training.backtesting.base_backtester import BaseBacktester


def create_test_data():
    """Create test data with a price crash to trigger forced liquidation"""
    index = pd.date_range(start='2023-01-01', periods=10, freq='1d')
    
    # First 5 days are stable, then a crash occurs
    prices = [100, 101, 102, 101, 100, 85, 80, 78, 76, 75]  # 25% crash
    
    df = pd.DataFrame({
        '$open': prices,
        '$high': [p * 1.02 for p in prices],
        '$low': [p * 0.98 for p in prices],
        '$close': prices,
        '$volume': [1000000] * len(prices)
    }, index=index)
    
    return df


def test_forced_liquidation_enabled():
    """Test that forced liquidation works when enabled"""
    # Create test data and setup
    data = create_test_data()
    
    # Create risk config with forced liquidation enabled
    risk_config = RiskConfig(
        max_position_size=1.0,  # Allow full position
        enable_forced_liquidation=True,
        forced_liquidation_drawdown=0.15  # 15% drawdown trigger
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy at the start
    timestamp = data.index[0]
    backtester.update(
        timestamp=timestamp,
        prices={'default': data.loc[timestamp, '$close']},
        actions={'default': 0.95}  # 95% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Move forward to day 5 (before crash)
    timestamp = data.index[4]
    backtester.update(
        timestamp=timestamp,
        prices={'default': data.loc[timestamp, '$close']},
        actions={'default': 0.95}  # maintain position
    )
    
    # Check position still exists
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    assert not backtester.risk_manager.liquidation_triggered
    
    # Move to day 6 (after crash, should trigger liquidation)
    timestamp = data.index[5]
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': data.loc[timestamp, '$close']},
        actions={'default': 0.95}  # try to maintain position
    )
    
    # Verify forced liquidation occurred
    trade = result['trades']['default'] 
    assert trade['is_forced_liquidation'] or abs(backtester.positions.get('default', {}).get('units', 0)) < 1e-8
    
    # Position should be closed
    assert 'default' not in backtester.positions or abs(backtester.positions['default']['units']) < 1e-8


def test_forced_liquidation_disabled():
    """Test that positions are maintained when forced liquidation is disabled"""
    # Create simple test data with a price crash on day 6
    data = create_test_data()
    
    # Create risk config with forced liquidation disabled AND stop loss disabled
    risk_config = RiskConfig(
        max_position_size=1.0,  # Allow full position
        enable_forced_liquidation=False,  # Disable forced liquidation
        forced_liquidation_drawdown=0.15,  # 15% drawdown trigger
        enable_stop_loss=False  # Also disable stop loss to make sure position remains
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        data=data,
        risk_config=risk_config
    )
    
    # Buy at the start
    timestamp = data.index[0]
    backtester.update(
        timestamp=timestamp,
        prices={'default': data.loc[timestamp, '$close']},
        actions={'default': 0.95}  # 95% position
    )
    
    # Verify we have a position
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    
    # Move to day 6 (after crash)
    timestamp = data.index[5]
    result = backtester.update(
        timestamp=timestamp,
        prices={'default': data.loc[timestamp, '$close']},
        actions={'default': 0.95}  # maintain position
    )
    
    # Position should still exist since liquidation is disabled
    assert 'default' in backtester.positions
    assert backtester.positions['default']['units'] > 0
    assert not backtester.risk_manager.liquidation_triggered


def test_multi_asset_forced_liquidation():
    """Test forced liquidation with multiple assets"""
    # Create test data for multiple assets
    index = pd.date_range(start='2023-01-01', periods=10, freq='1d')
    
    # BTC prices (will crash)
    btc_prices = [100, 101, 102, 101, 100, 85, 80, 78, 76, 75]  # 25% crash
    
    # ETH prices (will remain stable)
    eth_prices = [50, 51, 52, 51, 50, 51, 50, 49, 50, 51]
    
    # Create dataframe with multiindex columns for both assets
    columns = pd.MultiIndex.from_product([['BTC', 'ETH'], ['$open', '$high', '$low', '$close', '$volume']])
    data = pd.DataFrame(columns=columns, index=index)
    
    for i, ts in enumerate(index):
        data.loc[ts, ('BTC', '$open')] = btc_prices[i]
        data.loc[ts, ('BTC', '$high')] = btc_prices[i] * 1.02
        data.loc[ts, ('BTC', '$low')] = btc_prices[i] * 0.98
        data.loc[ts, ('BTC', '$close')] = btc_prices[i]
        data.loc[ts, ('BTC', '$volume')] = 1000000
        
        data.loc[ts, ('ETH', '$open')] = eth_prices[i]
        data.loc[ts, ('ETH', '$high')] = eth_prices[i] * 1.02
        data.loc[ts, ('ETH', '$low')] = eth_prices[i] * 0.98
        data.loc[ts, ('ETH', '$close')] = eth_prices[i]
        data.loc[ts, ('ETH', '$volume')] = 2000000
    
    # Create risk config with forced liquidation enabled
    risk_config = RiskConfig(
        max_position_size=1.0,  # Allow full position
        enable_forced_liquidation=True,
        forced_liquidation_drawdown=0.15  # 15% drawdown trigger
    )
    
    # Setup backtester
    backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        risk_config=risk_config
    )
    
    # Buy both assets at the start
    timestamp = index[0]
    backtester.update(
        timestamp=timestamp,
        prices={'BTC': btc_prices[0], 'ETH': eth_prices[0]},
        actions={'BTC': 0.4, 'ETH': 0.4}  # 40% in each asset
    )
    
    # Verify we have positions in both assets
    assert 'BTC' in backtester.positions
    assert 'ETH' in backtester.positions
    
    # Move to day 6 (after BTC crash)
    timestamp = index[5]
    result = backtester.update(
        timestamp=timestamp,
        prices={'BTC': btc_prices[5], 'ETH': eth_prices[5]},
        actions={'BTC': 0.4, 'ETH': 0.4}  # try to maintain positions
    )
    
    # Both positions should be liquidated (since total portfolio had >15% drawdown)
    assert 'BTC' not in backtester.positions or abs(backtester.positions['BTC']['units']) < 1e-8
    assert 'ETH' not in backtester.positions or abs(backtester.positions['ETH']['units']) < 1e-8


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 