"""Tests for experimental backtesting features"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.experimental_features import ExperimentalMixin
from training.backtesting.experimental_backtester import ExperimentalBacktester

class TestBacktester(ExperimentalMixin, BaseBacktester):
    """Test backtester with experimental features"""
    pass

def create_test_data(days: int = 100) -> pd.DataFrame:
    """Create test price data"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    price = 100.0
    data = []
    
    for date in dates:
        price *= (1 + np.random.normal(0, 0.02))  # 2% daily volatility
        data.append({
            'timestamp': date,
            '$open': price * (1 + np.random.normal(0, 0.001)),
            '$high': price * (1 + abs(np.random.normal(0, 0.002))),
            '$low': price * (1 - abs(np.random.normal(0, 0.002))),
            '$close': price,
            '$volume': np.random.uniform(1000, 5000)
        })
    
    return pd.DataFrame(data).set_index('timestamp')

def test_weighted_entry_price():
    """Test weighted entry price calculation"""
    data = create_test_data()
    backtester = TestBacktester(data=data, initial_capital=10000.0)
    
    # Execute multiple buys at different prices
    timestamp = data.index[0]
    price1 = 100.0
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.3,  # Buy 30%
        price_data={'default': price1}
    )
    assert result1['success']
    
    price2 = 105.0
    result2 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.2,  # Buy 20% more
        price_data={'default': price2}
    )
    assert result2['success']
    
    # Check weighted entry price with 1% tolerance
    position1 = abs(result1['amount'])
    position2 = abs(result2['amount'])
    expected_entry = (price1 * position1 + price2 * position2) / (position1 + position2)
    tolerance = expected_entry * 0.01  # 1% tolerance
    assert abs(backtester._entry_prices['default'] - expected_entry) < tolerance, \
        f"Entry price {backtester._entry_prices['default']} differs from expected {expected_entry}"

def test_partial_exit_pnl():
    """Test PnL calculation with partial position exits"""
    data = create_test_data()
    backtester = TestBacktester(data=data, initial_capital=10000.0)
    
    # Build position in parts
    timestamp = data.index[0]
    entry_price = 100.0
    
    # Buy first part
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.4,  # Buy 40%
        price_data={'default': entry_price}
    )
    assert result1['success']
    position1 = result1['amount']
    
    # Buy second part
    result2 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.3,  # Buy 30% more
        price_data={'default': entry_price}
    )
    assert result2['success']
    position2 = result2['amount']
    
    # Sell half at profit
    exit_price = 110.0
    exit_amount = (position1 + position2) * 0.5
    result3 = backtester.execute_trade(
        timestamp=timestamp,
        action=-0.35,  # Sell about half
        price_data={'default': exit_price}
    )
    assert result3['success']
    
    # Verify PnL calculation
    expected_pnl = (exit_price - entry_price) * abs(result3['amount'])
    assert abs(result3['realized_pnl'] - expected_pnl) < 0.01

def test_deprecated_experimental_backtester():
    """Test deprecated ExperimentalBacktester wrapper"""
    data = create_test_data()
    
    # Should raise deprecation warning
    with pytest.warns(DeprecationWarning):
        backtester = ExperimentalBacktester(data=data)
    
    # Should still work
    timestamp = data.index[0]
    result = backtester.execute_trade(
        timestamp=timestamp,
        action=0.1,
        price_data={'default': 100.0}
    )
    assert result['success']
    
    # Should have experimental features
    assert hasattr(backtester, '_entry_prices')
    assert hasattr(backtester, '_position_entries') 