"""
Risk Manager Integration Tests: checks partial clamping, daily trade limits, 
drawdown, min trade sizes, etc., using execute_trade(...) 
with a RiskAwareBacktester or a BaseBacktester + risk_config.
Tests for risk management integration with BaseBacktester
"""

import pytest
import pandas as pd
import numpy as np
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_manager import RiskConfig
from training.backtesting.risk_aware_backtester import RiskAwareBacktester

def create_test_data(days: int = 100) -> pd.DataFrame:
    """Create test OHLCV data"""
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

def test_risk_manager_initialization():
    """Test risk manager initialization in BaseBacktester"""
    data = create_test_data()
    
    # Without risk config
    backtester = BaseBacktester(data=data)
    assert backtester.risk_manager is None
    
    # With risk config
    risk_config = RiskConfig()
    backtester = BaseBacktester(data=data, risk_config=risk_config)
    assert backtester.risk_manager is not None
    assert backtester.risk_manager.config == risk_config

def test_position_size_limits():
    """Test position size limits from risk manager"""
    data = create_test_data()
    
    # Setup backtester with strict position limits
    risk_config = RiskConfig(max_position_size=0.1)  # 10% max position
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config
    )
    
    # Try to take a position larger than allowed
    result = backtester.execute_trade(
        timestamp=data.index[0],
        action=0.5,  # Try 50% position
        price_data={'default': data.iloc[0]['$close']},
        asset='default'
    )
    
    # Should be adjusted down to 10%
    assert result['success'] is True
    position_size = abs(result['amount'] * result['price'] / 10000.0)
    assert 0.09 <= position_size <= 0.11  # Allow small rounding differences

def test_daily_trade_limits():
    """Test daily trade limits from risk manager"""
    data = create_test_data()
    
    # Setup backtester with strict trade limits
    risk_config = RiskConfig(
        daily_trade_limit=2,  # Only 2 trades per day
        min_trade_size=0.001  # Small minimum to avoid size issues
    )
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config
    )
    
    timestamp = data.index[0]
    price = data.iloc[0]['$close']
    
    # First trade should succeed
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.1,
        price_data={'default': price},
        asset='default'
    )
    print(f"First trade result: {result1}")  # Debug print
    assert result1['success'] is True, f"First trade failed: {result1.get('reason', 'unknown')}"
    
    # Second trade should succeed
    result2 = backtester.execute_trade(
        timestamp=timestamp,
        action=-0.05,
        price_data={'default': price},
        asset='default'
    )
    print(f"Second trade result: {result2}")  # Debug print
    assert result2['success'] is True, f"Second trade failed: {result2.get('reason', 'unknown')}"
    
    # Third trade should be rejected
    result3 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.1,
        price_data={'default': price},
        asset='default'
    )
    print(f"Third trade result: {result3}")  # Debug print
    assert result3['success'] is False
    assert "Daily trade limit" in result3['reason']
    
    # Trade on next day should succeed
    next_day = timestamp + pd.Timedelta(days=1)
    result4 = backtester.execute_trade(
        timestamp=next_day,
        action=0.1,
        price_data={'default': price},
        asset='default'
    )
    print(f"Next day trade result: {result4}")  # Debug print
    assert result4['success'] is True, f"Next day trade failed: {result4.get('reason', 'unknown')}"

def test_drawdown_limits():
    """Test maximum drawdown limits"""
    data = create_test_data()
    
    # Setup backtester with strict drawdown limit
    risk_config = RiskConfig(
        max_drawdown_pct=0.05,  # 5% max drawdown
        min_trade_size=0.001,  # Small minimum to avoid size issues
        max_position_size=1.0  # Allow full portfolio position
    )
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config,
        max_position=1.0  # Allow full portfolio position
    )
    
    # Take initial position
    # Day1: Buy position (action=0.99 instead of 1.0)
    timestamp = data.index[0]
    price = 100.0  # Use fixed price for predictable math
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.99,   
        price_data={'default': price},
        asset='default'
    )
    print(f"Initial position result: {result1}")  # Debug print
    assert result1['success'] is True
    
    # Calculate initial portfolio value
    initial_portfolio = backtester.get_portfolio_value({'default': price})
    print(f"Initial portfolio value: {initial_portfolio}")  # Debug print
    
    # Force a few updates at the peak to ensure it's captured
    for _ in range(3):
        _ = backtester.get_portfolio_value({'default': price})
    
    # Simulate price drop causing >5% drawdown
    new_price = price * 0.93  # 7% drop
    current_portfolio = backtester.get_portfolio_value({'default': new_price})
    drawdown = (initial_portfolio - current_portfolio) / initial_portfolio
    print(f"Peak portfolio value: {backtester.risk_manager.peak_value}")  # Debug print
    print(f"Current portfolio value: {current_portfolio}")  # Debug print
    print(f"Current drawdown: {drawdown:.2%}")  # Debug print
    
    # Force a few updates at the low to ensure drawdown is captured
    for _ in range(3):
        _ = backtester.get_portfolio_value({'default': new_price})
    
    result2 = backtester.execute_trade(
        timestamp=timestamp + pd.Timedelta(days=1),
        action=-0.1,  # Try to reduce position (sell)
        price_data={'default': new_price},
        asset='default'
    )
    print(f"Trade during drawdown result: {result2}")  # Debug print
    assert result2['success'] is False
    assert "drawdown" in result2['reason'].lower()

def test_leverage_limits():
    """Test leverage limits"""
    data = create_test_data()
    
    # Setup backtester with strict leverage limit
    risk_config = RiskConfig(
        max_leverage=1.5,  # 1.5x max leverage
        min_trade_size=0.001  # Small minimum to avoid size issues
    )
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config
    )
    
    timestamp = data.index[0]
    price = data.iloc[0]['$close']
    
    # Take position up to leverage limit
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=1.0,  # Try to use full capital
        price_data={'default': price},
        asset='default'
    )
    assert result1['success'] is True
    
    # Try to exceed leverage limit
    result2 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.5,  # Try to add more leverage
        price_data={'default': price},
        asset='default'
    )
    
    # Should either be rejected or adjusted down
    if result2['success']:
        # If adjusted, check leverage limit
        position_value = abs(backtester.positions['default']['units'] * price)
        current_leverage = position_value / backtester.get_portfolio_value({'default': price})
        print(f"Current leverage: {current_leverage:.2f}x")  # Debug print
        assert current_leverage <= 1.5
    else:
        assert "leverage" in result2['reason'].lower()

def test_minimum_trade_size():
    """Test minimum trade size limits"""
    data = create_test_data()
    
    # Setup backtester with minimum trade size
    risk_config = RiskConfig(min_trade_size=0.01)  # 1% minimum trade
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config
    )
    
    timestamp = data.index[0]
    price = data.iloc[0]['$close']
    
    # Try very small trade
    result = backtester.execute_trade(
        timestamp=timestamp,
        action=0.001,  # 0.1% position
        price_data={'default': price},
        asset='default'
    )
    
    # Add debug output
    print("\nDEBUG INFO:")
    print(f"Action: {0.001}")
    print(f"Min trade size: {risk_config.min_trade_size}")
    print(f"Portfolio value: {backtester.get_portfolio_value({'default': price})}")
    print(f"Result: {result}")
    
    assert result['success'] is False
    assert "minimum" in result['reason'].lower() or result['reason'] == "trade_size_too_small"

def test_risk_reset():
    """Test risk manager state reset"""
    data = create_test_data()
    
    risk_config = RiskConfig(
        daily_trade_limit=2,
        min_trade_size=0.001  # Small minimum to avoid size issues
    )
    backtester = BaseBacktester(
        initial_capital=10000.0,
        data=data,
        risk_config=risk_config
    )
    
    timestamp = data.index[0]
    price = data.iloc[0]['$close']
    
    # Execute trades until limit
    result1 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.1,
        price_data={'default': price},
        asset='default'
    )
    print(f"First trade result: {result1}")  # Debug print
    
    result2 = backtester.execute_trade(
        timestamp=timestamp,
        action=-0.05,
        price_data={'default': price},
        asset='default'
    )
    print(f"Second trade result: {result2}")  # Debug print
    assert result1['success'] and result2['success']
    
    # Reset should clear trade counter
    backtester.reset()
    
    # Should be able to trade again
    result3 = backtester.execute_trade(
        timestamp=timestamp,
        action=0.1,
        price_data={'default': price},
        asset='default'
    )
    print(f"Post-reset trade result: {result3}")  # Debug print
    assert result3['success'] is True

def test_deprecated_risk_aware_backtester():
    """Test deprecated RiskAwareBacktester wrapper"""
    data = create_test_data()
    
    # Test with default risk config
    with pytest.warns(DeprecationWarning):
        backtester = RiskAwareBacktester(data=data)
    assert backtester.risk_manager is not None
    assert isinstance(backtester.risk_manager.config, RiskConfig)
    
    # Test with custom risk config
    risk_config = RiskConfig(max_position_size=0.1)
    with pytest.warns(DeprecationWarning):
        backtester = RiskAwareBacktester(data=data, risk_config=risk_config)
    assert backtester.risk_manager is not None
    assert backtester.risk_manager.config == risk_config 