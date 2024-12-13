"""
Test suite for backtesting system
"""
import pytest
import pandas as pd
import numpy as np
from training.utils.backtesting.backtest_engine import BacktestEngine
from training.utils.backtesting.performance_metrics import calculate_returns_metrics, calculate_trade_metrics

def test_backtest_engine_initialization():
    """Test basic initialization of backtest engine"""
    engine = BacktestEngine(initial_capital=10000)
    assert engine.portfolio_value == 10000
    assert engine.cash == 10000
    assert len(engine.positions) == 0
    assert len(engine.trades) == 0

def test_simple_buy_trade():
    """Test simple buy trade execution"""
    engine = BacktestEngine(initial_capital=10000)
    
    # Execute buy trade
    timestamp = pd.Timestamp('2024-01-01')
    prices = {'BTC': 100.0}
    actions = {'BTC': 0.5}  # 50% position
    
    state = engine.update(timestamp, prices, actions)
    
    # Verify position and portfolio value
    assert 'BTC' in state['positions']
    assert state['positions']['BTC'] > 0
    assert state['portfolio_value'] < 10000  # Should be less due to transaction costs
    assert len(engine.trades) == 1
    assert engine.trades[0]['type'] == 'buy'

def test_buy_sell_sequence():
    """Test a sequence of buy and sell trades"""
    engine = BacktestEngine(initial_capital=10000)
    
    # Buy sequence
    timestamps = [pd.Timestamp(f'2024-01-0{i}') for i in range(1, 4)]
    prices = [
        {'BTC': 100.0},
        {'BTC': 110.0},
        {'BTC': 90.0}
    ]
    actions = [
        {'BTC': 0.5},   # Buy 50%
        {'BTC': -0.2},  # Reduce to 30%
        {'BTC': 0.0}    # Sell all
    ]
    
    portfolio_values = []
    for t, p, a in zip(timestamps, prices, actions):
        state = engine.update(t, p, a)
        portfolio_values.append(state['portfolio_value'])
    
    # Verify trading sequence
    assert len(engine.trades) == 3
    assert engine.trades[0]['type'] == 'buy'
    assert engine.trades[1]['type'] == 'sell'
    assert engine.trades[2]['type'] == 'sell'
    
    # Verify positions are closed
    assert len(engine.positions) == 0

def test_performance_metrics():
    """Test performance metrics calculation"""
    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    # Calculate metrics
    metrics = calculate_returns_metrics(returns)
    
    # Verify metrics
    assert 'sharpe_ratio' in metrics
    assert 'sortino_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert isinstance(metrics['total_return'], float)

def test_position_limits():
    """Test position size limits"""
    engine = BacktestEngine(initial_capital=10000, max_position=0.5)
    
    # Try to take too large position
    timestamp = pd.Timestamp('2024-01-01')
    prices = {'BTC': 100.0}
    actions = {'BTC': 0.8}  # Attempt 80% position
    
    state = engine.update(timestamp, prices, actions)
    
    # Verify position was limited
    position_value = state['positions']['BTC'] * prices['BTC']
    assert position_value / state['portfolio_value'] <= 0.5

def test_transaction_costs():
    """Test transaction costs are properly applied"""
    engine = BacktestEngine(initial_capital=10000, transaction_cost=0.001)
    
    # Execute trade
    timestamp = pd.Timestamp('2024-01-01')
    prices = {'BTC': 100.0}
    actions = {'BTC': 0.5}
    
    state = engine.update(timestamp, prices, actions)
    
    # Verify costs were deducted
    trade = engine.trades[0]
    assert trade['cost'] > 0
    assert state['portfolio_value'] < 10000

def test_multiple_assets():
    """Test trading multiple assets simultaneously"""
    engine = BacktestEngine(initial_capital=10000)
    
    # Trade multiple assets
    timestamp = pd.Timestamp('2024-01-01')
    prices = {'BTC': 100.0, 'ETH': 50.0}
    actions = {'BTC': 0.3, 'ETH': 0.3}
    
    state = engine.update(timestamp, prices, actions)
    
    # Verify positions
    assert len(state['positions']) == 2
    assert 'BTC' in state['positions']
    assert 'ETH' in state['positions']

if __name__ == '__main__':
    pytest.main([__file__])
