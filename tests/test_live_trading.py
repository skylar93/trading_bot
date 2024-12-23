import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from envs.live_trading_env import LiveTradingEnvironment, OrderStatus
from datetime import datetime

@pytest.fixture
def live_env():
    """Create a test instance of LiveTradingEnvironment"""
    # Create mock exchange
    exchange = AsyncMock()
    exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.1,
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Create mock websocket
    websocket = AsyncMock()
    window_size = 10  # Match environment's window_size
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=window_size, freq='1min'),
        'open': np.ones(window_size) * 50000.0,
        'high': np.ones(window_size) * 50100.0,
        'low': np.ones(window_size) * 49900.0,
        'close': np.ones(window_size) * 50000.0,
        'volume': np.ones(window_size) * 1.0,
        'bid': np.ones(window_size) * 49950.0,
        'ask': np.ones(window_size) * 50050.0,
        'bid_volume': np.ones(window_size) * 0.5,
        'ask_volume': np.ones(window_size) * 0.5,
        'rsi': np.ones(window_size) * 50.0,
        'macd': np.zeros(window_size)
    })
    df.set_index('timestamp', inplace=True)
    websocket.get_current_data = Mock(return_value=df)
    websocket._latest_ticker = {'last': 50000.0}
    websocket._latest_orderbook = {
        'bids': [[49900.0, 1.0]],
        'asks': [[50100.0, 1.0]]
    }
    websocket.start = AsyncMock(return_value=None)
    websocket.stop = AsyncMock(return_value=None)
    websocket.get_latest_price = Mock(return_value=50000.0)
    
    # Create environment
    env = LiveTradingEnvironment(
        symbol='BTC/USD',
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=window_size,
        test_mode=True,
        exchange=exchange,
        websocket=websocket
    )
    
    return env

@pytest.mark.asyncio
async def test_environment_initialization(live_env):
    """Test environment initialization"""
    obs, info = await live_env.reset()
    assert obs.shape == (10, 13)  # window_size x n_features
    assert info['portfolio_value'] == 10000.0

@pytest.mark.asyncio
async def test_basic_trading(live_env):
    """Test basic trading functionality"""
    # Buy
    obs, info = await live_env.reset()
    
    # Mock successful order
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.1,
        'price': 50000.0,
        'remaining': 0.0
    }
    
    action = np.array([0.2])  # Buy 20% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    assert info['position'] > 0  # Position should increase after buying
    
    # Sell
    live_env.exchange.create_order.return_value = {'id': 'test_order_2', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order_2',
        'status': 'closed',
        'filled': 0.05,  # Sell half of position
        'price': 50000.0,
        'remaining': 0.0
    }
    
    action = np.array([-0.5])  # Sell 50% of position
    obs, reward, terminated, truncated, new_info = await live_env.step(action)
    assert new_info['position'] < info['position']  # Position should decrease after selling

@pytest.mark.asyncio
async def test_market_data(live_env):
    """Test market data handling"""
    await live_env.reset()
    
    # Check market data
    assert live_env.websocket.get_current_data.call_count > 0
    assert live_env.websocket._latest_ticker['last'] == 50000.0
    assert len(live_env.websocket._latest_orderbook['bids']) > 0
    assert len(live_env.websocket._latest_orderbook['asks']) > 0

@pytest.mark.asyncio
async def test_portfolio_value(live_env):
    """Test portfolio value calculation"""
    # Initialize environment
    obs, info = await live_env.reset()
    initial_value = float(live_env.portfolio_value)
    initial_balance = float(live_env.balance)
    
    # Mock successful order
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.1,  # Fill 10% of order
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Execute a buy trade
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, done, truncated, info = await live_env.step(action)
    
    # Portfolio value should be updated
    assert live_env.portfolio_value != initial_value
    assert info['position'] > 0  # Should have a position
    assert info['balance'] < initial_balance  # Balance should decrease due to purchase
    
    # Record portfolio value before price change
    portfolio_before_price_change = float(live_env.portfolio_value)
    
    # Mock price increase (2% increase)
    new_price = 51000.0
    live_env.websocket.get_latest_price = Mock(return_value=new_price)
    live_env.websocket.get_current_data = Mock(return_value=pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='1min'),
        'open': np.ones(10) * new_price,
        'high': np.ones(10) * (new_price * 1.002),
        'low': np.ones(10) * (new_price * 0.998),
        'close': np.ones(10) * new_price,
        'volume': np.ones(10) * 1.0,
        'bid': np.ones(10) * (new_price * 0.999),
        'ask': np.ones(10) * (new_price * 1.001),
        'bid_volume': np.ones(10) * 0.5,
        'ask_volume': np.ones(10) * 0.5,
        'rsi': np.ones(10) * 50.0,
        'macd': np.zeros(10)
    }).set_index('timestamp'))
    
    # Get new observation to update internal state
    obs, info = await live_env.reset()
    
    # Portfolio value should increase with price
    new_portfolio_value = float(live_env.portfolio_value)
    assert new_portfolio_value > portfolio_before_price_change
    assert abs(new_portfolio_value - portfolio_before_price_change) > 1.0  # Should be a significant change

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
