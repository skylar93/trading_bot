import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import ccxt
from envs.live_trading_env import LiveTradingEnvironment, OrderStatus, Order
import pandas as pd

@pytest.fixture
def mock_ccxt():
    """Mock CCXT exchange"""
    mock = AsyncMock()
    mock.fetch_ticker = AsyncMock(return_value={
        'bid': 50000.0,
        'ask': 50100.0,
        'last': 50050.0,
        'volume': 100.0,
    })
    mock.create_order = AsyncMock(return_value={'id': 'test_order', 'status': 'open'})
    mock.fetch_order = AsyncMock()
    mock.cancel_order = AsyncMock(return_value={'status': 'canceled'})
    mock.close = AsyncMock()
    return mock

@pytest.fixture
def live_env():
    """Create a live trading environment with mocked dependencies"""
    # Create mock websocket
    mock_websocket = AsyncMock()
    window_size = 30  # Match environment's window_size
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
    mock_websocket.get_current_data = Mock(return_value=df)
    mock_websocket._latest_ticker = {'last': 50000.0}
    mock_websocket._latest_orderbook = {
        'bids': [[49900.0, 1.0]],
        'asks': [[50100.0, 1.0]]
    }
    mock_websocket.start = AsyncMock(return_value=None)
    mock_websocket.stop = AsyncMock(return_value=None)
    mock_websocket.get_latest_price = Mock(return_value=50000.0)
    
    # Create mock exchange with network delay simulation
    mock_exchange = AsyncMock()
    mock_exchange.create_order = AsyncMock(side_effect=ccxt.NetworkError())  # Always fail with network error
    mock_exchange.fetch_order = AsyncMock(return_value={
        'id': 'test_order',
        'status': 'canceled',  # Always return canceled status
        'filled': 0.0,  # No fill
        'price': 50000.0,
        'remaining': 0.0
    })
    mock_exchange.cancel_order = AsyncMock(return_value={'status': 'canceled'})
    
    # Create environment
    env = LiveTradingEnvironment(
        symbol='BTC/USD',
        initial_balance=10000.0,
        trading_fee=0.001,
        window_size=window_size,
        test_mode=True,
        websocket=mock_websocket,
        exchange=mock_exchange
    )
    
    return env

@pytest.mark.asyncio
async def test_network_delay_handling(live_env):
    """Test handling of network delays"""
    # Initialize environment
    obs, info = await live_env.reset()
    initial_position = info['position']
    
    # Mock network delay
    live_env.exchange.create_order.side_effect = ccxt.NetworkError()
    
    # Place order
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    
    # Verify order was not executed
    assert info['position'] == initial_position  # Position should not change
    assert reward <= 0  # Should get penalty for failed trade
    assert live_env.exchange.create_order.call_count >= 1  # Should attempt to create order

@pytest.mark.asyncio
async def test_partial_fill_handling(live_env):
    """Test handling of partially filled orders"""
    # Initialize environment
    obs, info = await live_env.reset()
    
    # Mock partial fill
    live_env.exchange.create_order.side_effect = None  # Reset side effect
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.05,  # Only 5% filled
        'price': 50000.0,
        'remaining': 0.95
    }
    
    # Place order
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    
    # Verify partial fill
    assert info['position'] > 0
    assert info['position'] < 0.5  # Should be partially filled

@pytest.mark.asyncio
async def test_rate_limit_handling(live_env):
    """Test handling of rate limits"""
    # Initialize environment
    obs, info = await live_env.reset()
    initial_position = info['position']
    
    # Mock rate limit error
    live_env.exchange.create_order.side_effect = ccxt.RateLimitExceeded()
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'canceled',  # Order should be canceled
        'filled': 0.0,  # No fill
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Place order
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    
    # Verify order was not executed
    assert info['position'] == initial_position  # Position should not change
    assert reward <= 0  # Should get penalty for failed trade

@pytest.mark.asyncio
async def test_order_cancellation_with_partial_fill(live_env):
    """Test cancellation of partially filled orders"""
    # Initialize environment
    obs, info = await live_env.reset()
    
    # Mock partial fill
    live_env.exchange.create_order.side_effect = None  # Reset side effect
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.05,  # Only 5% filled
        'price': 50000.0,
        'remaining': 0.95
    }
    
    # Place order
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    
    # Cancel order
    await live_env.cancel_all_orders()
    
    # Verify partial fill
    assert info['position'] > 0
    assert info['position'] < 0.5  # Should be partially filled
    assert len(live_env.active_orders) == 0  # All orders should be canceled

@pytest.mark.asyncio
async def test_multiple_order_management(live_env):
    """Test managing multiple orders"""
    # Reset mock behavior
    live_env.exchange.create_order.side_effect = None
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.1,
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Place multiple orders
    obs1, reward1, done1, truncated1, info1 = await live_env.step(np.array([0.5]))  # Buy 50%
    obs2, reward2, done2, truncated2, info2 = await live_env.step(np.array([0.3]))  # Buy 30%
    obs3, reward3, done3, truncated3, info3 = await live_env.step(np.array([-0.4]))  # Sell 40%
    
    assert len(live_env.active_orders) <= 3

@pytest.mark.asyncio
async def test_error_handling(live_env):
    """Test handling of exchange errors"""
    # Initialize environment
    obs, info = await live_env.reset()
    initial_position = info['position']
    
    # Mock exchange error
    live_env.exchange.create_order.side_effect = ccxt.ExchangeError()
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'canceled',  # Order should be canceled
        'filled': 0.0,  # No fill
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Place order
    action = np.array([0.5])  # Buy 50% of balance
    obs, reward, terminated, truncated, info = await live_env.step(action)
    
    # Verify order was not executed
    assert info['position'] == initial_position  # Position should not change
    assert reward <= 0  # Should get penalty for failed trade

@pytest.mark.asyncio
async def test_cleanup_with_active_orders(live_env):
    """Test cleanup with active orders"""
    # Reset mock behavior
    live_env.exchange.create_order.side_effect = None
    live_env.exchange.create_order.return_value = {'id': 'test_order', 'status': 'open'}
    live_env.exchange.fetch_order.return_value = {
        'id': 'test_order',
        'status': 'closed',
        'filled': 0.1,
        'price': 50000.0,
        'remaining': 0.0
    }
    
    # Place some orders
    obs1, reward1, done1, truncated1, info1 = await live_env.step(np.array([0.5]))  # Buy 50%
    obs2, reward2, done2, truncated2, info2 = await live_env.step(np.array([0.3]))  # Buy 30%
    
    # Cleanup
    await live_env.cleanup()
    assert len(live_env.active_orders) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
