import pytest
import asyncio
import numpy as np
from envs.live_trading_env import LiveTradingEnvironment

@pytest.mark.asyncio
async def test_live_trading_env():
    """Test live trading environment functionality"""
    
    # Create environment
    env = LiveTradingEnvironment(
        symbol="BTC/USDT",
        initial_balance=10000.0,
        window_size=30
    )
    
    try:
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (30, 12)  # window_size x n_features
        
        # Test step
        action = np.array([0.5])  # Buy 50% of balance
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify observation shape
        assert obs.shape == (30, 12)
        
        # Check portfolio value is calculated
        assert info['portfolio_value'] > 0
        
        # Check position and balance updated
        assert info['position'] > 0  # Should have bought some
        assert info['balance'] < 10000.0  # Should have spent some money
        
        # Test market data
        assert info['current_price'] > 0
        
        # Wait for a few updates
        await asyncio.sleep(5)
        
        # Test selling
        action = np.array([-0.5])  # Sell 50% of position
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Position should be reduced
        assert info['position'] >= 0
        assert info['balance'] > 0
        
    finally:
        # Cleanup
        env.close()

if __name__ == "__main__":
    asyncio.run(test_live_trading_env())