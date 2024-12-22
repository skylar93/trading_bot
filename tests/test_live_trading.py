import pytest
import asyncio
import numpy as np
from envs.live_trading_env import LiveTradingEnvironment


@pytest.mark.skip(
    reason="Live trading tests can cause freezing and require network connection"
)
@pytest.mark.asyncio
async def test_live_trading_env():
    """Test live trading environment functionality"""

    # Create environment
    env = LiveTradingEnvironment(
        symbol="BTC/USDT", initial_balance=10000.0, window_size=30
    )

    try:
        # Test reset with timeout
        async with asyncio.timeout(10):  # 10 second timeout
            obs, info = await env.reset()
            assert obs.shape == (30, 12)  # window_size x n_features

        # Test step
        action = np.array([0.5])  # Buy 50% of balance
        obs, reward, terminated, truncated, info = env.step(action)

        # Verify observation shape
        assert obs.shape == (30, 12)

        # Check portfolio value is calculated
        assert info["portfolio_value"] > 0

        # Check position and balance updated
        assert info["position"] > 0  # Should have bought some
        assert info["balance"] < 10000.0  # Should have spent some money

        # Test market data
        assert info["current_price"] > 0

    finally:
        # Cleanup
        try:
            async with asyncio.timeout(5):
                await env.websocket.stop()
                await asyncio.sleep(1)  # Give time for cleanup
        except Exception as e:
            pytest.skip(f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_live_trading_env())
