import pytest
import asyncio
import numpy as np
from datetime import datetime
from envs.paper_trading_env import PaperTradingEnvironment

@pytest.mark.asyncio
async def test_paper_trading_basic():
    """Test basic paper trading functionality"""
    env = PaperTradingEnvironment(
        initial_balance=10000.0,
        max_position_size=0.5,
        stop_loss_pct=0.02
    )
    
    try:
        # Start environment
        await env.start()
        
        # Wait for initial data
        await asyncio.sleep(5)
        
        # Verify initial state
        metrics = env.get_metrics()
        assert metrics['total_return'] == 0
        assert metrics['current_balance'] == 10000.0
        assert metrics['current_position'] == 0
        
        # Execute a buy order
        result = await env.execute_order(0.25)  # Buy 25% of max position
        assert result['success']
        assert result['trade']['type'] == 'buy'
        assert env.position > 0
        
        # Wait for price updates
        await asyncio.sleep(5)
        
        # Get current data
        df = env.get_current_data()
        assert not df.empty
        assert 'open' in df.columns
        assert 'close' in df.columns
        
        # Verify metrics are being updated
        metrics = env.get_metrics()
        assert metrics['current_position'] > 0
        assert metrics['trade_count'] == 0  # No closed trades yet
        
        # Close position
        result = await env.execute_order(-1.0)
        assert result['success']
        assert result['trade']['type'] == 'sell'
        assert env.position == 0
        
        # Verify trade history
        metrics = env.get_metrics()
        assert metrics['trade_count'] == 1
        assert len(env.trade_history) == 1
        
    finally:
        await env.stop()

@pytest.mark.asyncio
async def test_risk_management():
    """Test risk management features"""
    env = PaperTradingEnvironment(
        initial_balance=10000.0,
        max_position_size=0.5,  # Max 50% of portfolio
        stop_loss_pct=0.02  # 2% stop loss
    )
    
    try:
        await env.start()
        await asyncio.sleep(5)
        
        # Try to buy more than max_position_size
        result = await env.execute_order(1.0)
        assert result['success']
        portfolio_value = env.get_portfolio_value()
        position_size = (env.position * env._latest_price) / portfolio_value
        assert position_size <= 0.5
        
        # Verify stop loss price is set
        assert env.stop_loss_price > 0
        old_price = env._latest_price
        
        # Simulate price drop to trigger stop loss
        env._latest_price = old_price * 0.97  # 3% drop
        await env._check_stop_loss()
        
        # Position should be closed
        assert env.position == 0
        assert env.stop_loss_price == 0
        
    finally:
        await env.stop()

if __name__ == "__main__":
    asyncio.run(test_paper_trading_basic())
    asyncio.run(test_risk_management())