import sys
import os
import numpy as np
import pandas as pd
import logging

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the environment
from envs.multi_agent_env import MultiAgentTradingEnv

def create_sample_data():
    """Generate synthetic OHLCV data for testing"""
    rows = 100
    np.random.seed(42)
    
    # Start with a base price
    base_price = 100
    
    # Generate synthetic price data
    close_prices = np.cumsum(np.random.normal(0, 1, rows)) + base_price
    open_prices = close_prices + np.random.normal(0, 0.5, rows)
    high_prices = np.maximum(close_prices, open_prices) + np.random.uniform(0, 2, rows)
    low_prices = np.minimum(close_prices, open_prices) - np.random.uniform(0, 2, rows)
    volumes = np.random.randint(100, 10000, rows)
    
    # Create a DataFrame with the required column names
    df = pd.DataFrame({
        "$open": open_prices,
        "$high": high_prices,
        "$low": low_prices,
        "$close": close_prices,
        "$volume": volumes
    })
    
    return df

def create_agent_configs():
    """Create agent configurations for testing"""
    return [
        {
            "id": "agent1",
            "strategy": "momentum",
            "initial_balance": 10000.0
        },
        {
            "id": "agent2",
            "strategy": "mean_reversion",
            "initial_balance": 10000.0
        }
    ]

def test_portfolio_values_initialization():
    """Test that portfolio_values is initialized correctly"""
    # Create test data
    sample_data = create_sample_data()
    agent_configs = create_agent_configs()
    
    # Create environment
    env = MultiAgentTradingEnv(
        data=sample_data,
        agent_configs=agent_configs,
        window_size=10
    )
    
    # Reset environment
    observations, info = env.reset()
    
    # Check that portfolio_values exists and is initialized correctly
    assert hasattr(env, 'portfolio_values'), "Environment missing portfolio_values attribute"
    assert 'agent1' in env.portfolio_values, "Agent1 missing from portfolio_values"
    assert 'agent2' in env.portfolio_values, "Agent2 missing from portfolio_values"
    assert len(env.portfolio_values['agent1']) == 1, "Portfolio values should have initial value only"
    assert env.portfolio_values['agent1'][0] == 10000.0, "Initial portfolio value incorrect"
    
    logger.info("Portfolio values initialized correctly")
    
    # Take a step to ensure portfolio_values is updated
    actions = {
        'agent1': np.array([0.5]),
        'agent2': np.array([-0.5])
    }
    
    next_obs, rewards, dones, truncated, info = env.step(actions)
    
    # Check that portfolio_values was updated
    assert len(env.portfolio_values['agent1']) == 2, "Portfolio values not updated after step"
    
    logger.info("Portfolio values updated after step")
    logger.info(f"Portfolio values: {env.portfolio_values}")
    
    return True

if __name__ == "__main__":
    # Run the test directly
    try:
        result = test_portfolio_values_initialization()
        print(f"Test result: {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc() 