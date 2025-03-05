import pytest
import numpy as np
import pandas as pd
from training.env_factory import create_env

@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns."""
    dates = pd.date_range(start="2025-01-01", periods=200, freq="h")
    df = pd.DataFrame(
        {
            "$open": np.random.normal(100, 1, 200).cumsum(),
            "$high": np.random.normal(100, 1, 200).cumsum(),
            "$low": np.random.normal(100, 1, 200).cumsum(),
            "$close": np.random.normal(100, 1, 200).cumsum(),
            "$volume": np.abs(np.random.randn(200) * 100),
        },
        index=dates,
    )
    return df

@pytest.fixture
def env_config():
    """Create basic environment configuration."""
    return {
        "type": "single_asset_rl",
        "initial_capital": 10000.0,
        "trading_fee": 0.001,
        "window_size": 10,
        "max_position_size": 1.0,
    }

def test_env_shapes(sample_data, env_config):
    """Test observation and action shapes from the environment."""
    # Create environment
    env = create_env(env_config, sample_data)
    
    # Test reset
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (env_config["window_size"], 5), f"Expected shape (10, 5), got {obs.shape}"
    
    # Test step
    action = np.array([0.5])  # Action in [0, 1]
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Verify shapes
    assert isinstance(next_obs, np.ndarray), "Next observation should be numpy array"
    assert next_obs.shape == (env_config["window_size"], 5), f"Expected shape (10, 5), got {next_obs.shape}"
    assert isinstance(reward, float), "Reward should be float"
    assert isinstance(done, bool), "Done should be boolean"
    assert isinstance(truncated, bool), "Truncated should be boolean"
    assert isinstance(info, dict), "Info should be dictionary"
    
    # Test multiple steps
    for _ in range(5):
        action = np.array([np.random.random()])
        next_obs, reward, done, truncated, info = env.step(action)
        assert next_obs.shape == (env_config["window_size"], 5), "Shape should remain consistent"
        
def test_env_edge_cases(sample_data, env_config):
    """Test environment behavior in edge cases."""
    env = create_env(env_config, sample_data)
    
    # Test invalid actions
    obs, _ = env.reset()
    action = np.array([1.5])  # Action > 1
    next_obs, reward, done, _, _ = env.step(action)
    assert next_obs.shape == (env_config["window_size"], 5), "Shape should be maintained even with invalid action"
    
    # Test episode end
    for _ in range(len(sample_data) - env_config["window_size"]):
        next_obs, _, done, truncated, _ = env.step(np.array([0.5]))
        if done or truncated:
            break
    assert done or truncated, "Episode should end when data is exhausted" 