import numpy as np
import pytest
from gymnasium import spaces
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent

@pytest.fixture
def sample_config():
    return {
        "observation_space": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20, 5),  # window_size=20, features=5 (OHLCV)
            dtype=np.float32
        ),
        "action_space": spaces.Box(
            low=-1, high=1,
            shape=(1,),
            dtype=np.float32
        ),
        "momentum_window": 10,
        "momentum_threshold": 0.02
    }

def test_agent_initialization(sample_config):
    agent = MomentumPPOAgent(sample_config)
    assert agent.momentum_window == 10
    assert agent.momentum_threshold == 0.02

def test_momentum_calculation(sample_config):
    agent = MomentumPPOAgent(sample_config)
    
    # Test upward momentum
    state = np.zeros((20, 5))
    base_price = 100.0
    for i in range(20):
        state[i, 3] = base_price * (1.01 ** i)  # Each step up 1%
    
    features = agent._calculate_momentum_features(state)
    momentum = features[0]
    trend = features[2]
    
    assert momentum > 0  # Should have positive momentum
    assert trend > 0  # Should have positive trend
    
    # Test downward momentum
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (0.99 ** i)  # Each step down 1%
    
    features = agent._calculate_momentum_features(state)
    momentum = features[0]
    trend = features[2]
    
    assert momentum < 0  # Should have negative momentum
    assert trend < 0  # Should have negative trend

def test_volatility_calculation(sample_config):
    agent = MomentumPPOAgent(sample_config)
    
    # Test low volatility
    state = np.zeros((20, 5))
    state[:, 3] = 100.0  # Flat price
    
    features = agent._calculate_momentum_features(state)
    volatility = features[1]
    
    assert volatility == 0.0  # Should have zero volatility
    
    # Test high volatility
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = 100.0 + ((-1) ** i) * 10  # Alternating +/-10
    
    features = agent._calculate_momentum_features(state)
    volatility = features[1]
    
    assert volatility > 5.0  # Should have high volatility

def test_action_momentum_bias(sample_config):
    agent = MomentumPPOAgent(sample_config)
    
    # Test strong upward momentum
    state = np.zeros((20, 5))
    base_price = 100.0
    for i in range(20):
        state[i, 3] = base_price * (1.02 ** i)  # Each step up 2%
    
    action = agent.get_action(state)
    assert action >= 0  # Should prefer buying in upward momentum
    
    # Test strong downward momentum
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (0.98 ** i)  # Each step down 2%
    
    action = agent.get_action(state)
    assert action <= 0  # Should prefer selling in downward momentum

def test_momentum_reward_modification(sample_config):
    agent = MomentumPPOAgent(sample_config)
    
    # Create state with strong upward momentum
    state = np.zeros((20, 5))
    base_price = 100.0
    for i in range(20):
        state[i, 3] = base_price * (1.02 ** i)  # Each step up 2%
    
    next_state = np.zeros((20, 5))
    next_state[:-1] = state[1:]  # Copy all but last
    next_state[-1, 3] = state[-1, 3] * 1.02  # Continue trend
    
    # Test reward for following momentum (buying in uptrend)
    metrics = agent.train_step(
        state=state,
        action=np.array([0.8]),  # Strong buy
        reward=0.1,
        next_state=next_state,
        done=False
    )
    
    assert metrics["momentum_reward"] > 0, "Should get positive momentum reward for following trend"
    assert metrics["momentum_value"] > 0, "Should detect positive momentum"
    assert metrics["momentum_trend"] > 0, "Should detect positive trend"
    
    # Test reward for going against momentum (selling in uptrend)
    metrics = agent.train_step(
        state=state,
        action=np.array([-0.8]),  # Strong sell
        reward=0.1,
        next_state=next_state,
        done=False
    )
    
    assert metrics["momentum_reward"] == 0, "Should not get momentum reward for going against trend"
