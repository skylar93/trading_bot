import numpy as np
import pytest
from gymnasium import spaces
from agents.strategies.agent_factory import create_agent

OBS_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

@pytest.fixture
def agent():
    return create_agent("Momentum", config={"momentum_window": 10, "momentum_threshold": 0.02},
                        observation_space=OBS_SPACE, action_space=ACT_SPACE)

def test_agent_initialization(agent):
    assert agent.momentum_window == 10
    assert agent.momentum_threshold == 0.02

def test_momentum_calculation(agent):
    base_price = 100.0

    # Upward momentum
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (1.01 ** i)
    features = agent._calculate_momentum_features(state)
    assert features[0] > 0  # positive momentum
    assert features[2] > 0  # positive trend

    # Downward momentum
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (0.99 ** i)
    features = agent._calculate_momentum_features(state)
    assert features[0] < 0  # negative momentum
    assert features[2] < 0  # negative trend

def test_volatility_calculation(agent):
    # Low volatility (flat)
    state = np.zeros((20, 5))
    state[:, 3] = 100.0
    features = agent._calculate_momentum_features(state)
    volatility = features[1]
    assert isinstance(float(volatility), float)

    # High volatility (alternating)
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = 100.0 + ((-1) ** i) * 10
    features = agent._calculate_momentum_features(state)
    assert features[1] > 0

def test_action_momentum_bias(agent):
    base_price = 100.0

    # Strong upward momentum → should buy
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (1.02 ** i)
    action = agent.get_action(state)
    assert action >= 0

    # Strong downward momentum → should sell
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (0.98 ** i)
    action = agent.get_action(state)
    assert action <= 0

def test_momentum_reward_modification(agent):
    base_price = 100.0
    state = np.zeros((20, 5))
    for i in range(20):
        state[i, 3] = base_price * (1.02 ** i)

    next_state = np.zeros((20, 5))
    next_state[:-1] = state[1:]
    next_state[-1, 3] = state[-1, 3] * 1.02

    # Following momentum (buy in uptrend)
    metrics = agent.train_step(state=state, action=np.array([0.8]), reward=0.1,
                               next_state=next_state, done=False)
    assert metrics["momentum_reward"] > 0
    assert metrics["momentum_value"] > 0
    assert metrics["momentum_trend"] > 0

    # Against momentum (sell in uptrend)
    metrics = agent.train_step(state=state, action=np.array([-0.8]), reward=0.1,
                               next_state=next_state, done=False)
    assert metrics["momentum_reward"] == 0
