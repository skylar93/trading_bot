import numpy as np
import pytest
from gymnasium import spaces
from agents.strategies.agent_factory import create_agent

OBS_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(20, 5), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

BASE_CONFIG = {
    "rsi_window": 14,
    "bb_window": 20,
    "bb_std": 2.0,
    "oversold_threshold": 30,
    "overbought_threshold": 70,
}

@pytest.fixture
def agent():
    return create_agent("MeanReversion", config=BASE_CONFIG,
                        observation_space=OBS_SPACE, action_space=ACT_SPACE)

def test_agent_initialization(agent):
    assert agent.rsi_window == 14
    assert agent.bb_window == 20
    assert agent.bb_std == 2.0
    assert agent.oversold_threshold == 30
    assert agent.overbought_threshold == 70

def test_rsi_calculation(agent):
    # Upward trend → high RSI
    prices = np.array([100.0] * 10 + [100.0 + i for i in range(5)])
    rsi = agent._calculate_rsi(prices)
    assert rsi > 50

    # Downward trend → low RSI
    prices = np.array([100.0] * 10 + [100.0 - i for i in range(5)])
    rsi = agent._calculate_rsi(prices)
    assert rsi < 50

def test_bollinger_bands_calculation(agent):
    # Flat prices → bands equal price
    prices = np.array([10.0] * 20)
    upper, lower = agent._calculate_bollinger_bands(prices)
    assert upper == 10.0
    assert lower == 10.0

    # Volatile prices → upper > mean, lower < mean
    prices = np.array([10.0 + i for i in range(20)])
    upper, lower = agent._calculate_bollinger_bands(prices)
    assert upper > np.mean(prices)
    assert lower < np.mean(prices)

def test_get_action_mean_reversion(agent):
    # Overbought state
    state = np.zeros((20, 5))
    state[:15, 3] = 100.0
    for i in range(5):
        state[15 + i, 3] = 100.0 * (1.02 ** (i + 1))
    action = agent.get_action(state)
    assert action[0] <= 0.5  # Should tend to sell

    # Oversold state
    state = np.zeros((20, 5))
    state[:15, 3] = 100.0
    for i in range(5):
        state[15 + i, 3] = 100.0 * (0.98 ** (i + 1))
    action = agent.get_action(state)
    assert action[0] >= -0.5  # Should tend to buy

def test_train_step_reward_modification(agent):
    base_price = 100.0
    state = np.zeros((20, 5))
    for i in range(10):
        state[i, 3] = base_price
    for i in range(5):
        state[10 + i, 3] = base_price * (1.01 ** (i + 1))
    peak_price = state[14, 3]
    for i in range(5):
        state[15 + i, 3] = peak_price * (0.85 ** (i + 1))
    for i in range(20):
        dv = state[i, 3] * 0.02
        state[i, 0] = state[i, 3] + dv
        state[i, 1] = state[i, 3] - dv
        state[i, 2] = state[i, 3]
        state[i, 4] = 1_000_000

    next_state = state.copy()
    next_state[-1, 3] *= 1.15

    metrics = agent.train_step(
        state=state, action=np.array([0.8]),
        reward=0.1, next_state=next_state, done=False
    )
    assert metrics["reversion_reward"] > 0
    assert metrics["rsi_value"] <= 30
    assert "bb_upper_dist" in metrics
