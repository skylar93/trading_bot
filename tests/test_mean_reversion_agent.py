import numpy as np
import pytest
from gymnasium import spaces
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent

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
        "rsi_window": 14,
        "bb_window": 20,
        "bb_std": 2.0,
        "oversold_threshold": 30,
        "overbought_threshold": 70
    }

def test_agent_initialization(sample_config):
    agent = MeanReversionPPOAgent(sample_config)
    assert agent.rsi_window == 14
    assert agent.bb_window == 20
    assert agent.bb_std == 2.0
    assert agent.oversold_threshold == 30
    assert agent.overbought_threshold == 70

def test_rsi_calculation(sample_config):
    agent = MeanReversionPPOAgent(sample_config)
    
    # Test RSI with upward trend
    prices = np.array([100.0] * 10 + [100.0 + i for i in range(5)])  # Last 5 prices trending up
    rsi = agent._calculate_rsi(prices)
    assert rsi > 50  # RSI should be high in upward trend
    
    # Test RSI with downward trend
    prices = np.array([100.0] * 10 + [100.0 - i for i in range(5)])  # Last 5 prices trending down
    rsi = agent._calculate_rsi(prices)
    assert rsi < 50  # RSI should be low in downward trend

def test_bollinger_bands_calculation(sample_config):
    agent = MeanReversionPPOAgent(sample_config)
    
    # Test with flat prices
    prices = np.array([10.0] * 20)
    upper, lower = agent._calculate_bollinger_bands(prices)
    assert upper == 10.0  # Upper band should equal price when no volatility
    assert lower == 10.0  # Lower band should equal price when no volatility
    
    # Test with volatile prices
    prices = np.array([10.0 + i for i in range(20)])
    upper, lower = agent._calculate_bollinger_bands(prices)
    assert upper > np.mean(prices)  # Upper band should be above mean
    assert lower < np.mean(prices)  # Lower band should be below mean

def test_get_action_mean_reversion(sample_config):
    agent = MeanReversionPPOAgent(sample_config)
    
    # Create a state where price is at BB upper and RSI is high (overbought)
    state = np.zeros((20, 5))
    state[:15, 3] = 100.0  # Set initial close prices
    for i in range(5):
        state[15+i, 3] = 100.0 * (1.02 ** (i+1))  # Each step up 2%
    
    action = agent.get_action(state)
    assert action <= 0  # Should prefer selling in overbought condition
    
    # Create a state where price is at BB lower and RSI is low (oversold)
    state = np.zeros((20, 5))
    state[:15, 3] = 100.0  # Set initial close prices
    for i in range(5):
        state[15+i, 3] = 100.0 * (0.98 ** (i+1))  # Each step down 2%
    
    action = agent.get_action(state)
    assert action >= 0  # Should prefer buying in oversold condition

def test_train_step_reward_modification(sample_config):
    agent = MeanReversionPPOAgent(sample_config)
    
    # Create a price series with initial stability followed by sharp decline
    state = np.zeros((20, 5))
    
    # Set initial stable period
    base_price = 100.0
    for i in range(10):
        state[i, 3] = base_price  # Flat prices
    
    # Add a period of small gains to establish baseline
    for i in range(5):
        state[10+i, 3] = base_price * (1.01 ** (i+1))  # Small gains
    
    # Then create a very sharp decline (each step down 15%)
    peak_price = state[14, 3]
    for i in range(5):
        state[15+i, 3] = peak_price * (0.85 ** (i+1))  # Each step down 15%
    
    # Ensure OHLC values are consistent and add some volatility
    for i in range(20):
        daily_volatility = state[i, 3] * 0.02  # 2% daily volatility
        state[i, 0] = state[i, 3] + daily_volatility  # High
        state[i, 1] = state[i, 3] - daily_volatility  # Low
        state[i, 2] = state[i, 3]  # Open same as close for simplicity
        state[i, 4] = 1000000  # Constant volume
    
    next_state = state.copy()
    # Price bounces back significantly (mean reversion)
    bounce_pct = 0.15  # 15% bounce
    next_state[-1, 3] *= (1.0 + bounce_pct)
    next_state[-1, 0] = next_state[-1, 3] * 1.02  # High
    next_state[-1, 1] = next_state[-1, 3] * 0.98  # Low
    next_state[-1, 2] = next_state[-1, 3]  # Open
    
    # Print initial debug information
    features = agent._calculate_reversion_features(state)
    print(f"Initial RSI: {features[0]}")
    print(f"Initial BB Upper Distance: {features[1]}")
    print(f"Initial BB Lower Distance: {features[2]}")
    print(f"Price series: {state[:, 3]}")
    print(f"Last 5 price changes (%): {[(state[i+1, 3] / state[i, 3] - 1) * 100 for i in range(-6, -1)]}")
    
    # Test reward modification for strong mean reversion trade
    metrics = agent.train_step(
        state=state,
        action=np.array([0.8]),  # Very strong buy in oversold (> 0.2 threshold)
        reward=0.1,
        next_state=next_state,
        done=False
    )
    
    # Print final debug information
    print(f"Final RSI: {metrics['rsi_value']}")
    print(f"Final BB Lower Distance: {metrics['bb_lower_dist']}")
    print(f"Final Reversion Reward: {metrics['reversion_reward']}")
    print(f"Price bounce: {(next_state[-1, 3] / state[-1, 3] - 1) * 100:.2f}%")
    
    # Verify the results
    assert metrics["reversion_reward"] > 0, f"Should get positive reversion reward, got {metrics['reversion_reward']}"
    assert metrics["rsi_value"] < 30, f"RSI should be oversold (<30), got {metrics['rsi_value']}"
    assert metrics["bb_lower_dist"] < 0.02, f"Price should be near BB lower band, got {metrics['bb_lower_dist']}"
    assert "bb_upper_dist" in metrics
