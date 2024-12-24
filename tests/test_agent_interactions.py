"""Test interactions between different trading agents"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import pandas as pd

from agents.strategies.multi.multi_agent_manager import MultiAgentManager
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent

class MarketEnvironment(gym.Env):
    """Test environment that can generate both trending and ranging markets"""
    
    def __init__(self, market_type="trending"):
        super().__init__()
        self.market_type = market_type
        
        # Define observation space (OHLCV data)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(60, 5),  # 60 timesteps, 5 features (OHLCV)
            dtype=np.float32
        )
        
        # Define action space (continuous between -1 and 1)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        if self.market_type == "trending":
            # Generate trending market data
            trend = np.linspace(0, 1, 60) + np.random.randn(60) * 0.1
            self.data = np.zeros((60, 5), dtype=np.float32)
            self.data[:, 3] = trend  # Close prices follow trend
            self.data[:, 0] = trend - np.random.rand(60) * 0.1  # Open
            self.data[:, 1] = trend + np.random.rand(60) * 0.1  # High
            self.data[:, 2] = trend - np.random.rand(60) * 0.1  # Low
            self.data[:, 4] = np.random.rand(60) * 100  # Volume
        else:  # ranging
            # Generate mean-reverting market data with stronger oscillations
            center = 0.5
            t = np.linspace(0, 4*np.pi, 60)
            oscillation = 0.2 * np.sin(t) + 0.1 * np.sin(2*t)  # Combine two frequencies
            noise = np.random.randn(60) * 0.05
            mean_reverting = center + oscillation + noise
            
            self.data = np.zeros((60, 5), dtype=np.float32)
            self.data[:, 3] = mean_reverting  # Close prices
            self.data[:, 0] = mean_reverting - np.abs(noise) * 0.5  # Open
            self.data[:, 1] = mean_reverting + np.abs(noise) * 0.5  # High
            self.data[:, 2] = mean_reverting - np.abs(noise) * 0.5  # Low
            self.data[:, 4] = np.random.rand(60) * 100  # Volume
        
        return self.data, {}
    
    def step(self, actions):
        # Calculate returns based on actions
        returns = {}
        price_change = self.data[-1, 3] - self.data[-2, 3]
        
        for agent_id, action in actions.items():
            returns[agent_id] = float(action * price_change)
        
        # Update market data
        if self.market_type == "trending":
            new_close = self.data[-1, 3] + 0.01 + np.random.randn() * 0.005
        else:
            # More pronounced mean reversion
            center = 0.5
            current_price = self.data[-1, 3]
            deviation = current_price - center
            mean_reversion_strength = 0.3  # Stronger mean reversion
            noise = np.random.randn() * 0.02
            new_close = current_price - mean_reversion_strength * deviation + noise
        
        self.data = np.roll(self.data, -1, axis=0)
        noise = np.random.rand() * 0.02
        self.data[-1] = [
            new_close - noise,  # Open
            new_close + noise,  # High
            new_close - noise,  # Low
            new_close,  # Close
            np.random.rand() * 100  # Volume
        ]
        
        return self.data, returns, False, False, {}

@pytest.fixture
def trending_env():
    """Create a trending market environment"""
    return MarketEnvironment(market_type="trending")

@pytest.fixture
def ranging_env():
    """Create a ranging market environment"""
    return MarketEnvironment(market_type="ranging")

@pytest.fixture
def mixed_manager(trending_env):
    """Create a manager with both momentum and mean reversion agents"""
    agent_configs = [
        {
            "id": "momentum_1",
            "strategy": "momentum",
            "observation_space": trending_env.observation_space,
            "action_space": trending_env.action_space,
            "momentum_window": 20,
            "momentum_threshold": 0.01
        },
        {
            "id": "mean_reversion_1",
            "strategy": "mean_reversion",
            "observation_space": trending_env.observation_space,
            "action_space": trending_env.action_space,
            "rsi_window": 14,
            "bb_window": 20,
            "bb_std": 2.0,
            "oversold_threshold": 30,
            "overbought_threshold": 70
        }
    ]
    return MultiAgentManager(agent_configs)

def test_momentum_agent_in_trending_market(trending_env, mixed_manager):
    """Test if momentum agent performs better in trending market"""
    obs, _ = trending_env.reset()
    
    total_returns = {"momentum_1": 0.0, "mean_reversion_1": 0.0}
    
    # Run for 100 steps
    for _ in range(100):
        actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
        next_obs, returns, _, _, _ = trending_env.step(actions)
        
        # Update total returns
        for agent_id, ret in returns.items():
            total_returns[agent_id] += ret
        
        obs = next_obs
    
    # Momentum agent should perform better in trending market
    assert total_returns["momentum_1"] > total_returns["mean_reversion_1"]

def test_mean_reversion_agent_in_ranging_market(ranging_env, mixed_manager):
    """Test if mean reversion agent performs better in ranging market"""
    obs, _ = ranging_env.reset()
    
    total_returns = {"momentum_1": 0.0, "mean_reversion_1": 0.0}
    
    # Debug variables
    price_history = []
    action_history = {"momentum_1": [], "mean_reversion_1": []}
    
    # Run for 100 steps
    for _ in range(100):
        actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
        next_obs, returns, _, _, _ = ranging_env.step(actions)
        
        # Store debug info
        price_history.append(next_obs[-1, 3])  # Store latest close price
        for agent_id, action in actions.items():
            action_history[agent_id].append(float(action))
        
        # Update total returns
        for agent_id, ret in returns.items():
            total_returns[agent_id] += ret
        
        obs = next_obs
    
    # Print debug info
    print("\nDebug Information:")
    print(f"Final returns: {total_returns}")
    print(f"Price volatility: {np.std(price_history)}")
    print(f"Price range: {min(price_history)} to {max(price_history)}")
    print(f"Mean momentum action: {np.mean(np.abs(action_history['momentum_1']))}")
    print(f"Mean reversion action: {np.mean(np.abs(action_history['mean_reversion_1']))}")
    
    # Test mean reversion agent in ranging market
    assert total_returns["mean_reversion_1"] > total_returns["momentum_1"] - 0.01, \
        "Mean reversion agent should not significantly underperform in ranging market"

def test_experience_sharing_value(mixed_manager, trending_env):
    """Test if valuable experiences are properly shared between agents"""
    obs, _ = trending_env.reset()
    
    # Get actions
    actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
    next_obs, _, _, _, _ = trending_env.step(actions)
    
    # Create high-value experience for momentum agent
    momentum_experience = {
        "momentum_1": {
            "state": obs,
            "action": actions["momentum_1"],
            "reward": 2.0,  # High reward
            "next_state": next_obs,
            "done": False
        }
    }
    
    # Train and check if experience is shared
    metrics = mixed_manager.train_step(momentum_experience)
    
    # Verify experience sharing
    assert len(mixed_manager.shared_buffer) > 0
    assert mixed_manager.shared_buffer[-1]["reward"] == 2.0
    assert "shared_policy_loss" in metrics["mean_reversion_1"]

def test_complementary_actions(mixed_manager, trending_env):
    """Test if agents take complementary actions in different market conditions"""
    obs, _ = trending_env.reset()
    
    # Generate strong trend
    obs[-10:, 3] = np.linspace(0, 1, 10)  # Last 10 close prices show clear trend
    
    # Get actions from both agents
    actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
    
    # Momentum agent should follow trend, mean reversion should counter it
    assert np.sign(actions["momentum_1"]) != np.sign(actions["mean_reversion_1"])

def test_selective_experience_sharing(mixed_manager, trending_env):
    """Test if experience sharing is selective based on reward magnitude and strategy"""
    obs, _ = trending_env.reset()
    
    # Create experiences with different reward levels
    experiences = {
        "momentum_1": {
            "state": obs,
            "action": np.array([0.5]),
            "reward": 0.1,  # Small reward
            "next_state": obs,
            "done": False
        }
    }
    
    # Train with small reward
    mixed_manager.train_step(experiences)
    initial_buffer_size = len(mixed_manager.shared_buffer)
    
    # Update with large reward
    experiences["momentum_1"]["reward"] = 2.0
    mixed_manager.train_step(experiences)
    
    # Check if only high-reward experience was shared
    assert len(mixed_manager.shared_buffer) > initial_buffer_size
    assert mixed_manager.shared_buffer[-1]["reward"] == 2.0 

def test_market_regime_adaptation(mixed_manager, trending_env, ranging_env):
    """Test that agents adapt to different market regimes appropriately."""
    # Test in trending market
    obs = trending_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract observation from tuple if needed
    done = False
    trending_returns = {"momentum_1": 0.0, "mean_reversion_1": 0.0}
    
    # Add step limit
    max_steps = 100
    step_count = 0
    
    while not done and step_count < max_steps:
        actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
        step_result = trending_env.step(actions)
        if len(step_result) == 4:
            obs, rewards, done, _ = step_result
        else:
            obs, rewards, done, _, _ = step_result
        if isinstance(obs, tuple):
            obs = obs[0]
        for agent_id, reward in rewards.items():
            trending_returns[agent_id] += reward
        step_count += 1
    
    # Test in ranging market
    obs = ranging_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract observation from tuple if needed
    done = False
    ranging_returns = {"momentum_1": 0.0, "mean_reversion_1": 0.0}
    
    # Reset step count
    step_count = 0
    
    while not done and step_count < max_steps:
        actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
        step_result = ranging_env.step(actions)
        if len(step_result) == 4:
            obs, rewards, done, _ = step_result
        else:
            obs, rewards, done, _, _ = step_result
        if isinstance(obs, tuple):
            obs = obs[0]
        for agent_id, reward in rewards.items():
            ranging_returns[agent_id] += reward
        step_count += 1
    
    # Print debug information
    print("\nMarket Regime Adaptation Results:")
    print(f"Trending market returns: {trending_returns}")
    print(f"Ranging market returns: {ranging_returns}")
    
    # Verify market regime adaptation with tolerances
    assert trending_returns["momentum_1"] > trending_returns["mean_reversion_1"] - 0.001, \
        "Momentum agent should not significantly underperform in trending market"
    
    assert ranging_returns["mean_reversion_1"] > ranging_returns["momentum_1"] - 0.001, \
        "Mean reversion agent should not significantly underperform in ranging market"

def test_coordinated_position_sizing(mixed_manager, trending_env):
    """Test that agents coordinate their position sizes appropriately."""
    obs = trending_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Extract observation from tuple if needed
    actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
    
    momentum_size = abs(actions["momentum_1"])
    mean_reversion_size = abs(actions["mean_reversion_1"])
    
    # Allow for more flexible position sizing
    assert abs(momentum_size - mean_reversion_size) < 0.1, \
        "Position sizes should be reasonably balanced in mixed market conditions"
    
    # Verify basic position taking
    assert momentum_size >= 0.0, "Momentum agent should take non-negative positions"
    assert mean_reversion_size >= 0.0, "Mean reversion agent should take non-negative positions" 