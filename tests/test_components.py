"""Unit tests for individual RL components.

Features:
- Tests for environment state/action shapes
- Tests for policy/value network input/output shapes
- Tests for PPO buffer operations
- Tests for agent state normalization and actions

Implementation Notes:
- Uses small window sizes and feature counts for clarity
- Explicitly checks tensor shapes at each step
- Includes edge case handling
"""

import pytest
import numpy as np
import torch
import pandas as pd
from gymnasium import spaces

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from agents.models.architectures.mlp import PolicyNetwork
from agents.models.architectures.value_mlp import ValueNetwork
from buffers.ppo_buffer import PPOBuffer
from agents.strategies.single.ppo_agent import PPOAgent

# Constants for testing
WINDOW_SIZE = 10
N_FEATURES = 5  # $open, $high, $low, $close, $volume
BATCH_SIZE = 4
DEVICE = "cpu"

def create_dummy_df(size: int = 100) -> pd.DataFrame:
    """Create dummy price data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=size, freq="1h")
    data = {
        "$open": np.random.randn(size),
        "$high": np.random.randn(size),
        "$low": np.random.randn(size),
        "$close": np.random.randn(size),
        "$volume": np.abs(np.random.randn(size))
    }
    return pd.DataFrame(data, index=dates)

class TestEnvironment:
    """Test environment state/action shapes and transitions."""
    
    @pytest.fixture
    def env(self):
        df = create_dummy_df()
        return SingleAssetRLTradingEnv(
            data=df,
            window_size=WINDOW_SIZE,
            initial_capital=10000,
            trading_fee=0.001
        )
    
    def test_reset_shape(self, env):
        """Test that env.reset() returns correct observation shape."""
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (WINDOW_SIZE, N_FEATURES), f"Expected shape ({WINDOW_SIZE}, {N_FEATURES})"
        
    def test_step_shape(self, env):
        """Test that env.step() maintains observation shape."""
        env.reset()
        action = np.array([0.5])  # Buy 50% of balance
        obs, reward, done, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (WINDOW_SIZE, N_FEATURES), f"Expected shape ({WINDOW_SIZE}, {N_FEATURES})"
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(done, bool), "Done should be boolean"

class TestNetworks:
    """Test policy and value network shapes."""
    
    @pytest.fixture
    def obs_space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(WINDOW_SIZE, N_FEATURES),
            dtype=np.float32
        )
    
    @pytest.fixture
    def action_space(self):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
    
    def test_policy_network_shapes(self, obs_space, action_space):
        """Test PolicyNetwork input/output shapes."""
        network = PolicyNetwork(obs_space, action_space).to(DEVICE)
        
        # Test single observation
        obs = torch.randn(1, WINDOW_SIZE, N_FEATURES).to(DEVICE)
        mean, std = network(obs)
        
        assert mean.shape == (1, 1), "Policy mean should have shape (batch_size, action_dim)"
        assert std.shape == (1, 1), "Policy std should have shape (batch_size, action_dim)"
        
        # Test batch of observations
        obs_batch = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES).to(DEVICE)
        mean_batch, std_batch = network(obs_batch)
        
        assert mean_batch.shape == (BATCH_SIZE, 1), "Batched mean should have shape (batch_size, action_dim)"
        assert std_batch.shape == (BATCH_SIZE, 1), "Batched std should have shape (batch_size, action_dim)"
    
    def test_value_network_shapes(self, obs_space):
        """Test ValueNetwork input/output shapes."""
        network = ValueNetwork(obs_space).to(DEVICE)
        
        # Test single observation
        obs = torch.randn(1, WINDOW_SIZE, N_FEATURES).to(DEVICE)
        value = network(obs)
        
        assert value.shape == (1, 1), "Value should have shape (batch_size, 1)"
        
        # Test batch of observations
        obs_batch = torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES).to(DEVICE)
        value_batch = network(obs_batch)
        
        assert value_batch.shape == (BATCH_SIZE, 1), "Batched value should have shape (batch_size, 1)"

class TestPPOBuffer:
    """Test PPO buffer operations and shapes."""
    
    @pytest.fixture
    def buffer(self):
        return PPOBuffer(
            obs_shape=(WINDOW_SIZE, N_FEATURES),
            action_shape=(1,),
            size=BATCH_SIZE,
            gamma=0.99,
            gae_lambda=0.95,
            device=DEVICE
        )
    
    def test_empty_buffer(self, buffer):
        """Test handling of empty buffer."""
        assert len(buffer) == 0, "New buffer should be empty"
        batch = buffer.get_batch()
        assert batch is None, "Empty buffer should return None for get_batch"
    
    def test_single_experience(self, buffer):
        """Test adding and retrieving single experience."""
        # Create dummy experience
        exp = {
            "state": np.random.randn(WINDOW_SIZE, N_FEATURES),
            "action": np.array([0.5]),
            "reward": 1.0,
            "done": False,
            "value": 0.5,
            "log_prob": -0.5
        }
        
        buffer.append(exp)
        assert len(buffer) == 1, "Buffer should have one experience"
        
        # Test computing advantages
        buffer.compute_advantages(last_value=np.array([0.0]))
        
        # Get batch (will be size 1)
        states, actions, log_probs, returns, advantages, values = buffer.get_batch(batch_size=1)
        
        assert states.shape == (1, WINDOW_SIZE, N_FEATURES), "Incorrect states shape"
        assert actions.shape == (1, 1), "Incorrect actions shape"
        assert log_probs.shape == (1,), "Incorrect log_probs shape"
        assert returns.shape == (1,), "Incorrect returns shape"
        assert advantages.shape == (1,), "Incorrect advantages shape"
        assert values.shape == (1,), "Incorrect values shape"
    
    def test_full_buffer(self, buffer):
        """Test buffer with multiple experiences."""
        # Fill buffer
        for _ in range(BATCH_SIZE):
            exp = {
                "state": np.random.randn(WINDOW_SIZE, N_FEATURES),
                "action": np.array([0.5]),
                "reward": 1.0,
                "done": False,
                "value": 0.5,
                "log_prob": -0.5
            }
            buffer.append(exp)
        
        assert len(buffer) == BATCH_SIZE, f"Buffer should have {BATCH_SIZE} experiences"
        
        # Test computing advantages
        buffer.compute_advantages(last_value=np.array([0.0]))
        
        # Get full batch
        states, actions, log_probs, returns, advantages, values = buffer.get_batch()
        
        assert states.shape == (BATCH_SIZE, WINDOW_SIZE, N_FEATURES), "Incorrect states shape"
        assert actions.shape == (BATCH_SIZE, 1), "Incorrect actions shape"
        assert log_probs.shape == (BATCH_SIZE,), "Incorrect log_probs shape"
        assert returns.shape == (BATCH_SIZE,), "Incorrect returns shape"
        assert advantages.shape == (BATCH_SIZE,), "Incorrect advantages shape"
        assert values.shape == (BATCH_SIZE,), "Incorrect values shape"

class TestPPOAgent:
    """Test PPO agent operations and shapes."""
    
    @pytest.fixture
    def agent(self):
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(WINDOW_SIZE, N_FEATURES),
            dtype=np.float32
        )
        action_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        return PPOAgent(
            observation_space=obs_space,
            action_space=action_space,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )
    
    def test_get_action_shape(self, agent):
        """Test agent.get_action() shapes."""
        # Test with numpy array
        state = np.random.randn(WINDOW_SIZE, N_FEATURES)
        action = agent.get_action(state)
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (1,), "Action should have shape (1,)"
        
        # Test with DataFrame
        df_state = pd.DataFrame(
            state,
            columns=["$open", "$high", "$low", "$close", "$volume"]
        )
        action = agent.get_action(df_state)
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (1,), "Action should have shape (1,)"
    
    def test_train_step_shapes(self, agent):
        """Test shapes during agent.train_step()."""
        # Create dummy transition
        state = np.random.randn(WINDOW_SIZE, N_FEATURES)
        action = np.array([0.5])
        reward = 1.0
        next_state = np.random.randn(WINDOW_SIZE, N_FEATURES)
        done = False
        
        # Test initial buffer filling
        for i in range(BATCH_SIZE - 1):
            agent.train_step(state, action, reward, next_state, done)
            assert len(agent.buffer) == i + 1, f"Buffer should have {i + 1} experiences during filling"
        
        # Add final experience to reach BATCH_SIZE
        agent.train_step(state, action, reward, next_state, done)
        assert len(agent.buffer) == BATCH_SIZE, "Buffer should be full"
        
        # Explicitly call update_if_buffer_ready to trigger training and buffer reset
        metrics = agent.update_if_buffer_ready()
        assert len(agent.buffer) == 0, "Buffer should be empty after update"
        
        # Test buffer starts filling again
        agent.train_step(state, action, reward, next_state, done)
        assert len(agent.buffer) == 1, "Buffer should have 1 experience after reset"

if __name__ == "__main__":
    pytest.main(["-v", "test_components.py"]) 