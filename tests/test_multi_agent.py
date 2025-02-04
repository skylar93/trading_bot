import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.strategies.multi.multi_agent_manager import MultiAgentManager
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent

class DummyMultiAgentEnv(gym.Env):
    """Simple environment for testing multi-agent system"""
    
    def __init__(self):
        super().__init__()
        
        # Define observation space (OHLCV data + some features)
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
        
        # Generate artificial trend data instead of random
        # Create a linear trend from 1.0 to 2.0 over 60 timesteps
        trend = np.linspace(1.0, 2.0, 60)
        
        # Add some noise to make it more realistic
        noise = np.random.randn(60) * 0.1
        close_prices = trend + noise
        
        # Generate OHLCV data with the trending close prices
        self.data = np.zeros((60, 5), dtype=np.float32)
        self.data[:, 3] = close_prices  # Close prices follow the trend
        self.data[:, 0] = close_prices - np.random.rand(60) * 0.1  # Open slightly lower
        self.data[:, 1] = close_prices + np.random.rand(60) * 0.1  # High slightly higher
        self.data[:, 2] = close_prices - np.random.rand(60) * 0.1  # Low slightly lower
        self.data[:, 4] = np.random.rand(60) * 1000  # Random volume
        
        return self.data, {}
    
    def step(self, actions):
        # Generate next state with trend continuation
        self.data = np.roll(self.data, -1, axis=0)
        last_close = self.data[-2, 3]
        trend_increment = 1.0 / 60  # Same trend as in reset
        
        # New close price continues the trend
        new_close = last_close + trend_increment + np.random.randn() * 0.1
        self.data[-1] = [
            new_close - np.random.rand() * 0.1,  # Open
            new_close + np.random.rand() * 0.1,  # High
            new_close - np.random.rand() * 0.1,  # Low
            new_close,  # Close
            np.random.rand() * 1000  # Volume
        ]
        
        # Calculate rewards (simplified)
        rewards = {}
        for agent_id, action in actions.items():
            rewards[agent_id] = float(action * (self.data[-1, 3] - self.data[-2, 3]))
        
        # Always return False for done (continuous trading)
        done = False
        truncated = False
        
        return self.data, rewards, done, truncated, {}

@pytest.fixture
def env():
    """Create a test environment"""
    return DummyMultiAgentEnv()

@pytest.fixture
def manager(env):
    """Create a test manager with momentum agents"""
    agent_configs = [
        {
            "id": "momentum_1",
            "strategy": "momentum",
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "momentum_window": 20,
            "momentum_threshold": 0.01
        }
    ]
    return MultiAgentManager(agent_configs)

@pytest.fixture
def multi_manager(env):
    """Create a test manager with multiple momentum agents"""
    agent_configs = [
        {
            "id": "momentum_1",
            "strategy": "momentum",
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "momentum_window": 20,
            "momentum_threshold": 0.01
        },
        {
            "id": "momentum_2",
            "strategy": "momentum",
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "momentum_window": 30,
            "momentum_threshold": 0.02
        }
    ]
    return MultiAgentManager(agent_configs)

def test_multi_agent_initialization(env, manager):
    """Test multi-agent system initialization"""
    assert len(manager.agents) == 1
    assert isinstance(manager.agents["momentum_1"], MomentumPPOAgent)

def test_multi_agent_action_selection(env, manager):
    """Test multi-agent action selection"""
    # Get initial observation
    obs, _ = env.reset()
    
    # Get actions from all agents
    actions = manager.act({"momentum_1": obs})
    
    assert isinstance(actions, dict)
    assert "momentum_1" in actions
    assert isinstance(actions["momentum_1"], np.ndarray)
    assert actions["momentum_1"].shape == (1,)
    assert -1 <= actions["momentum_1"] <= 1

def test_multi_agent_training_step(env, manager):
    """Test multi-agent training step"""
    # Get initial observation
    obs, _ = env.reset()
    
    # Get actions
    actions = manager.act({"momentum_1": obs})
    
    # Take step in environment
    next_obs, rewards, done, truncated, info = env.step(actions)
    
    # Create experience dictionary
    experiences = {
        "momentum_1": {
            "state": obs,
            "action": actions["momentum_1"],
            "reward": rewards["momentum_1"],
            "next_state": next_obs,
            "done": done
        }
    }
    
    # Train agents
    metrics = manager.train_step(experiences)
    
    assert isinstance(metrics, dict)
    assert "momentum_1" in metrics
    assert isinstance(metrics["momentum_1"], dict)

def test_multi_agent_experience_sharing(env, multi_manager):
    """Test experience sharing between agents"""
    # Get initial observation
    obs, _ = env.reset()
    
    # Get actions
    actions = multi_manager.act({"momentum_1": obs, "momentum_2": obs})
    
    # Take step in environment
    next_obs, rewards, done, truncated, info = env.step(actions)
    
    # Create experience dictionary with positive reward
    experiences = {
        "momentum_1": {
            "state": obs,
            "action": actions["momentum_1"],
            "reward": 1.0,  # Positive reward to ensure sharing
            "next_state": next_obs,
            "done": done
        }
    }
    
    # Train agents
    metrics = multi_manager.train_step(experiences)
    
    assert len(multi_manager.shared_buffer) > 0
    assert isinstance(metrics, dict)
    assert "momentum_1" in metrics

def test_multi_agent_save_load(env, manager, tmp_path):
    """Test saving and loading multi-agent system"""
    # Get initial observation
    obs, _ = env.reset()
    
    # Get actions from first manager
    actions1 = manager.act({"momentum_1": obs}, deterministic=True)
    
    # Save manager
    save_path = str(tmp_path / "test_save")
    manager.save(save_path)
    
    # Create new manager and load
    new_manager = MultiAgentManager([{
        "id": "momentum_1",
        "strategy": "momentum",
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "momentum_window": 20,
        "momentum_threshold": 0.01
    }])
    new_manager.load(save_path)
    
    # Get actions from loaded manager
    actions2 = new_manager.act({"momentum_1": obs}, deterministic=True)
    
    np.testing.assert_array_almost_equal(
        actions1["momentum_1"],
        actions2["momentum_1"]
    )
