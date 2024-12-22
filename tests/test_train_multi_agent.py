import pytest
import numpy as np
import os
from pathlib import Path
import shutil
import mlflow
from unittest.mock import patch
from gymnasium import spaces
import logging

from training.train_multi_agent import (
    train_multi_agent_system,
    evaluate_agents,
    setup_logging
)
from agents.strategies.single.ppo_agent import PPOAgent

# Setup debug logging for tests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DummyMultiAgentEnv:
    """Simple environment for testing multi-agent system"""
    
    def __init__(self):
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
        
        self.agents = ["agent1", "agent2"]
        self.max_steps = 100  # Add maximum steps
        self.current_step = 0  # Add step counter
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        # Generate random OHLCV data
        self.data = np.random.randn(60, 5).astype(np.float32)
        # Ensure close is within high/low
        self.data[:, 3] = (self.data[:, 1] + self.data[:, 2]) / 2
        
        self.current_step = 0  # Reset step counter
        observations = {agent_id: self.data for agent_id in self.agents}
        return observations, {}
    
    def step(self, actions):
        """Take a step in the environment"""
        self.current_step += 1
        logger.debug(f"Step {self.current_step} of {self.max_steps}")
        
        # Generate next state
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = np.random.randn(5)
        self.data[-1, 3] = (self.data[-1, 1] + self.data[-1, 2]) / 2
        
        # Calculate rewards (simplified)
        rewards = {
            agent_id: float(action * (self.data[-1, 3] - self.data[-2, 3]))
            for agent_id, action in actions.items()
        }
        
        # Return observations for all agents
        observations = {agent_id: self.data for agent_id in self.agents}
        
        # Episode ends after max_steps
        done = self.current_step >= self.max_steps
        dones = {agent_id: done for agent_id in self.agents}
        truncated = {agent_id: False for agent_id in self.agents}
        
        info = {
            agent_id: {
                "portfolio_value": 10000 * (1 + np.random.randn() * 0.1)
            }
            for agent_id in self.agents
        }
        
        return observations, rewards, dones, truncated, info

@pytest.fixture
def clean_mlflow(tmp_path):
    """Setup and cleanup MLflow artifacts"""
    mlflow_path = tmp_path / "mlruns"
    mlflow_path.mkdir(parents=True)
    os.environ["MLFLOW_TRACKING_URI"] = str(mlflow_path)
    yield
    shutil.rmtree(mlflow_path)

@pytest.fixture
def env():
    """Create a test environment"""
    return DummyMultiAgentEnv()

@pytest.fixture
def agents(env):
    """Create test agents"""
    return {
        "agent1": PPOAgent(env.observation_space, env.action_space),
        "agent2": PPOAgent(env.observation_space, env.action_space)
    }

def test_setup_logging(tmp_path):
    """Test logging setup"""
    log_dir = tmp_path / "logs"
    setup_logging(str(log_dir))
    
    assert log_dir.exists()
    log_files = list(log_dir.glob("training_*.log"))
    assert len(log_files) == 1

def test_evaluate_agents(env, agents):
    """Test agent evaluation"""
    eval_metrics = evaluate_agents(env, agents, num_episodes=2)
    
    for agent_id in agents.keys():
        assert agent_id in eval_metrics
        assert "mean_reward" in eval_metrics[agent_id]
        assert "mean_portfolio_value" in eval_metrics[agent_id]
        assert "sharpe_ratio" in eval_metrics[agent_id]
        assert "max_drawdown" in eval_metrics[agent_id]
        
        assert isinstance(eval_metrics[agent_id]["mean_reward"], float)
        assert isinstance(eval_metrics[agent_id]["mean_portfolio_value"], float)
        assert isinstance(eval_metrics[agent_id]["sharpe_ratio"], float)
        assert isinstance(eval_metrics[agent_id]["max_drawdown"], float)
        
        assert eval_metrics[agent_id]["max_drawdown"] >= 0
        assert eval_metrics[agent_id]["max_drawdown"] <= 1

def test_train_multi_agent_system(env, agents, tmp_path, clean_mlflow):
    """Test multi-agent training system"""
    save_path = tmp_path / "models"
    num_episodes = 5
    save_freq = 2
    eval_freq = 2
    
    metrics = train_multi_agent_system(
        env=env,
        agents=agents,
        num_episodes=num_episodes,
        save_path=str(save_path),
        save_freq=save_freq,
        eval_freq=eval_freq
    )
    
    # Check metrics structure
    for agent_id in agents.keys():
        assert agent_id in metrics
        assert "episode_rewards" in metrics[agent_id]
        assert "portfolio_values" in metrics[agent_id]
        assert "policy_losses" in metrics[agent_id]
        assert "value_losses" in metrics[agent_id]
        
        assert len(metrics[agent_id]["episode_rewards"]) == num_episodes
    
    # Check saved models
    expected_checkpoints = num_episodes // save_freq
    for agent_id in agents.keys():
        checkpoints = list(save_path.glob(f"{agent_id}_episode_*.pt"))
        assert len(checkpoints) == expected_checkpoints
    
    # Check MLflow logs
    mlflow_path = Path(os.environ["MLFLOW_TRACKING_URI"])
    assert mlflow_path.exists()
    
    # Check experiment artifacts
    experiment_id = mlflow.get_experiment_by_name("trading_bot_multi_agent_training").experiment_id
    runs = mlflow.search_runs(experiment_id)
    assert len(runs) == 1

def test_train_multi_agent_system_error_handling(env, agents, tmp_path):
    """Test error handling in training system"""
    # Mock MLflow to avoid actual MLflow operations
    with patch('mlflow.set_experiment'), \
         patch('mlflow.start_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'):
        
        # Test with invalid agent
        invalid_agents = agents.copy()
        invalid_agents["invalid"] = None
        
        with pytest.raises(AttributeError):
            train_multi_agent_system(
                env=env,
                agents=invalid_agents,
                num_episodes=1,
                save_path=str(tmp_path)
            )
        
        # Test with invalid environment
        with pytest.raises(AttributeError):
            train_multi_agent_system(
                env=None,
                agents=agents,
                num_episodes=1,
                save_path=str(tmp_path)
            )
