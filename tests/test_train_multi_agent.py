import pytest
import numpy as np
import os
from pathlib import Path
import shutil
import mlflow
from unittest.mock import patch
from gymnasium import spaces
import logging
import time

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
        # Define observation space (OHLCV data + other agents' positions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(60, 7),  # 60 timesteps, 5 OHLCV + 2 other agent positions
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
        self.max_steps = 100
        self.current_step = 0
        self.initial_balance = 10000.0
        self.market_impact = 0.001  # Price impact per unit of net position
        self.positions = {agent_id: 0.0 for agent_id in self.agents}
        self.balances = {agent_id: self.initial_balance for agent_id in self.agents}
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        self.data = np.random.randn(60, 5).astype(np.float32)
        self.data[:, 3] = (self.data[:, 1] + self.data[:, 2]) / 2
        
        self.current_step = 0
        self.positions = {agent_id: 0.0 for agent_id in self.agents}
        self.balances = {agent_id: self.initial_balance for agent_id in self.agents}
        
        observations = self._get_observations()
        return observations, {}
    
    def _get_observations(self):
        """Get observations for all agents including other agents' positions"""
        observations = {}
        for agent_id in self.agents:
            # Create observation with OHLCV data
            obs = np.copy(self.data)
            # Add other agents' positions as features
            other_positions = np.zeros((obs.shape[0], 2))
            pos_idx = 0
            for other_id in self.agents:
                if other_id != agent_id:
                    other_positions[:, pos_idx] = self.positions[other_id]
                    pos_idx += 1
            observations[agent_id] = np.concatenate([obs, other_positions], axis=1)
        return observations
    
    def step(self, actions):
        """Take a step in the environment"""
        self.current_step += 1
        logger.debug(f"Step {self.current_step} of {self.max_steps}")
        
        # Calculate net position impact on price
        net_position = sum(self.positions.values())
        price_impact = net_position * self.market_impact
        
        # Update market data with price impact
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = np.random.randn(5)
        self.data[-1, 3] = (self.data[-1, 1] + self.data[-1, 2]) / 2 + price_impact
        
        # Update positions and calculate rewards
        rewards = {}
        for agent_id, action in actions.items():
            # Update position
            old_position = self.positions[agent_id]
            position_change = float(action[0])
            self.positions[agent_id] += position_change
            
            # Calculate reward components
            price_change = self.data[-1, 3] - self.data[-2, 3]
            position_reward = old_position * price_change
            transaction_cost = abs(position_change) * 0.001
            
            # Competition/Cooperation component
            other_positions = [pos for aid, pos in self.positions.items() if aid != agent_id]
            if all(pos * self.positions[agent_id] > 0 for pos in other_positions):
                # Cooperation bonus if positions align
                cooperation_reward = 0.001 * abs(self.positions[agent_id])
            else:
                # Competition penalty if positions oppose
                cooperation_reward = -0.001 * abs(self.positions[agent_id])
            
            rewards[agent_id] = position_reward - transaction_cost + cooperation_reward
            
            # Update balance
            self.balances[agent_id] += rewards[agent_id]
        
        observations = self._get_observations()
        
        done = self.current_step >= self.max_steps
        dones = {agent_id: done for agent_id in self.agents}
        truncated = {agent_id: False for agent_id in self.agents}
        
        info = {
            agent_id: {
                "portfolio_value": self.balances[agent_id],
                "position": self.positions[agent_id],
                "price_impact": price_impact,
                "transaction_cost": abs(actions[agent_id][0]) * 0.001
            }
            for agent_id in self.agents
        }
        
        return observations, rewards, dones, truncated, info

@pytest.fixture
def clean_mlflow(tmp_path):
    """Setup and cleanup MLflow artifacts"""
    # Create a completely new MLflow directory for each test
    mlflow_path = tmp_path / "mlruns"
    if os.path.exists(mlflow_path):
        shutil.rmtree(mlflow_path)
    os.makedirs(mlflow_path, exist_ok=True)
    
    # Set MLflow tracking URI to the new directory
    mlflow.set_tracking_uri(f"file://{str(mlflow_path)}")
    
    # Reset MLflow state
    mlflow.end_run()
    
    # Create .trash directory to handle deleted experiments
    trash_path = mlflow_path / ".trash"
    os.makedirs(trash_path, exist_ok=True)
    
    yield
    
    # Cleanup after tests
    try:
        # End any active runs
        mlflow.end_run()
        
        # Delete all experiments
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        for experiment in experiments:
            try:
                # First mark as deleted
                client.delete_experiment(experiment.experiment_id)
                # Then permanently delete
                exp_dir = mlflow_path / experiment.experiment_id
                if os.path.exists(exp_dir):
                    shutil.rmtree(exp_dir)
                trash_exp_dir = trash_path / experiment.experiment_id
                if os.path.exists(trash_exp_dir):
                    shutil.rmtree(trash_exp_dir)
            except Exception as e:
                logger.warning(f"Failed to delete experiment {experiment.experiment_id}: {e}")
        
        # Remove the entire mlruns directory
        if os.path.exists(mlflow_path):
            shutil.rmtree(mlflow_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup MLflow directory: {e}")

def generate_unique_experiment_name(base_name):
    """Generate a unique experiment name by appending a timestamp"""
    return f"{base_name}_{int(time.time() * 1000)}"

@pytest.fixture
def unique_experiment_name():
    """Fixture to provide unique experiment names"""
    def _generate(base_name):
        return generate_unique_experiment_name(base_name)
    return _generate

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

@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow for testing"""
    def mock_create_experiment(name):
        return "mock-experiment-id"
    
    def mock_set_experiment(name):
        pass
    
    def mock_start_run():
        class MockContextManager:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                pass
        return MockContextManager()
    
    def mock_log_metrics(metrics, step=None):
        pass
    
    def mock_log_params(params):
        pass
    
    monkeypatch.setattr(mlflow, "create_experiment", mock_create_experiment)
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_experiment)
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(mlflow, "log_params", mock_log_params)

def test_train_multi_agent_system(env, agents, tmp_path, clean_mlflow):
    """Test multi-agent training system"""
    save_path = tmp_path / "models"
    num_episodes = 5
    save_freq = 2
    eval_freq = 2
    
    experiment_name = generate_unique_experiment_name("test_train_multi_agent_system")
    
    metrics = train_multi_agent_system(
        env=env,
        agents=agents,
        num_episodes=num_episodes,
        save_path=str(save_path),
        save_freq=save_freq,
        eval_freq=eval_freq,
        experiment_name=experiment_name
    )
    
    # Check metrics structure
    for agent_id in agents.keys():
        assert agent_id in metrics
        assert "episode_rewards" in metrics[agent_id]
        assert "portfolio_values" in metrics[agent_id]
        assert "policy_losses" in metrics[agent_id]

def test_quick_training(env, agents, tmp_path, clean_mlflow):
    """Test quick training to verify basic functionality"""
    logger.info("Starting quick training test...")
    save_path = tmp_path / "models"
    num_episodes = 2
    save_freq = 1
    eval_freq = 1
    
    experiment_name = generate_unique_experiment_name("test_quick_training")
    
    metrics = train_multi_agent_system(
        env=env,
        agents=agents,
        num_episodes=num_episodes,
        save_path=str(save_path),
        save_freq=save_freq,
        eval_freq=eval_freq,
        experiment_name=experiment_name
    )
    
    # Basic checks
    assert all(len(metrics[agent_id]["episode_rewards"]) == num_episodes for agent_id in agents)
    assert all(len(metrics[agent_id]["portfolio_values"]) == num_episodes for agent_id in agents)
    assert all(len(metrics[agent_id]["policy_losses"]) > 0 for agent_id in agents)
