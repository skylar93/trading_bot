"""Tests for training system"""

import pytest
import numpy as np
import pandas as pd
import mlflow
from training.train import TrainingPipeline, Trainer
from training.utils.unified_mlflow_manager import MLflowManager
from envs.trading_env import TradingEnvironment
from agents.strategies.single.ppo_agent import PPOAgent
import gym
from gym import spaces


@pytest.fixture
def config():
    """Create training configuration"""
    return {
        "env": {
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
            "window_size": 20,
        },
        "model": {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 32},
        "training": {"total_timesteps": 1000},
    }


@pytest.fixture
def trainer(config, mlflow_test_context):
    """Create trainer instance with MLflow integration"""
    trainer = TrainingPipeline(config)
    yield trainer


@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns"""
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="h")
    data = pd.DataFrame(
        {
            "$open": np.random.randn(1000).cumsum() + 100,
            "$high": np.random.randn(1000).cumsum() + 102,
            "$low": np.random.randn(1000).cumsum() + 98,
            "$close": np.random.randn(1000).cumsum() + 100,
            "$volume": np.abs(np.random.randn(1000) * 1000),
        },
        index=dates,
    )
    return data


@pytest.fixture
def mock_env():
    """Create a mock trading environment"""

    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10, 5),  # (window_size, features)
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )
            self.initial_balance = 10000.0
            self.returns = []
            self.steps = 0
            self.max_steps = 100  # End episode after 100 steps

        def reset(self):
            self.returns = []
            self.steps = 0
            return np.zeros((10, 5)), {}

        def step(self, action):
            self.steps += 1
            reward = np.random.normal(0, 1)
            self.returns.append(reward)
            done = (
                self.steps >= self.max_steps
            )  # End episode if max steps reached
            return (
                np.zeros((10, 5)),
                reward,
                done,
                False,
                {
                    "portfolio_value": self.initial_balance
                    * (1 + sum(self.returns))
                },
            )

    return MockEnv()


@pytest.fixture
def mock_agent():
    """Create a mock agent"""

    class MockAgent:
        def __init__(self):
            self.observation_space = None
            self.action_space = None

        def get_action(self, state):
            return np.array([0.0])

        def train_step(self, state, action, reward, next_state, done):
            return {"loss": 0.0}

    return MockAgent()


def test_trainer_initialization(mlflow_test_context, mock_env, mock_agent):
    """Test trainer initialization"""
    config = {
        "env": {
            "window_size": 10,
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
        },
        "model": {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.99},
        "training": {"num_episodes": 2, "total_timesteps": 1000},
        "experiment_name": mlflow_test_context.experiment_name,
        "mlflow_tracking_dir": "test_mlflow",
    }

    trainer = Trainer(config)
    trainer.env = mock_env  # Set environment after initialization
    trainer.agent = mock_agent  # Set agent after initialization

    assert trainer.env == mock_env
    assert trainer.agent == mock_agent
    assert trainer.mlflow_manager.experiment_name.startswith(
        "test_experiment_"
    )


def test_environment_creation(sample_data, config):
    """Test environment creation"""
    env = TradingEnvironment(
        df=sample_data,
        initial_balance=config["env"]["initial_balance"],
        trading_fee=config["env"]["trading_fee"],
        window_size=config["env"]["window_size"],
    )

    assert isinstance(env, TradingEnvironment)
    assert env.initial_balance == 10000.0
    assert env.trading_fee == 0.001
    assert env.window_size == 20

    # Test environment reset
    obs, info = env.reset()
    assert obs is not None
    assert info is not None
    assert "portfolio_value" in info


@pytest.mark.integration
def test_training_pipeline(mock_env):
    """Test the training pipeline with mock environment"""
    config = {
        "env": {
            "window_size": 10,
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
        },
        "model": {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.99},
        "training": {"num_episodes": 2, "total_timesteps": 1000},
        "experiment_name": "test_training",
        "mlflow_tracking_dir": "test_mlflow",
    }

    trainer = Trainer(config)
    trainer.env = mock_env  # Set environment before agent initialization
    trainer.agent = PPOAgent(  # Initialize agent after setting environment
        observation_space=mock_env.observation_space,
        action_space=mock_env.action_space,
        learning_rate=config["model"]["learning_rate"],
        gamma=config["model"]["gamma"],
    )
    trainer.train()  # Should not raise any errors


if __name__ == "__main__":
    pytest.main(["-v", __file__])
