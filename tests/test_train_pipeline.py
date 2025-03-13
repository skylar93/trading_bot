# tests/test_train_pipeline.py
"""
Tests for the new unified training pipeline and single-asset RL environment.

This file replaces the older test_trainer.py, removing references to the
deprecated Trainer/TrainingPipeline classes, and uses the new train_pipeline
function from `training/train_pipeline.py`.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import gymnasium as gym
from gymnasium import spaces

# Import your new pipeline and environment:
from training.train_pipeline import train_pipeline
from envs.single_asset_rl_env import SingleAssetRLTradingEnv

@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns."""
    # 200 timesteps of fake data, just enough for quick tests
    dates = pd.date_range(start="2025-01-01", periods=200, freq="h")
    df = pd.DataFrame(
        {
            "$open": np.random.normal(100, 1, 200).cumsum(),
            "$high": np.random.normal(100, 1, 200).cumsum(),
            "$low": np.random.normal(100, 1, 200).cumsum(),
            "$close": np.random.normal(100, 1, 200).cumsum(),
            "$volume": np.abs(np.random.randn(200) * 100),
        },
        index=dates,
    )
    return df


def test_environment_creation(sample_data):
    """
    Basic test to ensure SingleAssetRLTradingEnv can be created and reset.
    """
    env = SingleAssetRLTradingEnv(
        data=sample_data,
        initial_capital=5000.0,
        trading_fee=0.002,
        window_size=10,
        max_position_size=0.5,
    )

    assert isinstance(env, SingleAssetRLTradingEnv)
    assert env.initial_capital == 5000.0
    assert env.trading_fee == 0.002
    assert env.window_size == 10
    assert env.max_position_size == 0.5

    obs, info = env.reset()
    assert obs.shape == (10, 5), "Observation shape should be (window_size, 5)"
    assert isinstance(info, dict)
    assert "portfolio_value" in info


def test_train_pipeline_single_agent(sample_data):
    """
    Test the new train_pipeline function in a single-agent setup.
    We'll patch load_data so we don't need a real CSV path.
    """
    # Minimal single-agent config
    config = {
        "env": {
            "type": "single_asset_rl",  # key to pick SingleAssetRLTradingEnv
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
            "window_size": 10,
            "max_position_size": 1.0,
        },
        "agent": {
            "name": "PPO",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "batch_size": 32,
            "n_epochs": 2,
        },
        "training": {
            "total_timesteps": 500,  # keep small for quick test
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints",  # a test location
        },
        # We'll rely on data.data_path, but we'll patch load_data anyway
        "data": {
            "data_path": "does_not_exist.csv",
        },
    }

    # We patch the `training.env_factory.load_data` function so it returns our `sample_data`.
    with patch("training.env_factory.load_data", return_value=sample_data):
        results = train_pipeline(config)

    # We expect certain keys in `results`. For single-agent, we get:
    #   "episode_rewards", "episode_lengths", "best_eval_reward", ...
    # Make sure the pipeline returned something sensible.
    assert "episode_rewards" in results, "train_pipeline should return episode_rewards"
    assert len(results["episode_rewards"]) > 0, "We should have at least 1 finished episode"
    assert "best_eval_reward" in results, "Single-agent pipeline should track best_eval_reward"


@pytest.mark.integration
def test_train_pipeline_multi_agent(sample_data):
    """
    Test the new train_pipeline function in a multi-agent setup.
    This test creates actual agents and runs a small number of training steps.
    """
    # Minimal multi-agent config
    config = {
        "env": {
            "type": "multi_agent_rl",
            "initial_balance": 10000.0,
            "trading_fee": 0.001, 
            "window_size": 20,  # Explicitly set window_size to avoid confusion
            "multi_agent_configs": [
                {
                    "id": "agent1",
                    "agent_type": "ppo",  # Changed from type to agent_type
                    "strategy": "momentum",  # Added explicit strategy
                    "initial_balance": 5000.0,
                    "hyperparameters": {
                        "learning_rate": 3e-4,
                        "gamma": 0.95,
                    }
                },
                {
                    "id": "agent2",
                    "agent_type": "ppo",  # Changed from type to agent_type
                    "strategy": "meanreversion",  # Added explicit strategy
                    "initial_balance": 5000.0,
                    "hyperparameters": {
                        "learning_rate": 2e-4,
                        "gamma": 0.9,
                    }
                },
            ],
            "shared_capital": False,  # Make sure each agent has independent capital
        },
        "training": {
            "total_timesteps": 100,  # Keep very small for quick testing
            "eval_interval": 50,
            "checkpoint_interval": 50,
            "log_interval": 1
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints",
        },
        "data": {
            "data_path": "does_not_exist.csv",
        },
        "shared_experience": {
            "enabled": True,
            "buffer_size": 1000
        }
    }

    # We patch the `training.env_factory.load_data` function so it returns our `sample_data`.
    with patch("training.env_factory.load_data", return_value=sample_data):
        results = train_pipeline(config)

    # We expect certain keys in `results`. For multi-agent, we get:
    #   "episode_rewards", "episode_lengths", "best_eval_rewards", ...
    # Make sure the pipeline returned something sensible.
    assert "episode_rewards" in results, "train_pipeline should return episode_rewards"
    assert "agent1" in results["episode_rewards"], "Results should contain data for agent1"
    assert "agent2" in results["episode_rewards"], "Results should contain data for agent2"
    assert len(results["episode_rewards"]["agent1"]) > 0, "We should have at least 1 finished episode for agent1"
    assert len(results["episode_rewards"]["agent2"]) > 0, "We should have at least 1 finished episode for agent2"
    assert "best_eval_rewards" in results, "Multi-agent pipeline should track best_eval_rewards"
    assert "final_model_paths" in results, "Should have final model paths"
    assert "best_model_paths" in results, "Should have best model paths"


if __name__ == "__main__":
    # Run this file directly:
    pytest.main(["-v", __file__]) 