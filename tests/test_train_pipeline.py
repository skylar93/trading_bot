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
    Test multi-agent scenario with proper observation and action spaces.
    Each agent should have its own PPO instance with proper spaces.
    """
    # Create multi-agent config with proper spaces
    config = {
        "env": {
            "type": "multi_agent_rl",
            "initial_balance": 10000.0,
            "trading_fee": 0.001,
            "window_size": 20,  # Increased from 10 to 20 to match momentum_window
            "max_position_size": 1.0,
            # Environment-specific multi-agent configs
            "multi_agent_configs": [
                {
                    "id": "agent1",
                    "type": "momentum",
                    "strategy": "momentum",
                    "initial_capital_percentage": 0.5,
                    "priority": 1,
                    "hyperparameters": {
                        "learning_rate": 1e-4,
                        "gamma": 0.95,
                        "clip_epsilon": 0.2,
                        "batch_size": 32,
                        "n_epochs": 2,
                        "normalize_observations": True,
                        "momentum_window": 20  # Explicitly set momentum_window to match env window_size
                    }
                },
                {
                    "id": "agent2",
                    "type": "meanreversion",
                    "strategy": "meanreversion",
                    "initial_capital_percentage": 0.5,
                    "priority": 2,
                    "hyperparameters": {
                        "learning_rate": 2e-4,
                        "gamma": 0.90,
                        "clip_epsilon": 0.2,
                        "batch_size": 32,
                        "n_epochs": 2,
                        "normalize_observations": True
                    }
                }
            ]
        },
        "training": {
            "total_timesteps": 500,
            "checkpoint_interval": 100,
            "eval_interval": 100,
            "n_eval_episodes": 5
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints_multi",
        },
        "data": {
            "data_path": "multi_agent_data.csv",
        },
        # Flag to indicate multi-agent training
        "multi_agent": True
    }

    # Patch load_data to return sample_data
    with patch("training.env_factory.load_data", return_value=sample_data):
        try:
            results = train_pipeline(config)
            
            # For multi-agent, we expect "best_eval_rewards" dict
            assert "best_eval_rewards" in results, "Multi-agent results should have best_eval_rewards"
            assert isinstance(results["best_eval_rewards"], dict), "best_eval_rewards should be a dict"
            assert "episode_rewards" in results, "Should store a dict of agent_id -> list of rewards"
            assert isinstance(results["episode_rewards"], dict), "episode_rewards should be a dict"
            
            # Check that we have results for both agents
            assert "agent1" in results["best_eval_rewards"]
            assert "agent2" in results["best_eval_rewards"]
            assert "agent1" in results["episode_rewards"]
            assert "agent2" in results["episode_rewards"]
            
            # Verify reward values are reasonable
            for agent_id in ["agent1", "agent2"]:
                assert results["best_eval_rewards"][agent_id] > float('-inf'), f"{agent_id} has invalid reward"
                assert len(results["episode_rewards"][agent_id]) > 0, f"{agent_id} has no episode rewards"
            
        except ImportError:
            pytest.skip("MultiAgentTradingEnv not implemented; skipping multi-agent pipeline test.")


if __name__ == "__main__":
    # Run this file directly:
    pytest.main(["-v", __file__]) 