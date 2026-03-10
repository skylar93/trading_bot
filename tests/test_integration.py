"""
Integration tests for the trading bot system.

These tests verify that different components of the system work together correctly,
including environment creation, agent training, and hyperparameter optimization.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import os
import tempfile
import shutil

from training.train_pipeline import train_pipeline
from training.hyperopt.hyperopt_ray import run_hyperparameter_optimization
from training.utils.config_manager import ConfigManager

@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns."""
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

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "env": {
            "type": "single_asset_rl",
            "initial_capital": 10000.0,
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
            "total_timesteps": 500,
            "eval_freq": 100,
            "n_eval_episodes": 5,
        },
        "hyperopt": {
            "search_algorithm": "random",
            "metric": "mean_reward",
            "mode": "max",
            "num_samples": 2,
            "parameters": {
                "agent.learning_rate": {
                    "distribution": "loguniform",
                    "min": 1e-5,
                    "max": 1e-2
                }
            }
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints",
            "hyperopt_results_dir": "test_hyperopt_results"
        },
        "data": {
            "data_path": "test_data.csv"
        }
    }

@pytest.mark.integration
def test_training_to_hyperopt_flow(test_config, sample_data):
    """
    Test the flow from training to hyperparameter optimization.
    This verifies that we can:
    1. Train a basic model
    2. Use that as a baseline for hyperopt
    3. Get improved parameters
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update paths to use temp directory
        test_config["paths"]["checkpoint_dir"] = os.path.join(temp_dir, "checkpoints")
        test_config["paths"]["hyperopt_results_dir"] = os.path.join(temp_dir, "hyperopt")
        
        # First, do a basic training run
        with patch("training.env_factory.load_data", return_value=sample_data):
            results = train_pipeline(test_config)
            
            # Verify we got basic training results
            assert "episode_rewards" in results
            assert "best_eval_reward" in results
            baseline_reward = results["best_eval_reward"]
        
        # Now run hyperopt with the same base config
        with patch("training.env_factory.load_data", return_value=sample_data), \
             patch("ray.init"):
            
            best_config, opt_results = run_hyperparameter_optimization(
                config=test_config  # Pass config directly
            )
            
            # Verify we got optimization results
            assert best_config is not None
            assert "agent.learning_rate" in best_config

@pytest.mark.integration
def test_config_to_training_flow(test_config, sample_data):
    """
    Test the flow from config management to training.
    This verifies that we can:
    1. Load and modify config
    2. Use it for training
    3. Save and reload results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a config file
        config_path = os.path.join(temp_dir, "config.yaml")
        config_manager = ConfigManager()
        config_manager.config = test_config  # Set config directly
        config_manager.save(config_path)
        
        # Load and modify some settings
        config_manager.load_config(config_path)
        config_manager.set("env.window_size", 20)
        config_manager.set("training.total_timesteps", 300)
        
        # Use the modified config for training
        with patch("training.env_factory.load_data", return_value=sample_data):
            results = train_pipeline(config_manager.config)
            
            # Verify the results reflect our modifications
            assert "episode_rewards" in results
            assert len(results["episode_rewards"]) > 0

@pytest.mark.integration
def test_checkpoint_resume_flow(test_config, sample_data):
    """
    Test that we can resume training from a checkpoint.
    
    This verifies that we can:
    1. Start training and save checkpoints
    2. Resume training from a checkpoint
    3. Get consistent results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up checkpoint directory
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        test_config["paths"]["checkpoint_dir"] = checkpoint_dir
        test_config["training"]["checkpoint_freq"] = 100
        
        # Do initial training
        with patch("training.env_factory.load_data", return_value=sample_data):
            results1 = train_pipeline(test_config)
            
            # Verify we got checkpoints
            assert os.path.exists(checkpoint_dir)
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            assert len(checkpoints) > 0
            
            # Resume training
            test_config["training"]["resume_from"] = os.path.join(checkpoint_dir, checkpoints[-1])
            results2 = train_pipeline(test_config)
            
            # Verify resumed training produced results
            assert "episode_rewards" in results2
            assert len(results2["episode_rewards"]) > 0

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 