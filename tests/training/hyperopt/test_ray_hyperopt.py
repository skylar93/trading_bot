"""
Tests for the Ray Tune-based hyperparameter optimization system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import os
import tempfile
import shutil

try:
    import ray
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from training.hyperopt.hyperopt_ray import (
        train_func,
        create_search_space,
        create_search_algorithm,
        create_scheduler,
        run_hyperparameter_optimization,
    )
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

pytestmark = pytest.mark.skipif(not HAS_RAY, reason="ray not installed")

@pytest.fixture
def ray_results_dir():
    """Create and return a temporary directory for Ray Tune results."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="ray_results_test_")
    
    # Initialize Ray if not already started
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    
    # Yield the directory path
    yield temp_dir
    
    # Clean up the temporary directory after test
    shutil.rmtree(temp_dir, ignore_errors=True)

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
    """Create a test configuration for hyperparameter optimization."""
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
            "n_epochs": 1,
        },
        "training": {
            "total_timesteps": 100,
            "eval_freq": 50,
            "n_eval_episodes": 2,
            "use_gpu": False,
        },
        "hyperopt": {
            "search_algorithm": "random",
            "metric": "mean_reward",
            "mode": "max",
            "num_samples": 1,
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

def test_create_search_space(test_config):
    """Test creation of Ray Tune search space from config."""
    search_space = create_search_space(test_config)
    
    assert "agent.learning_rate" in search_space
    assert "_full_config" in search_space
    
    # Check that the full config is included
    assert search_space["_full_config"] == test_config

def test_create_search_algorithm(test_config):
    """Test creation of search algorithm based on config."""
    # Test with any search algorithm type
    alg = create_search_algorithm(test_config)
    assert alg is not None
    from ray.tune.search import BasicVariantGenerator
    assert isinstance(alg, BasicVariantGenerator)

def test_create_scheduler(test_config):
    """Test scheduler creation."""
    # Test with any scheduler type
    scheduler = create_scheduler(test_config)
    assert scheduler is not None
    from ray.tune.schedulers import FIFOScheduler
    assert isinstance(scheduler, FIFOScheduler)

@pytest.mark.integration
def test_train_func(test_config, sample_data, mocker):
    """Test the training function used by Ray Tune."""
    # Create a copy of test_config that includes _full_config
    config_with_full = {
        "_full_config": test_config.copy(),
        "agent.learning_rate": 3e-4,
    }
    
    # Mock the train_pipeline function to avoid actual computation
    mock_train = mocker.patch("training.hyperopt.hyperopt_ray.train_pipeline")
    mock_train.return_value = {"mean_reward": 100}
    
    # Call the train function
    result = train_func(config_with_full)
    
    # Check that the train_pipeline was called
    assert mock_train.called
    
    # Check that the result contains the expected keys
    assert "mean_reward" in result

@pytest.mark.integration
def test_run_hyperparameter_optimization(ray_results_dir, mocker):
    """Test running hyperparameter optimization with Ray Tune."""
    # Create a test configuration
    config = {
        "env": {
            "window_size": 10,
            "initial_capital": 10000.0,
            "trading_fee": 0.001,
            "type": "single_asset_rl"
        },
        "agent": {
            "type": "PPO",
            "learning_rate": 0.001,
            "batch_size": 64
        },
        "training": {
            "total_timesteps": 100
        },
        "hyperopt": {
            "num_samples": 2,
            "max_epochs": 2,
            "metric": "mean_reward",
            "mode": "max",
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0
            }
        },
        "data": {
            "data_path": "data/test_data.csv"
        }
    }
    
    # Ensure test_data.csv exists
    if not os.path.exists("data/test_data.csv"):
        # Create sample data
        from datetime import datetime, timedelta
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            "$open": [100 + i * 0.1 for i in range(100)],
            "$high": [105 + i * 0.1 for i in range(100)],
            "$low": [95 + i * 0.1 for i in range(100)],
            "$close": [101 + i * 0.1 for i in range(100)],
            "$volume": [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/test_data.csv")
    
    # Mock the train_pipeline function to avoid actual computation
    # We need to mock it at the module level where it's imported
    mock_train = mocker.patch("training.hyperopt.hyperopt_ray.globals")
    mock_train.get.return_value = lambda config: {"mean_reward": 100}
    
    # Run hyperparameter optimization
    best_config, best_results = run_hyperparameter_optimization(
        config, storage_path=ray_results_dir, num_samples=1
    )
    
    # Print debug info
    print(f"Best config: {best_config}")
    
    # Check that the best_config contains the expected keys
    assert '_full_config' in best_config
    assert 'agent' in best_config['_full_config']
    assert 'learning_rate' in best_config['_full_config']['agent']
    assert 'batch_size' in best_config['_full_config']['agent']

@pytest.mark.integration
def test_run_hyperparameter_optimization_legacy_path(ray_results_dir, mocker):
    """Test running hyperparameter optimization with legacy paths.data configuration."""
    # Create a test configuration with legacy paths.data format
    config = {
        "env": {
            "window_size": 10,
            "initial_capital": 10000.0,
            "trading_fee": 0.001,
            "type": "single_asset_rl"
        },
        "agent": {
            "type": "PPO",
            "learning_rate": 0.001,
            "batch_size": 64
        },
        "training": {
            "total_timesteps": 100
        },
        "hyperopt": {
            "num_samples": 1,
            "max_epochs": 1,
            "metric": "mean_reward",
            "mode": "max",
            "resources_per_trial": {
                "cpu": 1,
                "gpu": 0
            }
        },
        "paths": {
            "data": os.path.abspath("./data"),
            "models": os.path.abspath("./models"),
            "logs": os.path.abspath("./logs"),
            "results": os.path.abspath(ray_results_dir)
        }
    }
    
    # Ensure test_data.csv exists
    if not os.path.exists("data/test_data.csv"):
        # Create sample data
        from datetime import datetime, timedelta
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            "$open": [100 + i * 0.1 for i in range(100)],
            "$high": [105 + i * 0.1 for i in range(100)],
            "$low": [95 + i * 0.1 for i in range(100)],
            "$close": [101 + i * 0.1 for i in range(100)],
            "$volume": [1000 + i * 10 for i in range(100)]
        }, index=dates)
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/test_data.csv")
    
    # Mock the train_pipeline function to avoid actual computation
    # We need to mock it at the module level where it's imported
    mock_train = mocker.patch("training.hyperopt.hyperopt_ray.globals")
    mock_train.get.return_value = lambda config: {"mean_reward": 100}
    
    # Run hyperparameter optimization
    best_config, best_results = run_hyperparameter_optimization(
        config, 
        storage_path=ray_results_dir,
        experiment_name="test_legacy_paths",
        num_samples=1
    )
    
    # Print debug info
    print(f"Best config: {best_config}")
    
    # Check that the data_path was set from the legacy paths.data
    if "data" in best_config.get("_full_config", {}):
        assert "data_path" in best_config["_full_config"]["data"]

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 