"""Tests for Hyperparameter Optimization"""

import pytest
import ray
from ray import tune
import numpy as np
import os
import tempfile
import shutil

from training.hyperopt.hyperopt_tuner import (
    OptimizationConfig,
    HyperOptTuner
)

@pytest.fixture
def test_config():
    """Create test optimization config"""
    param_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_size": tune.choice([64, 128, 256])
    }
    
    return OptimizationConfig(
        param_space=param_space,
        num_samples=2,
        num_epochs=2,
        resources_per_trial={"cpu": 1, "gpu": 0}
    )

@pytest.fixture
def checkpoint_dir():
    """Create temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_optimization_config():
    """Test optimization config initialization"""
    param_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2)
    }
    
    config = OptimizationConfig(param_space=param_space)
    assert config.num_samples == 10
    assert config.resources_per_trial["cpu"] == 1
    assert config.scheduler_config is not None

def test_tuner_initialization(test_config):
    """Test tuner initialization"""
    tuner = HyperOptTuner(test_config)
    assert tuner.search_alg is not None
    assert tuner.scheduler is not None

def test_objective_function(test_config):
    """Test objective function"""
    tuner = HyperOptTuner(test_config)
    
    # Create test config
    test_trial_config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "hidden_size": 128
    }
    
    # Run objective
    result = tuner.objective(test_trial_config)
    
    assert "mean_reward" in result
    assert "std_reward" in result
    assert "max_reward" in result
    assert "min_reward" in result
    assert "final_reward" in result

def test_optimization_run(test_config, checkpoint_dir):
    """Test full optimization run"""
    tuner = HyperOptTuner(test_config)
    
    # Run optimization
    best_trials = tuner.run_optimization(checkpoint_dir)
    
    assert len(best_trials) > 0
    assert os.path.exists(os.path.join(checkpoint_dir, "best_config.json"))

@pytest.mark.integration
def test_mlflow_logging(test_config, checkpoint_dir):
    """Test MLflow logging during optimization"""
    import mlflow
    
    tuner = HyperOptTuner(test_config)
    
    # Run optimization
    with mlflow.start_run() as run:
        tuner.run_optimization(checkpoint_dir)
        
        # Check MLflow logging
        run_info = mlflow.get_run(run.info.run_id)
        assert len(run_info.data.params) > 0
        assert len(run_info.data.metrics) > 0

if __name__ == '__main__':
    pytest.main([__file__])