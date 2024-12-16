"""Tests for Hyperparameter Optimization"""

import pytest
import ray
from ray import tune
import numpy as np
import pandas as pd
import os
import tempfile
import shutil

from training.hyperopt.hyperopt_tuner import MinimalTuner

@pytest.fixture
def test_data():
    """Create test data"""
    return pd.DataFrame({
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.abs(np.random.randn(100))
    })

@pytest.fixture
def checkpoint_dir():
    """Create temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_tuner_initialization(test_data):
    """Test tuner initialization"""
    tuner = MinimalTuner(test_data)
    assert hasattr(tuner, 'df')
    assert hasattr(tuner, 'objective')
    assert hasattr(tuner, 'run_optimization')

def test_objective_function(test_data):
    """Test objective function"""
    tuner = MinimalTuner(test_data)
    
    # Create test config
    test_config = {
        "learning_rate": 0.001,
        "hidden_size": 64,
        "gamma": 0.99,
        "epsilon": 0.2,
        "initial_balance": 10000,
        "transaction_fee": 0.001,
        "total_timesteps": 100
    }
    
    # Run objective
    tuner.objective(test_config)

def test_optimization_run(test_data):
    """Test full optimization run"""
    tuner = MinimalTuner(test_data)
    
    # Run optimization with minimal samples
    best_config = tuner.run_optimization(num_samples=2)
    
    assert isinstance(best_config, dict)
    assert "learning_rate" in best_config
    assert "hidden_size" in best_config
    assert "gamma" in best_config
    assert "epsilon" in best_config

def test_evaluate_config(test_data):
    """Test config evaluation"""
    tuner = MinimalTuner(test_data)
    
    test_config = {
        "learning_rate": 0.001,
        "hidden_size": 64,
        "gamma": 0.99,
        "epsilon": 0.2,
        "initial_balance": 10000,
        "transaction_fee": 0.001,
        "total_timesteps": 100
    }
    
    metrics = tuner.evaluate_config(test_config, episodes=2)
    assert isinstance(metrics, dict)
    assert "sharpe_ratio" in metrics
    assert "total_return" in metrics

@pytest.mark.integration
def test_mlflow_logging(test_data):
    """Test MLflow logging during optimization"""
    import mlflow
    
    experiment_name = "test_trading_optimization"
    tuner = MinimalTuner(test_data, mlflow_experiment=experiment_name)
    
    # Run optimization
    with mlflow.start_run() as run:
        best_config = tuner.run_optimization(num_samples=2)
        
        # Check MLflow logging
        run_info = mlflow.get_run(run.info.run_id)
        assert len(run_info.data.metrics) > 0

if __name__ == '__main__':
    pytest.main([__file__])