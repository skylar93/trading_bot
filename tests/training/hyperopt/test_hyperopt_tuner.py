"""Tests for Hyperparameter Optimization"""

import pytest
import ray
from ray import tune
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import mlflow
from pathlib import Path
import time
from datetime import datetime

from training.hyperopt.hyperopt_tuner import MinimalTuner
from training.utils.mlflow_manager import MLflowManager
from envs.trading_env import TradingEnvironment
from agents.strategies.ppo_agent import PPOAgent

@pytest.fixture(scope="function")
def mlflow_test_context():
    """Create temporary MLflow test context"""
    # Set environment variable to disable strict metric checking
    os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'
    
    # Create temp directory for MLflow tracking
    temp_dir = tempfile.mkdtemp()
    tracking_uri = f"file://{temp_dir}"
    experiment_name = "test_experiment"
    
    # Clean up any existing MLflow files
    mlflow_dirs = [
        "./mlruns",
        "./mlflow_runs",
        os.path.join(os.getcwd(), "mlruns"),
        os.path.join(os.getcwd(), "mlflow_runs")
    ]
    for d in mlflow_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    # Delete experiment if it exists
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            # End any active runs
            for run in mlflow.search_runs([experiment.experiment_id]):
                if run.info.status == "RUNNING":
                    mlflow.end_run(run_id=run.info.run_id)
            # Delete experiment
            mlflow.delete_experiment(experiment.experiment_id)
            # Permanently delete experiment
            shutil.rmtree(os.path.join(temp_dir, experiment_name), ignore_errors=True)
    except:
        pass
    
    # Create new experiment
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=os.path.join(temp_dir, experiment_name)
    )
    
    # Set active experiment
    mlflow.set_experiment(experiment_name)
    
    # Wait for experiment to be fully created
    time.sleep(0.5)
    
    yield {
        'temp_dir': temp_dir,
        'tracking_uri': tracking_uri,
        'experiment_name': experiment_name,
        'experiment_id': experiment_id
    }
    
    # Cleanup
    if mlflow.active_run():
        mlflow.end_run()
        time.sleep(0.1)
    
    try:
        # End any active runs
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            for run in mlflow.search_runs([experiment.experiment_id]):
                if run.info.status == "RUNNING":
                    mlflow.end_run(run_id=run.info.run_id)
            # Delete experiment
            mlflow.delete_experiment(experiment.experiment_id)
            # Permanently delete experiment
            shutil.rmtree(os.path.join(temp_dir, experiment_name), ignore_errors=True)
    except:
        pass
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Clean up MLflow directories again
    for d in mlflow_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    # Reset environment variable
    os.environ.pop('TUNE_DISABLE_STRICT_METRIC_CHECKING', None)

@pytest.fixture(scope="module")
def ray_cluster():
    """Initialize Ray cluster for testing"""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture
def sample_data():
    """Create minimal sample market data with $ prefix columns"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')  # Reduced to 100 periods
    data = pd.DataFrame({
        '$open': np.random.randn(100).cumsum() + 100,
        '$high': np.random.randn(100).cumsum() + 102,
        '$low': np.random.randn(100).cumsum() + 98,
        '$close': np.random.randn(100).cumsum() + 100,
        '$volume': np.abs(np.random.randn(100) * 1000)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['$high'] = data[['$open', '$high', '$low', '$close']].max(axis=1)
    data['$low'] = data[['$open', '$high', '$low', '$close']].min(axis=1)
    
    return data

@pytest.fixture
def tuner(sample_data, mlflow_test_context):
    """Create tuner instance with MLflow integration"""
    # Ensure MLflow is properly initialized
    mlflow.set_tracking_uri(mlflow_test_context['tracking_uri'])
    
    tuner = MinimalTuner(
        sample_data,
        mlflow_experiment=mlflow_test_context['experiment_name']
    )
    yield tuner
    
    # Cleanup
    if mlflow.active_run():
        mlflow.end_run()
    tuner.cleanup()

def test_mlflow_logging(tuner, mlflow_test_context):
    """Test MLflow logging during hyperparameter optimization"""
    # Set up test data
    df = pd.DataFrame({
        '$open': np.random.randn(100).cumsum() + 100,
        '$high': np.random.randn(100).cumsum() + 102,
        '$low': np.random.randn(100).cumsum() + 98,
        '$close': np.random.randn(100).cumsum() + 100,
        '$volume': np.random.randint(1000, 10000, 100)
    })
    
    # Set up test search space with minimal values for quick testing
    search_space = {
        'learning_rate': tune.choice([1e-3]),  # Single value
        'hidden_size': tune.choice([32]),  # Single value
        'gamma': tune.choice([0.99]),  # Single value
        'gae_lambda': tune.choice([0.95]),  # Single value
        'clip_epsilon': tune.choice([0.2]),  # Single value
        'c1': tune.choice([1.0]),  # Single value
        'c2': tune.choice([0.01]),  # Single value
        'initial_balance': tune.choice([10000]),  # Single value
        'trading_fee': tune.choice([0.001]),  # Single value
        'total_timesteps': tune.choice([10])  # Minimal timesteps
    }
    
    # Run optimization with minimal samples and epochs
    best_trial = tuner.optimize(
        search_space=search_space,
        num_samples=1,  # Single trial
        max_concurrent=1,
        max_epochs=1  # Single epoch
    )
    
    # Wait for MLflow to finish writing
    time.sleep(1)
    
    # Get MLflow runs
    experiment = mlflow.get_experiment_by_name(mlflow_test_context['experiment_name'])
    assert experiment is not None, "MLflow experiment not found"
    
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) > 0, "No MLflow runs found"
    
    # Get the latest run that has metrics
    metric_runs = runs[runs['metrics.score'].notna()]
    assert len(metric_runs) > 0, "No runs with metrics found"
    latest_run = metric_runs.iloc[0]
    
    # Check that metrics were logged
    assert 'metrics.score' in latest_run.index, "Score metric not found"
    
    # Check that parameters were logged
    param_cols = [col for col in latest_run.index if col.startswith('params.')]
    assert len(param_cols) > 0, "No parameters were logged"
    
    # Check for specific parameters
    expected_params = [
        'params.learning_rate',
        'params.hidden_size',
        'params.gamma',
        'params.gae_lambda',
        'params.clip_epsilon',
        'params.c1',
        'params.c2',
        'params.initial_balance',
        'params.trading_fee',
        'params.total_timesteps'
    ]
    
    for param in expected_params:
        assert param in latest_run.index, f"{param} not found in logged parameters"
    
    # Check that best trial was returned
    assert best_trial is not None, "No best trial returned"
    assert hasattr(best_trial, 'metrics'), "Best trial has no metrics"
    assert best_trial.metrics is not None, "Best trial metrics is None"

if __name__ == "__main__":
    pytest.main(["-v", "test_hyperopt_tuner.py"])