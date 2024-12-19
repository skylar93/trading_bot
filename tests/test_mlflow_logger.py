import pytest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
from mlflow.entities import RunStatus
from mlflow.utils.file_utils import path_to_local_file_uri
from training.utils.mlflow_manager_new import MLflowManager
import time

@pytest.fixture(scope="function")
def test_dir():
    """Create temporary directory for test artifacts"""
    test_dir = tempfile.mkdtemp()
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    try:
        # End any active runs before cleanup
        while mlflow.active_run():
            mlflow.end_run()
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    except Exception as e:
        print(f"Failed to remove test directory: {str(e)}")

@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow(test_dir):
    """Ensure no MLflow runs are active before and after each test"""
    # Set up MLflow tracking
    tracking_uri = path_to_local_file_uri(test_dir)
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    # Clean up before test
    while mlflow.active_run():
        mlflow.end_run()
    
    yield
    
    # Clean up after test
    while mlflow.active_run():
        mlflow.end_run()

@pytest.fixture(scope="function")
def mlflow_manager(test_dir):
    """Create MLflow manager with temporary directory"""
    # Create manager
    manager = MLflowManager("test_experiment", tracking_dir=test_dir)
    yield manager
    
    # Clean up
    manager.cleanup()

def test_logger_initialization(mlflow_manager, test_dir):
    """Test MLflow logger initialization"""
    assert mlflow_manager.experiment_name == "test_experiment"
    assert mlflow_manager.experiment is not None
    assert mlflow_manager.active_run is None
    
    # Verify experiment exists in MLflow
    experiment = mlflow.get_experiment_by_name("test_experiment")
    assert experiment is not None
    assert experiment.experiment_id == mlflow_manager.experiment_id

def test_run_lifecycle(mlflow_manager):
    """Test run lifecycle management"""
    # Start run
    run = mlflow_manager.start_run()
    assert mlflow_manager.active_run is not None
    assert run is not None
    
    # End run
    mlflow_manager.end_run()
    assert mlflow.active_run() is None
    assert mlflow_manager.active_run is None

def test_nested_runs(mlflow_manager):
    """Test nested run management"""
    # Start parent run
    parent_run = mlflow_manager.start_run(run_name="parent")
    assert mlflow_manager.active_run is not None
    
    # Start nested run
    nested_run = mlflow_manager.start_run(run_name="nested", nested=True)
    assert mlflow_manager.active_run is not None
    
    # End nested run first
    mlflow_manager.end_run()
    
    # End parent run
    mlflow_manager.end_run()
    
    # Verify no active runs
    assert mlflow.active_run() is None
    assert mlflow_manager.active_run is None

def test_cleanup_active_runs(mlflow_manager):
    """Test cleanup of active runs"""
    # Start multiple runs
    run1 = mlflow_manager.start_run(run_name="run1")
    mlflow_manager.end_run()
    
    run2 = mlflow_manager.start_run(run_name="run2")
    mlflow_manager.end_run()
    
    # Verify no active runs
    assert mlflow.active_run() is None
    assert mlflow_manager.active_run is None
    
    # Create new manager (should clean up existing runs)
    mlflow_manager.cleanup()

def test_parameter_logging(mlflow_manager):
    """Test parameter logging"""
    params = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'hidden_size': 128
    }
    
    # Start run and log parameters
    run = mlflow_manager.start_run()
    mlflow_manager.log_params(params)
    run_id = run.info.run_id
    
    # End run
    mlflow_manager.end_run()
    
    # Verify params
    run = mlflow.get_run(run_id)
    for key, value in params.items():
        assert run.data.params[key] == str(value)

def test_metric_logging(mlflow_manager):
    """Test metric logging"""
    metrics = {
        'loss': 0.5,
        'accuracy': 0.95,
        'validation_loss': 0.55
    }
    
    # Start run and log metrics
    run = mlflow_manager.start_run()
    mlflow_manager.log_metrics(metrics)
    mlflow_manager.log_metrics({'step_loss': 0.45}, step=1)
    run_id = run.info.run_id
    
    # End run
    mlflow_manager.end_run()
    
    # Verify metrics
    run = mlflow.get_run(run_id)
    for key, value in metrics.items():
        assert run.data.metrics[key] == value
    assert run.data.metrics['step_loss'] == 0.45

def test_model_logging(mlflow_manager):
    """Test model logging"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # Start run and log model
    run = mlflow_manager.start_run()
    mlflow_manager.log_model(model, "model")
    run_id = run.info.run_id
    
    # End run
    mlflow_manager.end_run()
    
    # Verify model was logged
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    assert any("model" in artifact.path for artifact in artifacts)

def test_backtest_results_logging(mlflow_manager):
    """Test backtest results logging"""
    results = {
        'metrics': {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.1
        },
        'trades': pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=3),
            'type': ['buy', 'sell', 'buy'],
            'price': [100, 110, 105],
            'quantity': [1.0, 1.0, 0.5]
        }),
        'portfolio_values': pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5),
            'value': [10000, 10100, 10050, 10200, 10150]
        })
    }
    
    # Start run and log results
    run = mlflow_manager.start_run()
    mlflow_manager.log_backtest_results(results)
    run_id = run.info.run_id
    
    # End run
    mlflow_manager.end_run()
    
    # Verify results
    run = mlflow.get_run(run_id)
    assert run.data.metrics['total_return'] == 0.15
    assert run.data.metrics['sharpe_ratio'] == 1.2
    assert run.data.metrics['max_drawdown'] == 0.1

def test_artifact_logging(mlflow_manager, test_dir):
    """Test artifact logging"""
    # Create test artifact
    artifact_path = os.path.join(test_dir, "test.txt")
    with open(artifact_path, "w") as f:
        f.write("Test artifact content")
    
    # Start run and log artifact
    run = mlflow_manager.start_run()
    mlflow_manager.log_artifact(artifact_path)
    run_id = run.info.run_id
    
    # End run
    mlflow_manager.end_run()
    
    # Verify artifact was logged
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    assert any("test.txt" in artifact.path for artifact in artifacts)

if __name__ == "__main__":
    pytest.main(["-v", __file__])