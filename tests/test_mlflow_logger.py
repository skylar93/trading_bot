import pytest
import mlflow
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from training.utils.mlflow_logger import MLflowLogger

class DummyModel(nn.Module):
    """Dummy PyTorch model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results"""
    return {
        'metrics': {
            'total_return': 15.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -10.5,
            'win_rate': 55.0
        },
        'trades': [
            {
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1),
                'entry_price': 100.0,
                'exit_price': 105.0,
                'size': 1.0,
                'pnl': 5.0,
                'return': 5.0
            },
            {
                'entry_time': datetime.now() + timedelta(hours=2),
                'exit_time': datetime.now() + timedelta(hours=3),
                'entry_price': 105.0,
                'exit_price': 103.0,
                'size': 1.0,
                'pnl': -2.0,
                'return': -1.9
            }
        ],
        'portfolio_values_with_time': [
            (datetime.now(), 10000),
            (datetime.now() + timedelta(hours=1), 10500),
            (datetime.now() + timedelta(hours=2), 10300)
        ]
    }

@pytest.fixture
def mlflow_logger(tmp_path):
    """Create MLflowLogger instance"""
    artifact_location = str(tmp_path / "mlruns")
    return MLflowLogger(
        experiment_name="test_experiment",
        artifact_location=artifact_location
    )

def test_logger_initialization(mlflow_logger):
    """Test MLflowLogger initialization"""
    assert mlflow_logger.experiment_name == "test_experiment"
    assert mlflow_logger.experiment is not None
    assert mlflow_logger.experiment.experiment_id is not None

def test_run_lifecycle(mlflow_logger):
    """Test starting and ending runs"""
    # Start run
    mlflow_logger.start_run(run_name="test_run")
    assert mlflow.active_run() is not None
    
    # End run
    mlflow_logger.end_run()
    assert mlflow.active_run() is None

def test_parameter_logging(mlflow_logger):
    """Test parameter logging"""
    params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'model_config': {'layers': [64, 32, 16]},
        'optimizer': 'Adam'
    }
    
    with mlflow.start_run():
        mlflow_logger.log_params(params)
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        
        # Check if parameters were logged
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                assert run.data.params[key] == str(value)
            else:
                assert run.data.params[key] == str(value)

def test_metric_logging(mlflow_logger):
    """Test metric logging"""
    metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'val_loss': 0.6
    }
    
    with mlflow.start_run():
        mlflow_logger.log_metrics(metrics)
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        
        # Check if metrics were logged
        for key, value in metrics.items():
            assert run.data.metrics[key] == value

def test_model_logging(mlflow_logger):
    """Test model logging"""
    model = DummyModel()
    
    with mlflow.start_run():
        mlflow_logger.log_model(model, "model")
        
        # Verify model was logged
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        
        assert isinstance(loaded_model, DummyModel)
        assert type(loaded_model.linear) == type(model.linear)

def test_backtest_results_logging(mlflow_logger, sample_backtest_results):
    """Test logging backtest results"""
    with mlflow.start_run():
        mlflow_logger.log_backtest_results(sample_backtest_results)
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        
        # Check metrics
        for key, value in sample_backtest_results['metrics'].items():
            assert run.data.metrics[key] == value
        
        # Check artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = [artifact.path for artifact in 
                    client.list_artifacts(mlflow.active_run().info.run_id)]
        
        assert "backtest_results/trades.csv" in artifacts
        assert "backtest_results/portfolio_values.csv" in artifacts
        assert "backtest_results/full_results.json" in artifacts

def test_best_run_retrieval(mlflow_logger):
    """Test getting best run"""
    # Create multiple runs with different metrics
    metrics_list = [
        {'metric1': 0.5, 'metric2': 0.8},
        {'metric1': 0.7, 'metric2': 0.6},
        {'metric1': 0.3, 'metric2': 0.9}
    ]
    
    for metrics in metrics_list:
        with mlflow.start_run():
            mlflow_logger.log_metrics(metrics)
    
    # Get best run
    best_run = mlflow_logger.get_best_run('metric1', mode='max')
    assert best_run is not None
    assert best_run['metrics']['metric1'] == 0.7  # Second run has highest metric1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])