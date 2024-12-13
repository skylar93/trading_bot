import pytest
import mlflow
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.utils.mlflow_utils import MLflowManager

class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.layer(x)

@pytest.fixture
def mlflow_manager():
    """Create MLflowManager instance"""
    manager = MLflowManager(experiment_name="test_experiment")
    return manager

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)) * 100 + 1000,
        'portfolio_value': np.random.randn(len(dates)) * 1000 + 10000
    }, index=dates)
    return df

@pytest.fixture
def sample_trades():
    """Create sample trades for testing"""
    current_time = datetime.now()
    return [
        {
            'entry_time': current_time,
            'exit_time': current_time + timedelta(hours=1),
            'entry_price': 100,
            'exit_price': 105,
            'type': 'buy',
            'pnl': 5
        },
        {
            'entry_time': current_time + timedelta(hours=2),
            'exit_time': current_time + timedelta(hours=3),
            'entry_price': 105,
            'exit_price': 103,
            'type': 'sell',
            'pnl': -2
        }
    ]

def test_mlflow_initialization(mlflow_manager):
    """Test MLflowManager initialization"""
    assert mlflow_manager.experiment_name == "test_experiment"
    assert mlflow_manager.experiment_id is not None

def test_run_lifecycle(mlflow_manager):
    """Test MLflow run lifecycle"""
    # Start run
    mlflow_manager.start_run(run_name="test_run")
    assert mlflow.active_run() is not None
    
    # End run
    mlflow_manager.end_run()
    assert mlflow.active_run() is None

def test_parameter_logging(mlflow_manager):
    """Test parameter logging"""
    with mlflow.start_run():
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'hidden_size': 64
        }
        mlflow_manager.log_params(params)
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        for key, value in params.items():
            assert run.data.params[key] == str(value)

def test_metric_logging(mlflow_manager):
    """Test metric logging"""
    with mlflow.start_run():
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'sharpe_ratio': 1.2
        }
        mlflow_manager.log_metrics(metrics)
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        for key, value in metrics.items():
            assert run.data.metrics[key] == value

def test_model_logging(mlflow_manager):
    """Test model logging"""
    model = DummyModel()
    
    with mlflow.start_run():
        mlflow_manager.log_model(model, "test_model")
        
        # Try loading the model
        run_id = mlflow.active_run().info.run_id
        loaded_model = mlflow_manager.load_model(run_id, "test_model")
        
        assert isinstance(loaded_model, DummyModel)
        assert type(loaded_model.layer) == type(model.layer)

def test_backtest_results_logging(mlflow_manager, sample_data, sample_trades):
    """Test logging backtest results"""
    metrics = {
        'total_return': 15.5,
        'sharpe_ratio': 1.2,
        'max_drawdown': -10.5
    }
    
    with mlflow.start_run():
        mlflow_manager.log_backtest_results(
            df_results=sample_data,
            metrics=metrics,
            trades=sample_trades
        )
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        
        # Check metrics
        assert run.data.metrics['backtest_total_return'] == 15.5
        assert run.data.metrics['backtest_sharpe_ratio'] == 1.2
        
        # Check artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = [artifact.path for artifact in 
                    client.list_artifacts(mlflow.active_run().info.run_id)]
        
        assert 'portfolio_value.html' in artifacts
        assert 'trades.html' in artifacts

def test_get_best_run(mlflow_manager):
    """Test getting best run"""
    # Create multiple runs with different metrics
    metrics_list = [
        {'metric1': 0.5, 'metric2': 0.8},
        {'metric1': 0.7, 'metric2': 0.6},
        {'metric1': 0.3, 'metric2': 0.9}
    ]
    
    for metrics in metrics_list:
        with mlflow.start_run():
            mlflow_manager.log_metrics(metrics)
    
    # Get best run (maximum metric1)
    best_run = mlflow_manager.get_best_run('metric1', mode='max')
    assert best_run is not None
    assert best_run['metrics']['metric1'] == 0.7  # Second run had highest metric1

if __name__ == "__main__":
    pytest.main([__file__])