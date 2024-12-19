import pytest
import mlflow
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.layer(x)

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

def test_model_logging(mlflow_test_context):
    """Test model logging"""
    model = DummyModel()
    
    with mlflow_test_context:
        mlflow_test_context.log_model(model, "test_model")
        
        # Try loading the model
        run_id = mlflow.active_run().info.run_id
        loaded_model = mlflow_test_context.load_model(run_id, "test_model")
        
        assert isinstance(loaded_model, DummyModel)
        assert type(loaded_model.layer) == type(model.layer)

def test_backtest_results_logging(mlflow_test_context, sample_data, sample_trades):
    """Test logging backtest results"""
    metrics = {
        'total_return': 15.5,
        'sharpe_ratio': 1.2,
        'max_drawdown': -10.5
    }
    
    with mlflow_test_context:
        mlflow_test_context.log_backtest_results(
            df_results=sample_data,
            metrics=metrics,
            trades=sample_trades
        )
        
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        
        # Check metrics
        assert run.data.metrics['total_return'] == 15.5
        assert run.data.metrics['sharpe_ratio'] == 1.2
        
        # Check artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = []
        for artifact in client.list_artifacts(mlflow.active_run().info.run_id):
            if artifact.is_dir:
                # List artifacts in subdirectory
                sub_artifacts = [a.path for a in client.list_artifacts(mlflow.active_run().info.run_id, artifact.path)]
                artifacts.extend(sub_artifacts)
            else:
                artifacts.append(artifact.path)
        
        assert 'backtest/trades.json' in artifacts
        assert 'backtest/results.json' in artifacts

if __name__ == "__main__":
    pytest.main([__file__])