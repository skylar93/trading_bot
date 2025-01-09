import pytest
import mlflow
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.utils.unified_mlflow_manager import MLflowManager


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
            "entry_time": current_time,
            "exit_time": current_time + timedelta(hours=1),
            "entry_price": 100,
            "exit_price": 105,
            "type": "buy",
            "pnl": 5,
        },
        {
            "entry_time": current_time + timedelta(hours=2),
            "exit_time": current_time + timedelta(hours=3),
            "entry_price": 105,
            "exit_price": 103,
            "type": "sell",
            "pnl": -2,
        },
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


def test_backtest_results_logging(mlflow_test_context, sample_data):
    """Test logging backtest results"""
    metrics = {
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.2,
        "total_return": 0.3,
    }
    
    trades_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5),
            "type": ["buy", "sell", "buy", "sell", "buy"],
            "price": [100, 110, 105, 115, 108],
            "quantity": [1, 1, 2, 2, 1],
        }
    )

    portfolio_values = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5),
            "value": [10000, 10100, 10050, 10200, 10150],
        }
    )

    results = {
        "metrics": metrics,
        "trades": trades_df,
        "portfolio_values": portfolio_values
    }

    with mlflow_test_context:
        mlflow_test_context.log_backtest_results(results)


if __name__ == "__main__":
    pytest.main([__file__])
