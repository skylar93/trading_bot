"""Tests for MLflow experiment tracking"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from training.utils.unified_mlflow_manager import MLflowManager


@pytest.fixture
def sample_data():
    """Create sample OHLCV data with proper column names"""
    return pd.DataFrame(
        {
            "$open": [100, 101, 102],
            "$high": [102, 103, 104],
            "$low": [99, 100, 101],
            "$close": [101, 102, 103],
            "$volume": [1000, 1100, 1200],
        }
    )


def test_mlflow_manager_init(mlflow_test_context):
    """Test MLflow manager initialization"""
    assert mlflow_test_context.experiment_name.startswith("test_experiment_")
    assert mlflow_test_context.experiment_id is not None
    assert mlflow.active_run() is None


def test_run_lifecycle(mlflow_test_context):
    """Test MLflow run lifecycle"""
    # Start run
    run = mlflow_test_context.start_run(run_name="test_run")
    assert run is not None
    assert mlflow.active_run() is not None

    # Log metrics
    metrics = {"metric1": 1.0, "metric2": 2.0}
    mlflow_test_context.log_metrics(metrics)

    # Log parameters
    params = {"param1": "value1", "param2": 2}
    mlflow_test_context.log_params(params)

    # End run
    mlflow_test_context.end_run()
    assert mlflow.active_run() is None


def test_nested_runs(mlflow_test_context):
    """Test nested MLflow runs"""
    # Start parent run
    parent_run = mlflow_test_context.start_run(run_name="parent")
    assert parent_run is not None

    # Start nested run
    nested_run = mlflow_test_context.start_run(run_name="child", nested=True)
    assert nested_run is not None

    # End nested run
    mlflow_test_context.end_run()

    # End parent run
    mlflow_test_context.end_run()
    assert mlflow.active_run() is None


def test_model_logging(mlflow_test_context):
    """Test model logging"""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    # Create dummy model
    model = DummyModel()

    # Start run and log model
    with mlflow_test_context:
        mlflow_test_context.log_model(model, "model")

        # Verify model artifact exists
        artifact_uri = mlflow_test_context.get_artifact_uri("model")
        assert os.path.exists(artifact_uri.replace("file://", ""))


def test_dataframe_logging(mlflow_test_context):
    """Test DataFrame logging"""
    # Create dummy DataFrame
    df = pd.DataFrame(
        {"col1": np.random.randn(10), "col2": np.random.randn(10)}
    )

    # Start run and log DataFrame
    with mlflow_test_context:
        mlflow_test_context.log_dataframe(df, "data", "test.json")

        # Verify DataFrame artifact exists
        artifact_uri = mlflow_test_context.get_artifact_uri("data/test.json")
        assert os.path.exists(artifact_uri.replace("file://", ""))


def test_cleanup(mlflow_test_context):
    """Test MLflow cleanup"""
    # Get tracking directory
    tracking_dir = mlflow_test_context.tracking_dir

    # Start run and log something
    with mlflow_test_context:
        mlflow_test_context.log_metrics({"metric": 1.0})

    # Clean up
    mlflow_test_context.cleanup()

    # Verify cleanup
    assert not os.path.exists(tracking_dir)


def test_artifact_format(mlflow_test_context, sample_data):
    """Test artifact format and storage"""
    with mlflow_test_context:
        # Log DataFrame
        mlflow_test_context.log_dataframe(sample_data, "data", "test.json")

        # Verify artifact exists
        artifact_uri = mlflow_test_context.get_artifact_uri("data/test.json")
        assert os.path.exists(artifact_uri.replace("file://", ""))


def test_error_handling(mlflow_test_context):
    """Test error handling in MLflow operations"""
    # Test logging without active run
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow_test_context.log_metrics({"metric": 1.0})

    # Test nested run without parent
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow_test_context.start_run(nested=True)


def test_backtest_results(mlflow_test_context, sample_data):
    """Test logging backtest results"""
    with mlflow_test_context:
        # Log metrics
        metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.2,
            "total_return": 0.3,
        }
        mlflow_test_context.log_metrics(metrics)

        # Log trades DataFrame
        trades_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=5),
                "type": ["buy", "sell", "buy", "sell", "buy"],
                "price": [100, 110, 105, 115, 108],
                "quantity": [1, 1, 2, 2, 1],
            }
        )
        mlflow_test_context.log_dataframe(trades_df, "backtest", "trades.json")

        # Verify artifacts
        trades_uri = mlflow_test_context.get_artifact_uri(
            "backtest/trades.json"
        )
        assert os.path.exists(trades_uri.replace("file://", ""))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
