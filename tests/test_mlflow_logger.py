"""Tests for MLflow experiment tracking"""

import pytest
import pandas as pd
import numpy as np
import mlflow
from mlflow.entities import RunStatus
from mlflow.utils.file_utils import path_to_local_file_uri
import torch
import torch.nn as nn
from pathlib import Path
import shutil
import tempfile
import os
import time
import logging
from training.utils.unified_mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

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
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Clean up before test
    while mlflow.active_run():
        mlflow.end_run()

    yield

    # Clean up after test
    while mlflow.active_run():
        mlflow.end_run()

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

@pytest.fixture(scope="function")
def mlflow_manager(test_dir):
    """Create MLflow manager with temporary directory"""
    manager = MLflowManager(
        "test_experiment",
        tracking_dir=test_dir,
        append_timestamp=True,
        allow_deleted_experiment_cleanup=True
    )
    yield manager
    manager.cleanup()

def test_logger_initialization(mlflow_manager):
    """Test MLflow logger initialization"""
    assert mlflow_manager.experiment_name.startswith("test_experiment")
    assert mlflow_manager.experiment_id is not None
    assert mlflow.active_run() is None

def test_run_lifecycle(mlflow_manager):
    """Test run lifecycle management"""
    # Start run
    run = mlflow_manager.start_run(run_name="test_run")
    assert run is not None
    assert mlflow.active_run() is not None

    # Log metrics
    metrics = {"metric1": 1.0, "metric2": 2.0}
    mlflow_manager.log_metrics(metrics)

    # Log parameters
    params = {"param1": "value1", "param2": 2}
    mlflow_manager.log_params(params)

    # End run
    mlflow_manager.end_run()
    assert mlflow.active_run() is None

def test_nested_runs(mlflow_manager):
    """Test nested run management"""
    # Start parent run
    parent_run = mlflow_manager.start_run(run_name="parent")
    assert parent_run is not None

    # Start nested run
    nested_run = mlflow_manager.start_run(run_name="child", nested=True)
    assert nested_run is not None

    # End nested run
    mlflow_manager.end_run()

    # End parent run
    mlflow_manager.end_run()
    assert mlflow.active_run() is None

def test_model_logging(mlflow_manager):
    """Test model logging"""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()

    with mlflow_manager:
        mlflow_manager.log_model(model, "model")
        run_id = mlflow_manager.active_run.info.run_id
        loaded_model = mlflow_manager.load_model(run_id, "model")
        assert isinstance(loaded_model, nn.Module)

def test_dataframe_logging(mlflow_manager, sample_data):
    """Test DataFrame logging"""
    with mlflow_manager:
        mlflow_manager.log_dataframe(sample_data, "data", "test.parquet")

def test_cleanup(mlflow_manager):
    """Test MLflow cleanup"""
    # Get tracking directory
    tracking_dir = mlflow_manager.tracking_dir

    # Start run and log something
    with mlflow_manager:
        mlflow_manager.log_metrics({"metric": 1.0})

    # Clean up
    mlflow_manager.cleanup()

    # Verify cleanup
    assert not os.path.exists(tracking_dir)

def test_error_handling(mlflow_manager):
    """Test error handling in MLflow operations"""
    # Test logging without active run
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow_manager.log_metrics({"metric": 1.0})

    # Test nested run without parent
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow_manager.start_run(nested=True)

def test_backtest_results(mlflow_manager, sample_data):
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

    with mlflow_manager:
        results = {
            "metrics": metrics,
            "trades": trades_df,
            "portfolio_values": portfolio_values
        }
        mlflow_manager.log_backtest_results(results)

def test_context_manager(mlflow_manager):
    """Test context manager functionality"""
    with mlflow_manager as manager:
        assert manager.active_run is not None
        manager.log_metrics({"test": 1.0})
    assert manager.active_run is None

def test_experiment_deletion_handling(test_dir):
    """Test deleted experiment handling"""
    # Create first manager with a unique name
    manager1 = MLflowManager(
        "test_experiment_1",  # Different base name
        tracking_dir=test_dir,
        allow_deleted_experiment_cleanup=True,
        append_timestamp=True
    )
    experiment_id = manager1.experiment_id
    
    # Delete experiment
    mlflow.delete_experiment(experiment_id)
    time.sleep(0.5)  # Wait for deletion to complete
    
    # Verify experiment is marked as deleted
    deleted_exp = mlflow.get_experiment(experiment_id)
    assert deleted_exp.lifecycle_stage == "deleted"
    
    # Create new manager with a different name
    manager2 = MLflowManager(
        "test_experiment_2",  # Different name
        tracking_dir=test_dir,
        allow_deleted_experiment_cleanup=True,
        append_timestamp=True
    )
    
    # Verify new experiment was created
    assert manager2.experiment_id != experiment_id  # Different ID
    new_exp = mlflow.get_experiment(manager2.experiment_id)
    assert new_exp.lifecycle_stage == "active"  # New experiment is active
    assert new_exp.name != deleted_exp.name  # Different name

    # Cleanup
    manager1.cleanup()
    manager2.cleanup()

def test_get_best_run(mlflow_manager):
    """Test getting best run based on metric"""
    # Create multiple runs with different metrics
    metrics = [
        {"accuracy": 0.8, "loss": 0.3},
        {"accuracy": 0.9, "loss": 0.2},
        {"accuracy": 0.7, "loss": 0.4}
    ]
    
    run_ids = []
    for m in metrics:
        with mlflow_manager:
            mlflow_manager.log_metrics(m)
            run_ids.append(mlflow_manager.active_run.info.run_id)
    
    # Test max mode
    best_run = mlflow_manager.get_best_run("accuracy", mode="max")
    assert best_run.info.run_id == run_ids[1]  # Second run has highest accuracy
    
    # Test min mode
    best_run = mlflow_manager.get_best_run("loss", mode="min")
    assert best_run.info.run_id == run_ids[1]  # Second run has lowest loss

def test_list_runs(mlflow_manager):
    """Test listing runs with filters"""
    # Create some runs
    with mlflow_manager:
        mlflow_manager.log_metrics({"metric": 1.0})
    with mlflow_manager:
        mlflow_manager.log_metrics({"metric": 2.0})
    
    # List all runs
    runs = mlflow_manager.list_runs()
    assert len(runs) == 2
    
    # List with status filter
    runs = mlflow_manager.list_runs(status="FINISHED")
    assert len(runs) == 2
    assert all(run.info.status == "FINISHED" for run in runs)
    
    # List with ordering
    runs = mlflow_manager.list_runs(order_by=["metrics.metric DESC"])
    assert runs[0].data.metrics["metric"] > runs[1].data.metrics["metric"]

def test_delete_run(mlflow_manager):
    """Test run deletion"""
    # Create a run
    with mlflow_manager:
        mlflow_manager.log_metrics({"metric": 1.0})
        run_id = mlflow_manager.active_run.info.run_id
    
    # Delete the run
    assert mlflow_manager.delete_run(run_id)
    
    # Verify deletion
    runs = mlflow_manager.list_runs()
    assert not any(run.info.run_id == run_id for run in runs)

def test_log_figure(mlflow_manager):
    """Test figure logging for both matplotlib and plotly"""
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    
    # Test matplotlib figure
    plt_fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    with mlflow_manager:
        mlflow_manager.log_figure(plt_fig, "test_figures", "matplotlib_test")
        run_id = mlflow_manager.active_run.info.run_id
    
    # Verify matplotlib figure was logged
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    print("\nMatplotlib artifacts:", [art.path for art in artifacts])
    
    # List nested artifacts
    for art in artifacts:
        if art.is_dir:
            nested = client.list_artifacts(run_id, art.path)
            print(f"Nested artifacts in {art.path}:", [a.path for a in nested])
    
    # Test plotly figure
    plotly_fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    
    with mlflow_manager:
        mlflow_manager.log_figure(plotly_fig, "test_figures", "plotly_test")
        run_id = mlflow_manager.active_run.info.run_id
    
    # Verify plotly figure was logged
    artifacts = client.list_artifacts(run_id)
    print("\nPlotly artifacts:", [art.path for art in artifacts])
    
    # List nested artifacts
    for art in artifacts:
        if art.is_dir:
            nested = client.list_artifacts(run_id, art.path)
            print(f"Nested artifacts in {art.path}:", [a.path for a in nested])

def test_backtest_results_with_figures(mlflow_manager, sample_data):
    """Test logging backtest results with figures"""
    import matplotlib.pyplot as plt
    
    # Create sample results with figures
    metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.2}
    trades = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=3),
        "type": ["buy", "sell", "buy"],
        "price": [100, 110, 105],
        "quantity": [1, 1, 1]
    })
    portfolio_values = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=3),
        "value": [10000, 10100, 10050]
    })
    
    # Create sample figures
    fig1, ax1 = plt.subplots()
    ax1.plot(portfolio_values["timestamp"], portfolio_values["value"])
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(trades["timestamp"], trades["price"])
    
    figures = {
        "portfolio_performance": fig1,
        "trades_scatter": fig2
    }
    
    results = {
        "metrics": metrics,
        "trades": trades,
        "portfolio_values": portfolio_values,
        "figures": figures
    }
    
    # Log results with figures
    with mlflow_manager:
        mlflow_manager.log_backtest_results(results)
        run_id = mlflow_manager.active_run.info.run_id
    
    # Verify all components were logged
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    # Check metrics
    assert "sharpe_ratio" in run.data.metrics
    assert "max_drawdown" in run.data.metrics
    
    # Check artifacts
    artifacts = client.list_artifacts(run_id)
    print("\nBacktest artifacts:", [art.path for art in artifacts])
    
    # List nested artifacts
    for art in artifacts:
        if art.is_dir:
            nested = client.list_artifacts(run_id, art.path)
            print(f"Nested artifacts in {art.path}:", [a.path for a in nested])

if __name__ == "__main__":
    pytest.main(["-v", __file__])
