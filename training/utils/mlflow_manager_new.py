"""MLflow experiment tracking manager"""

import os
import mlflow
from mlflow.entities import RunStatus
import logging
import warnings
import yaml
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import torch
import shutil
import time
import tempfile
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow experiment tracking with standardized paths and formats."""

    def __init__(
        self,
        experiment_name: str = "default",
        tracking_dir: str = "./mlflow_runs",
    ):
        """Initialize MLflow experiment manager."""
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir).absolute()
        self._active_run = None

        # Set up MLflow tracking
        os.makedirs(self.tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{self.tracking_dir}")

        # Initialize experiment
        self._initialize_experiment()

    def _initialize_experiment(self):
        """Initialize or get existing MLflow experiment."""
        try:
            # End any active runs first
            self.end_active_runs()

            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=os.path.join(
                        self.tracking_dir, self.experiment_name, "artifacts"
                    ),
                )

            # Set as active experiment
            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.error(f"Failed to initialize MLflow experiment: {str(e)}")
            raise

    def end_active_runs(self):
        """End all active MLflow runs."""
        try:
            # End our tracked active run
            if self._active_run:
                mlflow.end_run()
                self._active_run = None

            # End any other active runs
            while mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Error ending active runs: {str(e)}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Start a new MLflow run."""
        try:
            if nested and not mlflow.active_run():
                raise mlflow.exceptions.MlflowException(
                    "No active parent run found for nested run"
                )
            elif not nested:
                self.end_active_runs()

            self._active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags,
                experiment_id=self.experiment_id,
            )
            return self._active_run

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            raise

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        try:
            # End our tracked run first if it exists
            if self._active_run:
                mlflow.end_run(status=status)
                self._active_run = None

            # End any remaining active runs
            while mlflow.active_run():
                mlflow.end_run(status=status)
        except Exception as e:
            logger.warning(f"Failed to end run: {str(e)}")
            # Try to force end all runs
            try:
                while mlflow.active_run():
                    mlflow.end_run(status=status)
                self._active_run = None
            except Exception:
                pass

    def cleanup(self):
        """Clean up MLflow resources."""
        try:
            self.end_active_runs()
            if os.path.exists(self.tracking_dir):
                shutil.rmtree(self.tracking_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup MLflow resources: {str(e)}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ):
        """Log metrics to MLflow."""
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run found")
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run found")
        mlflow.log_params(params)

    def log_model(self, model: nn.Module, artifact_path: str):
        """Log PyTorch model to MLflow."""
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run found")
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            mlflow.log_artifact(f.name, artifact_path)
            os.remove(f.name)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ):
        """Log a local file as an MLflow artifact."""
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run exists")
        mlflow.log_artifact(local_path, artifact_path)

    def log_backtest_results(self, results: Dict[str, Any]):
        """Log backtest results including metrics, trades, and portfolio values."""
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run exists")

        if "metrics" in results:
            self.log_metrics(results["metrics"])

        if "trades" in results and isinstance(results["trades"], pd.DataFrame):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as f:
                results["trades"].to_parquet(f.name)
                self.log_artifact(f.name, "backtest/trades.parquet")
                os.remove(f.name)

        if "portfolio_values" in results and isinstance(
            results["portfolio_values"], pd.DataFrame
        ):
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as f:
                results["portfolio_values"].to_parquet(f.name)
                self.log_artifact(f.name, "backtest/portfolio_values.parquet")
                os.remove(f.name)

    @property
    def experiment(self):
        """Get the current experiment."""
        return mlflow.get_experiment(self.experiment_id)

    @property
    def active_run(self):
        """Get the current active run."""
        return self._active_run

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)
