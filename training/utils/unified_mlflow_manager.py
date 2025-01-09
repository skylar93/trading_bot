"""Unified MLflow experiment tracking manager"""

import os
import time
import shutil
import tempfile
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.entities import RunStatus
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLflowManager:
    """
    Unified MLflow experiment tracking manager.
    
    Features:
    - Create/get experiment with optional deletion of 'deleted' experiment
    - Start/end run (supports nested runs)
    - Optional dummy metric logging for run validation
    - Log params, metrics, artifacts, PyTorch models
    - Context manager support for automatic run management
    - Optional experiment cleanup and reinitialization
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_dir: Optional[str] = None,
        allow_deleted_experiment_cleanup: bool = False,
        use_dummy_run: bool = False,
        append_timestamp: bool = False
    ):
        """Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_dir: Directory for MLflow artifacts & DB
            allow_deleted_experiment_cleanup: If True, forcibly remove 'deleted' experiments
            use_dummy_run: If True, logs a dummy metric for DB initialization
            append_timestamp: If True, appends timestamp to experiment name
        """
        # For test environments, optionally append timestamp
        if append_timestamp and experiment_name.startswith("test_"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = experiment_name

        self.allow_deleted_experiment_cleanup = allow_deleted_experiment_cleanup
        self.use_dummy_run = use_dummy_run

        # Set up tracking directory
        if tracking_dir:
            self.tracking_dir = Path(tracking_dir).absolute()
        else:
            home_dir = Path.home()
            self.tracking_dir = home_dir / ".trading_bot_mlflow"

        # Create directory with proper permissions
        os.makedirs(self.tracking_dir, mode=0o755, exist_ok=True)

        # Set up MLflow tracking
        db_path = self.tracking_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        self._active_run = None
        self._initialize_experiment()

    def _initialize_experiment(self):
        """Initialize or get existing MLflow experiment."""
        try:
            # End any active runs first
            self.end_active_runs()

            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                if experiment.lifecycle_stage == "deleted":
                    logger.info(f"Found deleted experiment: {self.experiment_name}")
                    if self.allow_deleted_experiment_cleanup:
                        logger.info("Cleaning up deleted experiment...")
                        try:
                            mlflow.delete_experiment(experiment.experiment_id)
                        except mlflow.exceptions.MlflowException as e:
                            logger.warning(f"Failed to delete experiment: {e}")
                        
                        # Clean up related directories
                        paths_to_clean = [
                            self.tracking_dir / ".trash",
                            self.tracking_dir / str(experiment.experiment_id),
                            Path(experiment.artifact_location.replace("file://", ""))
                        ]
                        for path in paths_to_clean:
                            if path.exists():
                                shutil.rmtree(path)
                        
                        experiment = None
                        time.sleep(0.2)  # Wait for cleanup
                    else:
                        logger.warning(
                            "Found deleted experiment but cleanup not allowed. "
                            "This may cause unexpected behavior."
                        )
                        self.experiment_id = experiment.experiment_id
                else:
                    self.experiment_id = experiment.experiment_id

            if not experiment or experiment.lifecycle_stage == "deleted":
                # Create new experiment
                artifact_location = os.path.join(
                    str(self.tracking_dir), "artifacts", self.experiment_name
                )
                os.makedirs(artifact_location, mode=0o755, exist_ok=True)
                
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=artifact_location
                )

            mlflow.set_experiment(self.experiment_name)

            # Optional dummy run
            if self.use_dummy_run:
                with mlflow.start_run() as run:
                    mlflow.log_metric("_dummy", 0.0)
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to initialize experiment: {str(e)}")
            raise

    def end_active_runs(self):
        """End all active MLflow runs."""
        try:
            if self._active_run:
                mlflow.end_run()
                self._active_run = None

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
                self.end_run()

            self._active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags,
                experiment_id=self.experiment_id,
            )

            if self.use_dummy_run:
                mlflow.log_metric("_dummy", 0.0)
                time.sleep(0.05)

            return self._active_run

        except Exception as e:
            logger.error(f"Failed to start run: {str(e)}")
            raise

    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                if self.use_dummy_run:
                    mlflow.log_metric("_dummy_end", 0.0)
                    time.sleep(0.05)
                mlflow.end_run(status=status)
                time.sleep(0.05)
            self._active_run = None
        except Exception as e:
            logger.error(f"Failed to end run: {str(e)}")
            raise

    def cleanup(self):
        """Clean up MLflow resources."""
        try:
            self.end_active_runs()
            if os.path.exists(self.tracking_dir):
                shutil.rmtree(self.tracking_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup resources: {str(e)}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ):
        """Log metrics to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.log_params(params)

    def log_model(self, model: nn.Module, artifact_path: str):
        """Log PyTorch model to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.pytorch.log_model(model, artifact_path)

    def load_model(self, run_id: str, artifact_path: str) -> nn.Module:
        """Load a PyTorch model from MLflow."""
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ):
        """Log a local file as an MLflow artifact."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.log_artifact(local_path, artifact_path)

    def log_dataframe(
        self, df: pd.DataFrame, artifact_path: str, filename: str
    ):
        """Log DataFrame to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        
        suffix = ".parquet" if filename.endswith(".parquet") else ".json"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            if suffix == ".parquet":
                df.to_parquet(f.name)
            else:
                df.to_json(f.name, orient="records", lines=False)
            
            mlflow.log_artifact(f.name, os.path.join(artifact_path, filename))
            os.remove(f.name)

    def log_backtest_results(self, results: Dict[str, Any]):
        """Log backtest results including metrics, trades, portfolio values, and plots.
        
        Args:
            results: Dictionary containing:
                - metrics: Dict of metric values
                - trades: DataFrame of trades
                - portfolio_values: DataFrame of portfolio values
                - figures: Optional dict of matplotlib/plotly figures
        """
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")

        # Log metrics
        if "metrics" in results:
            self.log_metrics(results["metrics"])

        # Log trades
        if "trades" in results and isinstance(results["trades"], pd.DataFrame):
            self.log_dataframe(
                results["trades"], "backtest", "trades.parquet"
            )

        # Log portfolio values
        if "portfolio_values" in results and isinstance(
            results["portfolio_values"], pd.DataFrame
        ):
            self.log_dataframe(
                results["portfolio_values"], 
                "backtest",
                "portfolio_values.parquet"
            )

        # Log figures
        if "figures" in results and isinstance(results["figures"], dict):
            for name, fig in results["figures"].items():
                self.log_figure(fig, "backtest/figures", f"{name}.png")

    def get_best_run(self, metric_name: str, mode: str = "max") -> Optional[mlflow.entities.Run]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            mode: Either 'max' (default) or 'min'
            
        Returns:
            Best run or None if no runs found
        """
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                [self.experiment_id],
                order_by=[f"metrics.{metric_name} {'DESC' if mode == 'max' else 'ASC'}"]
            )
            return runs[0] if runs else None
        except Exception as e:
            logger.error(f"Failed to get best run: {str(e)}")
            return None

    def list_runs(
        self,
        status: Optional[str] = None,
        order_by: Optional[List[str]] = None
    ) -> List[mlflow.entities.Run]:
        """List runs in the experiment.
        
        Args:
            status: Filter by run status (e.g., 'FINISHED', 'RUNNING', etc.)
            order_by: List of columns to sort by (e.g., ['metrics.accuracy DESC'])
            
        Returns:
            List of runs
        """
        try:
            client = mlflow.tracking.MlflowClient()
            filter_string = f"status = '{status}'" if status else None
            runs = client.search_runs(
                [self.experiment_id],
                filter_string=filter_string,
                order_by=order_by
            )
            return runs
        except Exception as e:
            logger.error(f"Failed to list runs: {str(e)}")
            return []

    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID.
        
        Args:
            run_id: ID of the run to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.delete_run(run_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete run {run_id}: {str(e)}")
            return False

    def log_figure(
        self,
        figure: Any,
        artifact_path: str,
        filename: str,
        format: str = "png",
        **kwargs
    ):
        """Log a matplotlib/plotly figure as an artifact.
        
        Args:
            figure: matplotlib.Figure or plotly.Figure
            artifact_path: Directory within artifacts to save to
            filename: Name of the file (without extension)
            format: File format (default: 'png')
            **kwargs: Additional arguments passed to savefig/write_image
        """
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")

        try:
            # Ensure filename has extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create full path for the file
                temp_path = os.path.join(temp_dir, filename)
                
                # Save figure
                if hasattr(figure, "savefig"):  # matplotlib
                    figure.savefig(temp_path, format=format, **kwargs)
                elif hasattr(figure, "write_image"):  # plotly
                    figure.write_image(temp_path, format=format, **kwargs)
                else:
                    raise ValueError("Unsupported figure type")
                
                # Log artifact
                mlflow.log_artifact(temp_path, artifact_path)
                
                # Sleep briefly to ensure file is saved
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to log figure: {str(e)}")
            raise

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
        return False  # Don't suppress exceptions 