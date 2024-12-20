"""MLflow experiment tracking manager"""

import os
import mlflow
import pandas as pd
import numpy as np
import torch
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from mlflow.entities import RunStatus
import tempfile
import json
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow experiment tracking"""
    
    def __init__(self, experiment_name: str, tracking_dir: Optional[str] = None):
        """Initialize MLflow manager
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_dir: Optional directory for MLflow tracking
        """
        # For test environments, append timestamp to experiment name
        if experiment_name.startswith("test_"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Set up tracking directory
        if tracking_dir:
            self.tracking_dir = Path(tracking_dir).absolute()
        else:
            # Use a temporary directory in the user's home directory
            home_dir = Path.home()
            self.tracking_dir = home_dir / ".trading_bot_mlflow"
        
        # Ensure directory exists with proper permissions
        os.makedirs(self.tracking_dir, mode=0o755, exist_ok=True)
        
        # Set up MLflow tracking with file URI
        db_path = self.tracking_dir / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        self._active_run = None
        self.max_retries = 3
        self.retry_delay = 0.5
        
        # Create or get experiment
        try:
            # End any active runs
            if mlflow.active_run():
                mlflow.end_run()
                time.sleep(self.retry_delay)
            
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                artifact_location = os.path.join(str(self.tracking_dir), "artifacts", self.experiment_name)
                os.makedirs(artifact_location, mode=0o755, exist_ok=True)
                
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=artifact_location
                )
            
            mlflow.set_experiment(self.experiment_name)
            time.sleep(0.5)  # Wait for experiment creation
            
            # Create dummy run to ensure database initialization
            with mlflow.start_run() as run:
                mlflow.log_metric("_dummy", 0.0)
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {str(e)}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False, tags: Optional[Dict[str, Any]] = None):
        """Start a new MLflow run."""
        try:
            if nested and not mlflow.active_run():
                raise mlflow.exceptions.MlflowException(
                    "No active parent run found for nested run"
                )
            elif not nested:
                self.end_run()  # End any existing runs
            
            self._active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=tags,
                experiment_id=self.experiment_id
            )
            
            # Log a dummy metric to ensure run is active
            mlflow.log_metric("_dummy", 0.0)
            time.sleep(0.1)  # Small delay to ensure run is registered
            
            return self._active_run
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            raise
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                # Log final dummy metric
                mlflow.log_metric("_dummy_end", 0.0)
                time.sleep(0.1)  # Small delay to ensure metric is logged
                
                # End run
                mlflow.end_run(status=status)
                time.sleep(0.1)  # Small delay to ensure run is ended
                
            self._active_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        try:
            mlflow.log_metrics(metrics, step=step)
            time.sleep(0.1)  # Small delay to ensure metrics are logged
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            raise
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        try:
            mlflow.log_params(params)
            time.sleep(0.1)  # Small delay to ensure parameters are logged
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")
            raise
    
    def log_model(self, model: torch.nn.Module, artifact_path: str):
        """Log PyTorch model to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        try:
            mlflow.pytorch.log_model(model, artifact_path)
            time.sleep(0.1)  # Small delay to ensure model is logged
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise
    
    def load_model(self, run_id: str, artifact_path: str) -> torch.nn.Module:
        """Load a PyTorch model from MLflow
        
        Args:
            run_id: ID of the run containing the model
            artifact_path: Path to the model artifacts
            
        Returns:
            Loaded PyTorch model
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model = mlflow.pytorch.load_model(model_uri)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def log_dataframe(self, df: pd.DataFrame, artifact_path: str, filename: str):
        """Log DataFrame to MLflow."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        try:
            # Save DataFrame to temporary file
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                df.to_parquet(f.name)
                mlflow.log_artifact(f.name, os.path.join(artifact_path, filename))
                os.remove(f.name)
            time.sleep(0.1)  # Small delay to ensure DataFrame is logged
        except Exception as e:
            logger.error(f"Failed to log DataFrame: {str(e)}")
            raise
    
    def log_backtest_results(self, df_results: pd.DataFrame, metrics: dict, trades: list):
        """Log backtest results to MLflow
        
        Args:
            df_results: DataFrame with backtest results
            metrics: Dictionary of metrics
            trades: List of trades
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        
        try:
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log results DataFrame
            self.log_dataframe(df_results, "backtest", "results.json")
            
            # Convert trades list to DataFrame and log
            trades_df = pd.DataFrame(trades)
            self.log_dataframe(trades_df, "backtest", "trades.json")
            
        except Exception as e:
            logger.error(f"Failed to log backtest results: {str(e)}")
            raise
    
    def get_artifact_uri(self, artifact_path: str) -> str:
        """Get the URI for an MLflow artifact."""
        if not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active run")
        try:
            return mlflow.get_artifact_uri(artifact_path)
        except Exception as e:
            logger.error(f"Failed to get artifact URI: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up MLflow resources."""
        try:
            # End any active runs
            if mlflow.active_run():
                mlflow.end_run()
            self._active_run = None
            
            # Delete tracking directory if it exists
            if os.path.exists(self.tracking_dir):
                shutil.rmtree(self.tracking_dir)
                time.sleep(0.1)  # Small delay to ensure cleanup is complete
        except Exception as e:
            logger.error(f"Failed to clean up MLflow resources: {str(e)}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        status = "FAILED" if exc_type is not None else "FINISHED"
        if mlflow.active_run():
            mlflow.end_run(status=status)
        self._active_run = None
