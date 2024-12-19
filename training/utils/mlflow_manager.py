"""MLflow experiment tracking manager"""

import os
import mlflow
import pandas as pd
import numpy as np
import torch
import logging
import time
from pathlib import Path
import shutil
from typing import Dict, Any, Optional, List, Union
from mlflow.entities import RunStatus
import tempfile
import json

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow experiment tracking"""
    
    def __init__(self, experiment_name: str = "trading_bot", tracking_dir: Optional[str] = None):
        """Initialize MLflow manager
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_dir: Optional directory for MLflow tracking
        """
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
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> mlflow.ActiveRun:
        """Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Returns:
            Active MLflow run
        """
        # End any existing non-nested run
        if not nested and mlflow.active_run():
            mlflow.end_run()
            time.sleep(self.retry_delay)
        
        if nested and not mlflow.active_run():
            raise mlflow.exceptions.MlflowException("No active parent run found for nested run")
        
        # Start run with retries
        for attempt in range(self.max_retries):
            try:
                self._active_run = mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    experiment_id=self.experiment_id
                )
                
                # Log dummy metric
                mlflow.log_metric("_dummy", 0.0)
                time.sleep(self.retry_delay)
                
                # Verify run creation
                run_id = self._active_run.info.run_id
                run = mlflow.get_run(run_id)
                
                if run and run.info.run_id == run_id:
                    break
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to start MLflow run: {str(e)}")
                    raise
                time.sleep(self.retry_delay)
        
        return self._active_run
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End current MLflow run
        
        Args:
            status: Run status (FINISHED, FAILED, etc.)
        """
        try:
            # Log final dummy metric
            if self._active_run:
                mlflow.log_metric("_dummy_end", 0.0)
                time.sleep(self.retry_delay)
            
            # End run
            if mlflow.active_run():
                mlflow.end_run(status=status)
                time.sleep(self.retry_delay)
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {str(e)}")
            raise
        finally:
            self._active_run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.log_metrics(metrics, step=step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.log_params(params)
    
    def log_model(self, model: torch.nn.Module, artifact_path: str) -> None:
        """Log PyTorch model to MLflow
        
        Args:
            model: PyTorch model to log
            artifact_path: Path to save model artifacts
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        mlflow.pytorch.log_model(model, artifact_path)
    
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
    
    def log_dataframe(self, df: pd.DataFrame, artifact_dir: str, filename: str) -> None:
        """Log DataFrame to MLflow
        
        Args:
            df: DataFrame to log
            artifact_dir: Directory for artifacts
            filename: Name of the file
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        
        try:
            # Convert DataFrame to dictionary with datetime handling
            df_dict = {}
            for column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df_dict[column] = df[column].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                else:
                    df_dict[column] = df[column].tolist()
            
            # Create temporary directory structure
            temp_dir = tempfile.mkdtemp()
            try:
                # Create artifact directory
                artifact_path = os.path.join(temp_dir, artifact_dir)
                os.makedirs(artifact_path, exist_ok=True)
                
                # Save DataFrame to file
                temp_file = os.path.join(artifact_path, filename)
                with open(temp_file, 'w') as f:
                    json.dump(df_dict, f, indent=2)
                
                # Log artifact with relative path
                mlflow.log_artifact(temp_file, artifact_dir)
                
                # List artifacts to verify
                client = mlflow.tracking.MlflowClient()
                artifacts = [artifact.path for artifact in 
                           client.list_artifacts(mlflow.active_run().info.run_id, artifact_dir)]
                logger.debug(f"Current artifacts in {artifact_dir}: {artifacts}")
                
            finally:
                shutil.rmtree(temp_dir)
            
        except Exception as e:
            logger.error(f"Failed to log DataFrame: {str(e)}")
            raise
    
    def log_backtest_results(self, df_results: pd.DataFrame, metrics: dict, trades: list) -> None:
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
        """Get URI for an artifact
        
        Args:
            artifact_path: Path to the artifact
            
        Returns:
            Full URI for the artifact
        """
        if not self._active_run:
            raise mlflow.exceptions.MlflowException("No active run")
        return mlflow.get_artifact_uri(artifact_path)
    
    def cleanup(self) -> None:
        """Clean up MLflow resources"""
        try:
            # End active run
            if mlflow.active_run():
                mlflow.end_run()
                time.sleep(self.retry_delay)
            
            # Delete experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                mlflow.delete_experiment(experiment.experiment_id)
                time.sleep(0.5)
            
            # Clean up tracking directory
            if self.tracking_dir and self.tracking_dir.exists():
                shutil.rmtree(self.tracking_dir)
                
        except Exception as e:
            logger.error(f"Failed to cleanup MLflow resources: {str(e)}")
            raise
    
    def __enter__(self):
        """Context manager entry"""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
