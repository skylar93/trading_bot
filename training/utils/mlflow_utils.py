import mlflow
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import torch
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow experiment tracking"""
    
    def __init__(self, experiment_name: str = "trading_bot"):
        """Initialize MLflow experiment"""
        self.experiment_name = experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Setup MLflow experiment"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
        logger.info(f"Using MLflow experiment: {self.experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run"""
        # End any active run before starting a new one
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()
            
        if not run_name:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
    
    def end_run(self):
        """End current MLflow run"""
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, artifact_path: str = "model"):
        """Log PyTorch model to MLflow"""
        active_run = mlflow.active_run()
        if active_run is not None:
            mlflow.pytorch.log_model(model, artifact_path)
    
    def log_training_metrics(self, 
                           episode: int,
                           train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float]):
        """Log training and validation metrics"""
        # Add prefix to distinguish between train and val metrics
        train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        # Combine metrics and log
        metrics = {**train_metrics, **val_metrics}
        self.log_metrics(metrics, step=episode)