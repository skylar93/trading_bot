import mlflow
from contextlib import contextmanager
from typing import Dict, Any, Optional
import logging

class MLflowManager:
    """Manages MLflow experiment tracking and logging."""
    
    def __init__(self):
        """Initialize MLflowManager."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure we're using the test experiment in test environment
        if mlflow.get_experiment_by_name("test_experiment"):
            mlflow.set_experiment("test_experiment")
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            nested: Whether this is a nested run
            
        Yields:
            The active MLflow run
        """
        run = mlflow.start_run(run_name=run_name, nested=nested)
        try:
            yield run
        finally:
            if mlflow.active_run() == run:
                mlflow.end_run()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the current MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run. Metrics will not be logged.")
            return
            
        mlflow.log_metrics(metrics, step=step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run. Parameters will not be logged.")
            return
            
        mlflow.log_params(params)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if mlflow.active_run():
            mlflow.end_run() 