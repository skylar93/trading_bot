import mlflow
import logging

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self):
        self.current_run = None
    
    def start_run(self):
        """Start a new MLflow run"""
        self.current_run = mlflow.start_run()
        logger.info(f"Started MLflow run with ID: {self.current_run.info.run_id}")
    
    def end_run(self):
        """End current MLflow run"""
        if self.current_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
    
    def log_params(self, params_dict):
        """Log parameters to MLflow"""
        mlflow.log_params(params_dict)
    
    def log_metrics(self, metrics_dict, step=None):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics_dict, step=step)
    
    def log_artifact(self, local_path):
        """Log an artifact to MLflow"""
        mlflow.log_artifact(local_path)
