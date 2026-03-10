"""
Fix MLflow database issues by creating a new clean setup.

This script creates a completely new MLflow tracking environment based on
the test approach that's known to work correctly.
"""

import os
import sys
import shutil
import tempfile
import logging
import mlflow
from pathlib import Path
from mlflow.utils.file_utils import path_to_local_file_uri

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlflow_fixer")

def fix_mlflow():
    """Create a new clean MLflow environment."""
    # Create a temporary directory for migration
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Current workspace directory
    workspace_dir = os.path.abspath(".")
    
    # Paths
    old_mlruns_dir = os.path.join(workspace_dir, "mlruns")
    old_backup_dir = os.path.join(workspace_dir, "mlruns_old_backup")
    
    try:
        # Backup current mlruns if it exists
        if os.path.exists(old_mlruns_dir):
            if os.path.exists(old_backup_dir):
                shutil.rmtree(old_backup_dir)
            
            logger.info(f"Backing up current mlruns to {old_backup_dir}")
            shutil.copytree(old_mlruns_dir, old_backup_dir)
            
            # Delete existing mlruns directory
            logger.info("Removing current mlruns directory")
            shutil.rmtree(old_mlruns_dir)
        
        # Create new mlruns directory
        os.makedirs(old_mlruns_dir, exist_ok=True)
        
        # Set up MLflow tracking with file URI
        tracking_uri = path_to_local_file_uri(old_mlruns_dir)
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Register a dummy experiment to initialize the database
        experiment_name = "SingleAssetRL_Debug"
        logger.info(f"Creating experiment: {experiment_name}")
        
        # Force reset if experiment exists
        existing_exp = mlflow.get_experiment_by_name(experiment_name)
        if existing_exp:
            logger.info(f"Deleting existing experiment: {experiment_name}")
            mlflow.delete_experiment(existing_exp.experiment_id)
            # Wait for deletion to complete
            import time
            time.sleep(1)
        
        # Create the experiment
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.join(str(old_mlruns_dir), "artifacts", experiment_name)
        )
        logger.info(f"Created experiment with ID: {experiment_id}")
        
        # Log a dummy run to ensure DB is working
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="test_run"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            logger.info("Logged test run successfully")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        logger.info("==== MLflow database fixed successfully ====")
        logger.info(f"You can now run 'mlflow ui' to start the UI")
        logger.info(f"Your old MLflow data is backed up at: {old_backup_dir}")
        logger.info(f"The SingleAssetRL_Debug experiment has been recreated")
        
    except Exception as e:
        logger.error(f"Error fixing MLflow database: {str(e)}")
        logger.info("Attempting to restore from backup...")
        
        try:
            if os.path.exists(old_backup_dir):
                if os.path.exists(old_mlruns_dir):
                    shutil.rmtree(old_mlruns_dir)
                shutil.copytree(old_backup_dir, old_mlruns_dir)
                logger.info("Restored from backup")
        except Exception as restore_error:
            logger.error(f"Failed to restore: {str(restore_error)}")
        
        raise

if __name__ == "__main__":
    fix_mlflow() 