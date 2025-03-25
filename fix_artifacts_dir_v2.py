"""
Fix MLflow artifacts directory metadata issues.

This script adds a proper meta.yaml file to the artifacts directory to prevent
the 'Malformed experiment' error.
"""

import os
import sys
import yaml
import logging
import mlflow
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlflow_artifacts_fixer")

def fix_artifacts_directory():
    """Add proper metadata to the artifacts directory to prevent errors."""
    
    # Current workspace directory
    workspace_dir = os.path.abspath(".")
    
    # Paths
    mlruns_dir = os.path.join(workspace_dir, "mlruns")
    artifacts_dir = os.path.join(mlruns_dir, "artifacts")
    meta_file = os.path.join(artifacts_dir, "meta.yaml")
    
    if not os.path.exists(mlruns_dir):
        logger.error(f"MLflow directory does not exist: {mlruns_dir}")
        return False
    
    try:
        # Check if artifacts directory exists
        if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
            logger.info(f"Found artifacts directory: {artifacts_dir}")
            
            # Create or update meta.yaml if it doesn't exist
            if not os.path.exists(meta_file):
                logger.info(f"Creating meta.yaml in artifacts directory")
                
                # Create a simple meta.yaml to make MLflow recognize this as a valid experiment
                meta_data = {
                    "artifact_location": str(Path(artifacts_dir).absolute()),
                    "experiment_id": "artifacts",
                    "lifecycle_stage": "deleted",  # Mark as deleted so MLflow ignores it
                    "name": ".artifacts",
                    "creation_time": 0,
                    "last_update_time": 0
                }
                
                # Write the meta.yaml file
                with open(meta_file, 'w') as f:
                    yaml.safe_dump(meta_data, f)
                
                logger.info("Added meta.yaml to artifacts directory")
                
                # Add empty .trash directory for completeness
                trash_dir = os.path.join(artifacts_dir, ".trash")
                if not os.path.exists(trash_dir):
                    os.makedirs(trash_dir, exist_ok=True)
                
                return True
            else:
                logger.info(f"meta.yaml already exists in artifacts directory")
                return True
        else:
            # Create artifacts directory if it doesn't exist
            logger.info(f"Creating artifacts directory: {artifacts_dir}")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Create a simple meta.yaml
            meta_data = {
                "artifact_location": str(Path(artifacts_dir).absolute()),
                "experiment_id": "artifacts",
                "lifecycle_stage": "deleted",  # Mark as deleted so MLflow ignores it
                "name": ".artifacts",
                "creation_time": 0,
                "last_update_time": 0
            }
            
            # Write the meta.yaml file
            with open(meta_file, 'w') as f:
                yaml.safe_dump(meta_data, f)
            
            logger.info("Created artifacts directory with meta.yaml")
            
            # Add empty .trash directory for completeness
            trash_dir = os.path.join(artifacts_dir, ".trash")
            os.makedirs(trash_dir, exist_ok=True)
            
            return True
            
    except Exception as e:
        logger.error(f"Error fixing artifacts directory: {str(e)}")
        return False

def fix_mlflow_experiments():
    """Fix all experiment directories to ensure consistent structure."""
    
    # Current workspace directory
    workspace_dir = os.path.abspath(".")
    
    # Paths
    mlruns_dir = os.path.join(workspace_dir, "mlruns")
    
    if not os.path.exists(mlruns_dir):
        logger.error(f"MLflow directory does not exist: {mlruns_dir}")
        return False
    
    try:
        logger.info(f"Scanning MLflow directory: {mlruns_dir}")
        
        # Scan for experiment directories
        for item in os.listdir(mlruns_dir):
            item_path = os.path.join(mlruns_dir, item)
            
            # Skip files, .trash, etc.
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
                
            # Skip known special folders
            if item in ['artifacts', 'models']:
                continue
                
            meta_file = os.path.join(item_path, "meta.yaml")
            
            # If meta.yaml doesn't exist, this is not a valid experiment
            if not os.path.exists(meta_file):
                logger.warning(f"Directory {item} is missing meta.yaml, skipping")
                continue
                
            # Read and update meta.yaml to ensure experiment_id is a string
            try:
                with open(meta_file, 'r') as f:
                    meta_data = yaml.safe_load(f)
                    
                if 'experiment_id' in meta_data:
                    # Ensure experiment_id is a string
                    meta_data['experiment_id'] = str(meta_data['experiment_id'])
                    
                    # Rewrite the file
                    with open(meta_file, 'w') as f:
                        yaml.safe_dump(meta_data, f)
                        
                    logger.info(f"Updated experiment_id in {item}/meta.yaml")
            except Exception as e:
                logger.warning(f"Error processing {meta_file}: {str(e)}")
        
        logger.info("Finished processing experiment directories")
        return True
            
    except Exception as e:
        logger.error(f"Error fixing experiment directories: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting MLflow artifacts directory fix...")
    
    # Fix artifacts directory first
    if fix_artifacts_directory():
        logger.info("Artifacts directory fix successful")
    else:
        logger.error("Failed to fix artifacts directory")
        
    # Fix experiment directories
    if fix_mlflow_experiments():
        logger.info("Experiment directories fix successful")
    else:
        logger.error("Failed to fix experiment directories")
        
    logger.info("MLflow directory structure fix completed")
    logger.info("You can now run 'mlflow ui --port 5000' to start the UI") 