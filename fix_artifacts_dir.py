"""
Fix MLflow artifacts directory structure.

This script fixes the specific issue with the artifacts directory
being mistakenly treated as an experiment.
"""

import os
import sys
import shutil
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("artifacts_fixer")

def fix_artifacts_directory():
    """Fix the artifacts directory structure issue."""
    
    # Current workspace directory
    workspace_dir = os.path.abspath(".")
    
    # Paths
    mlruns_dir = os.path.join(workspace_dir, "mlruns")
    artifacts_dir = os.path.join(mlruns_dir, "artifacts")
    
    if not os.path.exists(mlruns_dir):
        logger.error(f"MLflow directory does not exist: {mlruns_dir}")
        return False
    
    try:
        # Check if artifacts directory exists and is being treated as an experiment
        if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
            logger.info(f"Found artifacts directory: {artifacts_dir}")
            
            # Create a proper artifacts directory structure
            # Move all content to a temporary location
            temp_artifacts = os.path.join(workspace_dir, "temp_artifacts")
            if os.path.exists(temp_artifacts):
                shutil.rmtree(temp_artifacts)
            
            logger.info(f"Moving artifacts content to temporary location: {temp_artifacts}")
            shutil.copytree(artifacts_dir, temp_artifacts)
            
            # Remove the artifacts directory
            logger.info("Removing the artifacts directory")
            shutil.rmtree(artifacts_dir)
            
            # Create a new artifacts directory
            logger.info("Creating a new artifacts directory")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Create a .noexperiment marker file to prevent it from being treated as an experiment
            with open(os.path.join(artifacts_dir, ".noexperiment"), "w") as f:
                f.write("# This file prevents MLflow from treating the artifacts directory as an experiment")
            
            # Move content back
            logger.info("Moving content back to artifacts directory")
            for item in os.listdir(temp_artifacts):
                source = os.path.join(temp_artifacts, item)
                target = os.path.join(artifacts_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            
            # Clean up
            logger.info("Cleaning up temporary directory")
            shutil.rmtree(temp_artifacts)
            
            logger.info("==== Artifacts directory fixed successfully ====")
            logger.info("You can now run 'mlflow ui' to start the UI")
            return True
        else:
            logger.info("Artifacts directory does not exist or is not causing issues")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing artifacts directory: {str(e)}")
        return False

if __name__ == "__main__":
    fix_artifacts_directory() 