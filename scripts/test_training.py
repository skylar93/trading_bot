#!/usr/bin/env python3
"""
Test script for training pipeline with minimal configuration.
Runs a quick test of the training pipeline with a small number of timesteps.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.train_pipeline import train_pipeline
from data.data_fetcher import load_or_fetch_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_training")

def main():
    """Run test training with minimal configuration."""
    # Load configuration
    config_path = os.path.join(project_root, "config", "test_config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load or fetch data
    try:
        data = load_or_fetch_data(config)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Verify data format
        required_columns = ['$open', '$high', '$low', '$close', '$volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Run training pipeline
        results = train_pipeline(config, data)
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        if "best_eval_reward" in results:
            logger.info(f"Best evaluation reward: {results['best_eval_reward']}")
            
        if "episode_rewards" in results:
            import numpy as np
            recent_rewards = results["episode_rewards"][-100:]
            logger.info(f"Average reward over last 100 episodes: {np.mean(recent_rewards):.2f}")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 