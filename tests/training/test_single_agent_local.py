#!/usr/bin/env python3
"""
A minimal script to test single-agent training locally using the new pipeline.
"""

import logging
import sys
import os

# Adjust python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils.config_manager import ConfigManager
from training.train_pipeline import train_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting local single-agent training test")
    
    # 1) Load config
    config_path = "config/training_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.info("Please ensure you have created the training_config.yaml file")
        return
    
    config_mgr = ConfigManager(config_path)
    config = config_mgr.load_config()
    
    # 2) Force single-agent for local test
    config_mgr.set("env.type", "single_asset_rl")
    
    # 3) Reduce total_timesteps for quick test
    config_mgr.set("training.total_timesteps", 1000)
    
    # 4) Run pipeline
    logger.info("Running training pipeline with reduced timesteps (1000)")
    try:
        results = train_pipeline(config)
        
        # 5) Print out final results
        logger.info("Training complete!")
        if "episode_rewards" in results:
            final_avg_reward = sum(results["episode_rewards"][-100:]) / min(100, len(results["episode_rewards"]))
            logger.info(f"Final average reward: {final_avg_reward:.4f}")
        
        if "best_eval_reward" in results:
            logger.info(f"Best evaluation reward: {results['best_eval_reward']:.4f}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 