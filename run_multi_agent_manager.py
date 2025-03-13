#!/usr/bin/env python3
"""
Run multi-agent training with the MultiAgentManager for coordinated training.

This script demonstrates how to use the MultiAgentManager to train multiple agents
together with a meta-agent that learns to coordinate their actions.
"""

import os
import sys
import logging
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.train_pipeline import train_pipeline
from utils.mlflow_manager import MLflowManager
from utils.log_utils import setup_logging

# Configure logging
logger = setup_logging("run_multi_agent_manager", log_level=logging.INFO)

def main():
    """Run multi-agent training with manager."""
    parser = argparse.ArgumentParser(description="Train multiple agents with MultiAgentManager")
    parser.add_argument("--config", type=str, default="config/multi_agent_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--ensemble-method", type=str, default=None, choices=["weighted", "best", "meta"],
                        help="Ensemble method to use (overrides config)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps for training (overrides config)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Ensure config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.ensemble_method is not None:
        logger.info(f"Overriding ensemble method to: {args.ensemble_method}")
        config["env"]["ensemble_method"] = args.ensemble_method
    
    if args.timesteps is not None:
        logger.info(f"Overriding total timesteps to: {args.timesteps}")
        config["training"]["total_timesteps"] = args.timesteps
    
    # Ensure the environment type is set
    if "type" not in config["env"]:
        config["env"]["type"] = "multi_agent_rl"
    
    # Ensure use_manager is set to True
    config["env"]["use_manager"] = True
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["paths"].get("log_dir", "logs")) / f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the effective configuration
    config_path = run_dir / "effective_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved effective configuration to {config_path}")
    
    # Log configuration details
    ensemble_method = config["env"].get("ensemble_method", "weighted")
    num_agents = len(config["env"].get("multi_agent_configs", []))
    logger.info(f"Starting multi-agent training with {num_agents} agents")
    logger.info(f"Ensemble method: {ensemble_method}")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Run the training pipeline
        results = train_pipeline(config)
        
        # Log results
        logger.info("Training completed successfully")
        logger.info(f"Best evaluation rewards: {results.get('best_eval_rewards', {})}")
        logger.info(f"Training time: {results.get('training_time', 0):.2f} seconds")
        
        # Save results summary
        results_path = run_dir / "results_summary.yaml"
        serializable_results = {
            k: v for k, v in results.items() 
            if k not in ["manager", "agents"] and isinstance(v, (dict, list, str, int, float, bool))
        }
        with open(results_path, "w") as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
        logger.info(f"Saved results summary to {results_path}")
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 