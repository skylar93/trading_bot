"""
Training Pipeline Validation Script.

This script runs a quick validation of the training pipeline with a small number of timesteps
to catch any potential issues before submitting to SLURM for long computations.

Features:
- Validates all training pipeline modes (single/multi agent, single/multi asset)
- Checks for common errors and issues
- Reports key metrics to verify training is working correctly
- Uses a small subset of data for quick validation

Implementation Notes:
- Runs with reduced timesteps for quick feedback
- Verifies MLflow logging is working correctly
- Checks model saving and loading
- Tests agent interaction in multi-agent scenarios
"""

import os
import sys
import logging
import yaml
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.train_pipeline import train_pipeline
from training.env_factory import load_data
from training.utils.unified_mlflow_manager import MLflowManager
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("validation")

def create_test_data() -> pd.DataFrame:
    """Create small synthetic data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=500)
    
    # Start price and random walk
    price = 100.0
    prices = [price]
    
    # Generate a price time series with some randomness and trends
    for _ in range(499):
        # Random shock
        shock = np.random.normal(0, 1)
        
        # Small trend and mean reversion
        trend = 0.1 if np.random.random() > 0.5 else -0.1
        reversion = 0.05 * (100 - price)
        
        # Update price
        price *= 1 + 0.001 * (shock + trend + reversion)
        prices.append(price)
    
    # Create dataframe with OHLCV data
    df = pd.DataFrame(index=dates)
    df['$open'] = prices
    df['$high'] = [p * (1 + np.random.uniform(0, 0.01)) for p in prices]
    df['$low'] = [p * (1 - np.random.uniform(0, 0.01)) for p in prices]
    df['$close'] = [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices]
    df['$volume'] = [np.random.uniform(1000, 5000) for _ in prices]
    
    return df

def load_test_config(config_path: str) -> Dict[str, Any]:
    """Load config and modify for quick testing."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Reduce training time for quick testing
    if 'training' in config:
        config['training']['total_timesteps'] = 5000
        config['training']['checkpoint_interval'] = 1000
        config['training']['eval_interval'] = 1000
        config['training']['log_interval'] = 100
    
    # Set paths to test directories
    if 'paths' in config:
        config['paths']['checkpoint_dir'] = 'test_checkpoints'
        config['paths']['log_dir'] = 'test_logs'
    
    return config

def validate_single_agent_training() -> bool:
    """Validate single-agent training pipeline."""
    logger.info("Validating single-agent training pipeline...")
    
    # Load and modify config
    config_path = os.path.join(project_root, "configs", "single_agent.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = load_test_config(config_path)
    
    # Create test data
    data = create_test_data()
    
    # Run training
    try:
        results = train_pipeline(config, data)
        
        # Check for key metrics that indicate successful training
        if "best_eval_rewards" not in results:
            logger.error("Missing 'best_eval_rewards' in results")
            return False
        
        if "training_time" not in results:
            logger.error("Missing 'training_time' in results")
            return False
        
        # Check if model was saved
        if "final_model_paths" not in results:
            logger.error("Missing 'final_model_paths' in results")
            return False
        
        for path in results["final_model_paths"].values():
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
        
        # Report metrics
        logger.info(f"Training completed in {results['training_time']:.2f} seconds")
        logger.info(f"Best evaluation reward: {results.get('best_eval_rewards', {}).get('agent', 'N/A')}")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in single-agent validation: {e}")
        return False

def validate_multi_agent_training() -> bool:
    """Validate multi-agent training pipeline."""
    logger.info("Validating multi-agent training pipeline...")
    
    # Load and modify config
    config_path = os.path.join(project_root, "configs", "multi_agent.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = load_test_config(config_path)
    
    # Create test data
    data = create_test_data()
    
    # Run training
    try:
        results = train_pipeline(config, data)
        
        # Check for key metrics
        if "best_eval_rewards" not in results:
            logger.error("Missing 'best_eval_rewards' in results")
            return False
        
        if "agent_interactions" not in results:
            logger.warning("No 'agent_interactions' data in results")
        
        # Check if models were saved
        if "final_model_paths" not in results:
            logger.error("Missing 'final_model_paths' in results")
            return False
        
        for agent_id, path in results["final_model_paths"].items():
            if not os.path.exists(path):
                logger.error(f"Model file not found for {agent_id}: {path}")
                return False
        
        # Report metrics for each agent
        logger.info(f"Training completed in {results.get('training_time', 0):.2f} seconds")
        for agent_id, reward in results.get("best_eval_rewards", {}).items():
            logger.info(f"Agent {agent_id} best evaluation reward: {reward}")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in multi-agent validation: {e}")
        return False

def validate_multi_asset_training() -> bool:
    """Validate multi-asset training pipeline."""
    logger.info("Validating multi-asset training pipeline...")
    
    # Load and modify config
    config_path = os.path.join(project_root, "configs", "multi_asset.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = load_test_config(config_path)
    
    # Create multi-asset test data
    assets = ['BTC', 'ETH', 'LTC']
    all_data = []
    
    for asset in assets:
        data = create_test_data()
        data['asset'] = asset
        all_data.append(data)
    
    combined_data = pd.concat(all_data)
    
    # Run training
    try:
        results = train_pipeline(config, combined_data)
        
        # Check for key metrics
        if "best_eval_rewards" not in results:
            logger.error("Missing 'best_eval_rewards' in results")
            return False
        
        if "asset_allocations" not in results:
            logger.warning("No 'asset_allocations' data in results")
        
        # Report metrics
        logger.info(f"Training completed in {results.get('training_time', 0):.2f} seconds")
        logger.info(f"Best evaluation reward: {results.get('best_eval_rewards', {}).get('agent', 'N/A')}")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in multi-asset validation: {e}")
        return False

def validate_multi_agent_multi_asset_training() -> bool:
    """Validate multi-agent multi-asset training pipeline."""
    logger.info("Validating multi-agent multi-asset training pipeline...")
    
    # Load and modify config
    config_path = os.path.join(project_root, "configs", "multi_agent_multi_asset.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    config = load_test_config(config_path)
    
    # Create multi-asset test data
    assets = ['BTC', 'ETH', 'LTC']
    all_data = []
    
    for asset in assets:
        data = create_test_data()
        data['asset'] = asset
        all_data.append(data)
    
    combined_data = pd.concat(all_data)
    
    # Run training
    try:
        results = train_pipeline(config, combined_data)
        
        # Check for key metrics
        if "best_eval_rewards" not in results:
            logger.error("Missing 'best_eval_rewards' in results")
            return False
        
        # Check for meta-agent results if using meta ensemble
        ensemble_method = config.get("env", {}).get("ensemble_method", "")
        if ensemble_method == "meta" and "meta_agent" not in results.get("best_eval_rewards", {}):
            logger.warning("Meta agent results not found")
        
        # Report metrics for each agent
        logger.info(f"Training completed in {results.get('training_time', 0):.2f} seconds")
        for agent_id, reward in results.get("best_eval_rewards", {}).items():
            logger.info(f"Agent {agent_id} best evaluation reward: {reward}")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error in multi-agent multi-asset validation: {e}")
        return False

def check_mlflow_logging():
    """Verify MLflow logging is working correctly."""
    logger.info("Checking MLflow logging...")
    
    try:
        # Get active experiment
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        if not experiments:
            logger.error("No MLflow experiments found")
            return False
        
        logger.info(f"Found {len(experiments)} experiments")
        
        # Check recent runs
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id)
            if runs:
                logger.info(f"Experiment '{exp.name}' has {len(runs)} runs")
                
                # Check most recent run
                most_recent = sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]
                run_id = most_recent.info.run_id
                
                logger.info(f"Most recent run: {run_id}")
                metrics = client.get_run(run_id).data.metrics
                logger.info(f"Metrics: {metrics}")
                
                # Check for artifacts
                artifacts = client.list_artifacts(run_id)
                logger.info(f"Artifacts: {[a.path for a in artifacts]}")
        
        return True
    
    except Exception as e:
        logger.exception(f"Error checking MLflow logging: {e}")
        return False

def main():
    """Run validation for all training pipelines."""
    start_time = time.time()
    logger.info("Starting training pipeline validation...")
    
    validation_results = {
        "single_agent": validate_single_agent_training(),
        "multi_agent": validate_multi_agent_training(),
        "multi_asset": validate_multi_asset_training(),
        "multi_agent_multi_asset": validate_multi_agent_multi_asset_training(),
        "mlflow_logging": check_mlflow_logging()
    }
    
    # Report overall results
    logger.info(f"Validation completed in {time.time() - start_time:.2f} seconds")
    logger.info("Validation results:")
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    # Overall status
    if all(validation_results.values()):
        logger.info("All validations passed! Safe to proceed with SLURM jobs.")
        return 0
    else:
        logger.error("Some validations failed! Fix issues before proceeding with SLURM jobs.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 