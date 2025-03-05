"""
Example script showing how to use the unified configuration system for training.

This script demonstrates:
1. Loading the unified configuration
2. Modifying configuration settings
3. Setting up a training run
4. Saving the configuration snapshot for reproducibility

Features:
- Configuration-driven training setup
- Examples for both single-agent and multi-agent modes
- Integration with MLflow for tracking

Implementation Notes:
- Uses ConfigManager to handle all configuration aspects
- Creates proper experiment tracking for reproducibility
- Shows how to fetch data and set up environments

Recent Changes:
- Added support for UW Hyak SLURM parameters
- Integrated with the unified config system
- Added multi-agent configuration example
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils.config_manager import ConfigManager
from agents.strategies.agent_factory import create_agent
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from envs.multi_agent_env import MultiAgentTradingEnv
from training.utils.unified_mlflow_manager import MLflowManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_data(config):
    """
    Fetch and prepare data based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with prepared data
    """
    data_config = config["data"]
    
    # Check if direct path provided
    if "data_path" in data_config and os.path.exists(data_config["data_path"]):
        logger.info(f"Loading data from file: {data_config['data_path']}")
        df = pd.read_csv(data_config["data_path"])
        return df
    
    # Otherwise, fetch from exchange using CCXT
    # This is a placeholder - you would implement the actual data fetching
    logger.info(f"Would fetch data from {data_config['exchange']} for {data_config['symbols']}")
    logger.info(f"Timeframe: {data_config['timeframe']}")
    logger.info(f"Date range: {data_config['start_date']} to {data_config['end_date']}")
    
    # For this example, we'll create some dummy data
    import numpy as np
    
    num_samples = 1000
    dates = pd.date_range(
        start=data_config["start_date"], 
        end=data_config["end_date"], 
        periods=num_samples
    )
    
    # Create dummy OHLCV data with $-prefixed column names
    df = pd.DataFrame({
        "timestamp": dates,
        "$open": np.random.normal(100, 10, num_samples).cumsum(),
        "$high": np.random.normal(100, 10, num_samples).cumsum() * 1.02,
        "$low": np.random.normal(100, 10, num_samples).cumsum() * 0.98,
        "$close": np.random.normal(100, 10, num_samples).cumsum() * 1.01,
        "$volume": np.abs(np.random.normal(1000, 200, num_samples))
    })
    
    logger.info(f"Created dummy data with {len(df)} rows")
    return df

def setup_single_agent_training(config, data):
    """
    Set up a single-agent training run.
    
    Args:
        config: Configuration dictionary
        data: DataFrame with prepared data
        
    Returns:
        Environment and agent objects
    """
    # Create environment
    env_config = config["env"]
    env = SingleAssetRLTradingEnv(
        data=data,
        initial_balance=env_config["initial_balance"],
        trading_fee=env_config["trading_fee"],
        window_size=env_config["window_size"],
        max_position_size=env_config.get("max_position_size", 1.0),
    )
    
    # Wrap environment if needed
    if env_config.get("normalize", False):
        from envs.wrap_env import make_env
        env = make_env(
            env, 
            normalize=env_config.get("normalize", True),
            stack_size=env_config.get("stack_size", 4)
        )
    
    # Create agent
    agent_config = config["agent"]
    agent = create_agent(
        agent_name=agent_config["name"],
        config={
            "env": env,
            "learning_rate": agent_config["learning_rate"],
            "gamma": agent_config["gamma"],
            "gae_lambda": agent_config.get("gae_lambda", 0.95),
            "clip_epsilon": agent_config["clip_epsilon"],
            "value_coef": agent_config.get("value_coef", 1.0),
            "entropy_coef": agent_config.get("entropy_coef", 0.01),
            "loss_coef": agent_config.get("loss_coef", 0.5),
            "batch_size": agent_config["batch_size"],
            "n_epochs": agent_config.get("n_epochs", 10),
            "target_kl": agent_config.get("target_kl", 0.015),
            "fcnet_hiddens": agent_config.get("fcnet_hiddens", [64, 64]),
            "activation": agent_config.get("activation", "tanh"),
            "use_lstm": agent_config.get("use_lstm", False),
            "lstm_size": agent_config.get("lstm_size", 128),
        }
    )
    
    return env, agent

def setup_multi_agent_training(config, data):
    """
    Set up a multi-agent training run.
    
    Args:
        config: Configuration dictionary
        data: DataFrame with prepared data
        
    Returns:
        Environment and dictionary of agents
    """
    # Create environment
    env_config = config["env"]
    multi_agent_configs = env_config["multi_agent_configs"]
    
    # Create a list of agent configurations for the environment
    agent_configs = []
    for agent_cfg in multi_agent_configs:
        agent_configs.append({
            "id": agent_cfg["id"],
            "type": agent_cfg["type"],
            "initial_capital_percentage": agent_cfg.get("initial_capital_percentage", 1.0),
            "priority": agent_cfg.get("priority", 1),
            # Additional environment-related agent parameters
        })
    
    # Create the multi-agent environment
    env = MultiAgentTradingEnv(
        data=data,
        initial_balance=env_config["initial_balance"],
        trading_fee=env_config["trading_fee"],
        window_size=env_config["window_size"],
        agent_configs=agent_configs
    )
    
    # Create agents
    agents = {}
    for agent_cfg in multi_agent_configs:
        # Either use the agent's hyperparameters or the default ones
        hyperparams = agent_cfg.get("hyperparameters", {})
        
        # Build agent config by merging with default agent config
        agent_params = {
            "env": env,
            "learning_rate": hyperparams.get("learning_rate", config["agent"]["learning_rate"]),
            "gamma": hyperparams.get("gamma", config["agent"]["gamma"]),
            "clip_epsilon": hyperparams.get("clip_epsilon", config["agent"]["clip_epsilon"]),
            # Include other parameters with defaults from main agent config
        }
        
        # Create agent
        agents[agent_cfg["id"]] = create_agent(
            agent_name=agent_cfg["type"],
            config=agent_params
        )
    
    return env, agents

def main():
    """
    Main function to run the training example.
    """
    parser = argparse.ArgumentParser(description="Run training with unified config")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                        help="Path to config file")
    parser.add_argument("--multi-agent", action="store_true", 
                        help="Use multi-agent training mode")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment ID for MLflow tracking")
    args = parser.parse_args()
    
    # Generate experiment ID if not provided
    experiment_id = args.experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load configuration
    config_mgr = ConfigManager(default_config_path=args.config)
    config = config_mgr.load_config()
    
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override env type based on arguments
    if args.multi_agent:
        config_mgr.set("env.type", "multi_agent_rl")
        logger.info("Set environment type to multi_agent_rl")
    
    # Fetch data
    data = fetch_data(config)
    
    # Initialize MLflow for tracking
    mlflow_manager = MLflowManager(
        tracking_uri=config["paths"]["mlflow_tracking_uri"],
        experiment_name=experiment_id
    )
    
    # Setup based on environment type
    env_type = config["env"]["type"]
    
    if env_type == "single_asset_rl":
        logger.info("Setting up single-agent training")
        env, agent = setup_single_agent_training(config, data)
        
        # Log config parameters to MLflow
        with mlflow_manager.start_run("training") as mlflow_run:
            # Log key parameters
            mlflow_manager.log_params({
                "env_type": env_type,
                "agent_type": config["agent"]["name"],
                "learning_rate": config["agent"]["learning_rate"],
                "total_timesteps": config["training"]["total_timesteps"]
            })
            
            # Save configuration snapshot
            config_snapshot_path = config_mgr.save_snapshot(experiment_id)
            logger.info(f"Saved config snapshot to {config_snapshot_path}")
            
            # This is where you would run your training loop
            logger.info("Would run single-agent training here")
            logger.info(f"Environment: {env}")
            logger.info(f"Agent: {agent}")
            logger.info(f"Training for {config['training']['total_timesteps']} timesteps")
            
            # For this example, just log some dummy metrics
            for i in range(5):
                mlflow_manager.log_metrics({
                    "reward": i * 10,
                    "loss": 1.0 / (i + 1)
                }, step=i)
    
    elif env_type == "multi_agent_rl":
        logger.info("Setting up multi-agent training")
        env, agents = setup_multi_agent_training(config, data)
        
        # Log config parameters to MLflow
        with mlflow_manager.start_run("multi_agent_training") as mlflow_run:
            # Log environment parameters
            mlflow_manager.log_params({
                "env_type": env_type,
                "num_agents": len(agents),
                "agent_types": ",".join([a_cfg["type"] for a_cfg in config["env"]["multi_agent_configs"]]),
                "total_timesteps": config["training"]["total_timesteps"]
            })
            
            # Save configuration snapshot
            config_snapshot_path = config_mgr.save_snapshot(experiment_id)
            logger.info(f"Saved config snapshot to {config_snapshot_path}")
            
            # This is where you would run your multi-agent training loop
            logger.info("Would run multi-agent training here")
            logger.info(f"Environment: {env}")
            logger.info(f"Agents: {agents}")
            logger.info(f"Training for {config['training']['total_timesteps']} timesteps")
            
            # For this example, just log some dummy metrics
            for i in range(5):
                for agent_id, agent in agents.items():
                    mlflow_manager.log_metrics({
                        f"{agent_id}_reward": i * 10,
                        f"{agent_id}_loss": 1.0 / (i + 1)
                    }, step=i)
    
    else:
        logger.error(f"Unsupported environment type: {env_type}")
        sys.exit(1)
    
    logger.info("Training example complete")

if __name__ == "__main__":
    main() 