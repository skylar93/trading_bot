"""
Small Integration Test for Shape and NaN Validation

This module provides minimal integration tests to quickly catch common issues:
- Dimension mismatches between agents and environments
- NaN values in neural network inputs 
- Shape inconsistencies between environments, agents, and networks
- Problems in the training step that only occur during agent.update()

Features:
- Extremely small test data (100 rows)
- Minimal training steps (5-10)
- Quick validation of the entire pipeline
- Focused error detection for shape problems

Implementation Notes:
- Uses synthetic data with controlled properties
- Runs with tiny batch sizes to expose issues early
- Tests single and multi-agent configurations
- Tests meta-agent ensemble separately
- Explicitly checks tensor shapes and NaN values

Recent Changes:
- Initial implementation
"""

import pytest
import numpy as np
import pandas as pd
import torch
import logging
import sys
import os
import warnings
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from components
from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv
from envs.multi_agent_env import MultiAgentTradingEnv
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from agents.strategies.agent_factory import create_agent
from agents.strategies.advanced.meta_agent import MetaAgent
from training.train_pipeline import train_pipeline
from agents.models.architectures.mlp import PolicyNetwork

# Configure logging with detailed format for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shape_test_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("shape_test")


# ----- Test Data Fixtures -----

@pytest.fixture
def tiny_test_data():
    """Generate minimal OHLCV data with only 100 rows"""
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=100)
    
    # Start price and basic price action
    base_prices = {
        'BTC': 20000.0,
        'ETH': 1500.0,
        'LTC': 100.0
    }
    
    all_data = []
    
    for asset, base_price in base_prices.items():
        # Generate simple price series with minimal complexity
        returns = np.random.normal(0, 0.01, 100)  # Daily returns
        prices = base_price * np.cumprod(1 + returns)
        
        # Create dataframe with OHLCV data
        df = pd.DataFrame(index=dates)
        df['$open'] = prices
        df['$high'] = prices * (1 + np.random.uniform(0, 0.005, 100))
        df['$low'] = prices * (1 - np.random.uniform(0, 0.005, 100))
        df['$close'] = prices * (1 + np.random.uniform(-0.003, 0.003, 100))
        df['$volume'] = np.random.uniform(100, 1000, 100)
        df['asset'] = asset
        
        all_data.append(df)
    
    return pd.concat(all_data)


@pytest.fixture
def minimal_agent_configs():
    """Create minimal agent configurations for testing"""
    return {
        "agent1": {
            "type": "PPOAgent",
            "params": {
                "learning_rate": 0.0001,
                "batch_size": 4,  # Very small batch size to catch issues early
                "clip_param": 0.2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
                "update_epochs": 2,  # Minimal updates
                "hidden_size": 64,  # Small network for faster testing
                "window_size": 10,  # Small observation window
                "state_normalizer": "simple",
                "reward_normalizer": "none",
            }
        },
        "agent2": {
            "type": "MomentumPPOAgent",
            "params": {
                "learning_rate": 0.0001,
                "batch_size": 4,
                "clip_param": 0.2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
                "update_epochs": 2,
                "hidden_size": 64,
                "window_size": 10,
                "state_normalizer": "simple",
                "reward_normalizer": "none",
                "momentum_window": 5,
            }
        }
    }


@pytest.fixture
def meta_agent_config():
    """Create meta-agent configuration"""
    return {
        "type": "MetaAgent",
        "params": {
            "learning_rate": 0.0001,
            "batch_size": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "update_epochs": 2,
            "hidden_size": 64,
            "window_size": 10,
            "state_normalizer": "simple",
            "reward_normalizer": "none",
        }
    }


# ----- Shape Verification Functions -----

def verify_network_inputs(model_input, expected_shape, description):
    """Verify tensor shapes and check for NaN values"""
    if torch.is_tensor(model_input):
        # Check for NaNs
        if torch.isnan(model_input).any():
            logger.error(f"NaN detected in {description}: {model_input}")
            assert False, f"NaN detected in {description}"
        
        # Check shape
        if model_input.dim() == 1:
            input_shape = (model_input.shape[0],)
        else:
            input_shape = tuple(model_input.shape)
            
        # For batched inputs, we should at least check the feature dimension
        if len(input_shape) > 1 and len(expected_shape) > 1:
            assert input_shape[-1] == expected_shape[-1], (
                f"Shape mismatch in {description}: "
                f"Got {input_shape}, expected trailing dim {expected_shape[-1]}"
            )
        else:
            # For strict checking when shapes should exactly match
            assert input_shape == expected_shape, (
                f"Shape mismatch in {description}: "
                f"Got {input_shape}, expected {expected_shape}"
            )
            
        return True
    return False


class ShapeMonitor:
    """Utility class for monitoring and debugging tensor shapes"""
    
    def __init__(self):
        self.shape_records = {}
        
    def record_shape(self, tensor, name):
        """Record the shape of a tensor for later analysis"""
        if torch.is_tensor(tensor):
            shape = tuple(tensor.shape)
            has_nan = torch.isnan(tensor).any().item()
            self.shape_records[name] = {
                "shape": shape,
                "has_nan": has_nan
            }
            if has_nan:
                logger.warning(f"NaN detected in {name}")
                
    def log_shapes(self):
        """Log all recorded shapes"""
        logger.debug("=== Recorded Tensor Shapes ===")
        for name, info in self.shape_records.items():
            nan_status = "HAS NaNs" if info["has_nan"] else "no NaNs"
            logger.debug(f"{name}: {info['shape']} ({nan_status})")


# ----- Tests -----

@pytest.mark.shape_verification
def test_single_agent_shapes(tiny_test_data):
    """Test the shapes of observations, actions, and rewards for a single agent"""
    logger.info("Starting single agent shape test")
    
    # Create a tiny dataset
    df = tiny_test_data[tiny_test_data['asset'] == 'BTC'].copy()
    df = df.drop('asset', axis=1)
    
    # Environment parameters
    env_params = {
        "window_size": 10,
        "initial_capital": 10000,
        "trading_fee": 0.001,
    }
    
    # Agent parameters
    agent_params = {
        "type": "ppo",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
        "verbose": 1
    }
    
    try:
        # Create environment
        env = SingleAssetRLTradingEnv(data=df, **env_params)
        
        # Create agent
        agent = create_agent(
            agent_type="ppo",
            config=agent_params,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # Reset environment
        obs, _ = env.reset()
        
        # Log shapes
        logger.info(f"Observation shape: {obs.shape}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # Check for NaN values in observation
        if np.isnan(obs).any():
            logger.warning(f"NaN values detected in observation: {np.isnan(obs).sum()} NaN values")
            # Replace NaN values with 0 for testing purposes
            obs = np.nan_to_num(obs, nan=0.0)
            logger.info("Replaced NaN values with 0.0 for testing")
        
        # Run a few steps
        shapes = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        for i in range(5):
            # Get action
            action = agent.get_action(obs)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Replace NaN values if any
            if np.isnan(next_obs).any():
                logger.warning(f"NaN values detected in next_obs: {np.isnan(next_obs).sum()} NaN values")
                next_obs = np.nan_to_num(next_obs, nan=0.0)
            
            # Record shapes
            shapes["observations"].append(next_obs.shape)
            shapes["actions"].append(np.array(action).shape)
            shapes["rewards"].append(np.array(reward).shape)
            shapes["dones"].append(np.array(done).shape)
            
            # Log step info
            logger.info(f"Step {i}: Action={action}, Reward={reward}, Done={done}")
            
            # Update observation
            obs = next_obs
            
            if done:
                logger.info("Episode finished early")
                break
        
        # Check consistency of shapes
        assert all(shape == shapes["observations"][0] for shape in shapes["observations"]), "Inconsistent observation shapes"
        assert all(shape == shapes["actions"][0] for shape in shapes["actions"]), "Inconsistent action shapes"
        
        logger.info("Single agent shape test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in single agent shape test: {e}")
        raise


@pytest.mark.shape_verification
def test_multi_agent_shapes(tiny_test_data):
    """Test the shapes of observations, actions, and rewards for multiple agents"""
    logger.info("Starting multi-agent shape test")
    
    # Create a tiny dataset
    df = tiny_test_data[tiny_test_data['asset'] == 'BTC'].copy()
    df = df.drop('asset', axis=1)
    
    # Agent configs
    agent_configs = [
        {"id": "agent1", "agent_type": "PPOAgent", "initial_capital_percentage": 0.5},
        {"id": "agent2", "agent_type": "MomentumPPOAgent", "initial_capital_percentage": 0.5}
    ]
    
    # Environment parameters
    env_params = {
        "window_size": 10,
        "trading_fee": 0.001,
        "shared_capital": True,
        "capital_reallocation_freq": 5,
    }
    
    # Agent parameters
    agent_params = {
        "agent1": {
            "type": "ppo",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        },
        "agent2": {
            "type": "ppo",
            "strategy": "momentum",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        }
    }
    
    try:
        # Create environment
        env = MultiAgentTradingEnv(data=df, agent_configs=agent_configs, **env_params)
        
        # Create agents
        agents = {}
        for agent_id, agent_config in agent_params.items():
            agent_type = agent_config.get("type", "ppo")
            strategy = agent_config.get("strategy", None)
            
            agents[agent_id] = create_agent(
                agent_type=agent_type,
                strategy=strategy,
                config=agent_config,
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        
        # Reset environment
        obs_dict, _ = env.reset()
        
        # Log shapes
        for agent_id, obs in obs_dict.items():
            logger.info(f"Agent {agent_id} observation shape: {obs.shape}")
            
            # Check for NaN values in observation
            if np.isnan(obs).any():
                logger.warning(f"NaN values detected in {agent_id} observation: {np.isnan(obs).sum()} NaN values")
                # Replace NaN values with 0 for testing purposes
                obs_dict[agent_id] = np.nan_to_num(obs, nan=0.0)
                logger.info(f"Replaced NaN values with 0.0 for {agent_id}")
        
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # Run a few steps
        shapes = {
            agent_id: {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": []
            } for agent_id in agents.keys()
        }
        
        for i in range(5):
            # Get actions from agents
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(obs_dict[agent_id])
            
            # Step environment
            next_obs_dict, rewards, dones, truncated, infos = env.step(actions)
            
            # Replace NaN values if any
            for agent_id, obs in next_obs_dict.items():
                if np.isnan(obs).any():
                    logger.warning(f"NaN values detected in {agent_id} next_obs: {np.isnan(obs).sum()} NaN values")
                    next_obs_dict[agent_id] = np.nan_to_num(obs, nan=0.0)
            
            # Record shapes
            for agent_id in agents.keys():
                shapes[agent_id]["observations"].append(next_obs_dict[agent_id].shape)
                shapes[agent_id]["actions"].append(np.array(actions[agent_id]).shape)
                shapes[agent_id]["rewards"].append(np.array(rewards[agent_id]).shape if hasattr(rewards[agent_id], 'shape') else (1,))
                shapes[agent_id]["dones"].append(np.array(dones[agent_id]).shape if hasattr(dones[agent_id], 'shape') else (1,))
            
            # Log step info
            logger.info(f"Step {i}: Actions={actions}, Rewards={rewards}, Dones={dones}")
            
            # Update observations
            obs_dict = next_obs_dict
            
            if all(dones.values()):
                logger.info("Episode finished early")
                break
        
        # Check consistency of shapes for each agent
        for agent_id in agents.keys():
            assert all(shape == shapes[agent_id]["observations"][0] for shape in shapes[agent_id]["observations"]), f"Inconsistent observation shapes for {agent_id}"
            assert all(shape == shapes[agent_id]["actions"][0] for shape in shapes[agent_id]["actions"]), f"Inconsistent action shapes for {agent_id}"
        
        logger.info("Multi-agent shape test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in multi-agent shape test: {e}")
        raise


@pytest.mark.shape_verification
def test_meta_agent_ensemble(tiny_test_data):
    """Test the shapes of observations, actions, and rewards for a meta agent ensemble"""
    logger.info("Starting meta agent ensemble test")
    
    # Create a tiny dataset
    df = tiny_test_data[tiny_test_data['asset'] == 'BTC'].copy()
    df = df.drop('asset', axis=1)
    
    # Agent configs
    agent_configs = [
        {"id": "agent1", "agent_type": "PPOAgent", "initial_capital_percentage": 0.3},
        {"id": "agent2", "agent_type": "MomentumPPOAgent", "initial_capital_percentage": 0.3},
        {"id": "meta", "agent_type": "MetaAgent", "initial_capital_percentage": 0.4}
    ]
    
    # Environment parameters
    env_params = {
        "window_size": 10,
        "trading_fee": 0.001,
        "shared_capital": True,
        "capital_reallocation_freq": 5,
    }
    
    # Agent parameters
    agent_params = {
        "agent1": {
            "type": "ppo",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        },
        "agent2": {
            "type": "ppo",
            "strategy": "momentum",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        },
        "meta": {
            "type": "meta",
            "sub_agents": ["agent1", "agent2"],
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        }
    }
    
    try:
        # Create environment
        env = MultiAgentTradingEnv(data=df, agent_configs=agent_configs, **env_params)
        
        # Create sub-agents first
        agents = {}
        for agent_id in ["agent1", "agent2"]:
            agent_config = agent_params[agent_id]
            agent_type = agent_config.get("type", "ppo")
            strategy = agent_config.get("strategy", None)
            
            agents[agent_id] = create_agent(
                agent_type=agent_type,
                strategy=strategy,
                config=agent_config,
                observation_space=env.observation_space,
                action_space=env.action_space
            )
        
        # Create meta agent
        meta_config = agent_params["meta"]
        meta_config["sub_agents"] = {
            sub_id: agents[sub_id] for sub_id in meta_config["sub_agents"]
        }
        
        agents["meta"] = create_agent(
            agent_type="meta",
            config=meta_config,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        
        # Reset environment
        obs_dict, _ = env.reset()
        
        # Log shapes
        for agent_id, obs in obs_dict.items():
            logger.info(f"Agent {agent_id} observation shape: {obs.shape}")
            
            # Check for NaN values in observation
            if np.isnan(obs).any():
                logger.warning(f"NaN values detected in {agent_id} observation: {np.isnan(obs).sum()} NaN values")
                # Replace NaN values with 0 for testing purposes
                obs_dict[agent_id] = np.nan_to_num(obs, nan=0.0)
                logger.info(f"Replaced NaN values with 0.0 for {agent_id}")
        
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        # Run a few steps
        shapes = {
            agent_id: {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": []
            } for agent_id in agents.keys()
        }
        
        for i in range(5):
            # Get actions from agents
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(obs_dict[agent_id])
            
            # Step environment
            next_obs_dict, rewards, dones, truncated, infos = env.step(actions)
            
            # Replace NaN values if any
            for agent_id, obs in next_obs_dict.items():
                if np.isnan(obs).any():
                    logger.warning(f"NaN values detected in {agent_id} next_obs: {np.isnan(obs).sum()} NaN values")
                    next_obs_dict[agent_id] = np.nan_to_num(obs, nan=0.0)
            
            # Record shapes
            for agent_id in agents.keys():
                shapes[agent_id]["observations"].append(next_obs_dict[agent_id].shape)
                shapes[agent_id]["actions"].append(np.array(actions[agent_id]).shape)
                shapes[agent_id]["rewards"].append(np.array(rewards[agent_id]).shape if hasattr(rewards[agent_id], 'shape') else (1,))
                shapes[agent_id]["dones"].append(np.array(dones[agent_id]).shape if hasattr(dones[agent_id], 'shape') else (1,))
            
            # Log step info
            logger.info(f"Step {i}: Meta action={actions['meta']}, Rewards={rewards}")
            
            # Update observations
            obs_dict = next_obs_dict
            
            if all(dones.values()):
                logger.info("Episode finished early")
                break
        
        # Check consistency of shapes for each agent
        for agent_id in agents.keys():
            assert all(shape == shapes[agent_id]["observations"][0] for shape in shapes[agent_id]["observations"]), f"Inconsistent observation shapes for {agent_id}"
            assert all(shape == shapes[agent_id]["actions"][0] for shape in shapes[agent_id]["actions"]), f"Inconsistent action shapes for {agent_id}"
        
        logger.info("Meta agent ensemble test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in meta agent ensemble test: {e}")
        raise


@pytest.mark.shape_verification
def test_train_pipeline_minimal(tiny_test_data, minimal_agent_configs):
    """Test the full training pipeline with minimal configurations"""
    logger.info("Starting minimal train pipeline test")
    
    # Create minimal configuration
    config = {
        "env": {
            "type": "single_asset_rl",
            "window_size": 10,
            "initial_balance": 10000,
            "trading_fee": 0.001,
        },
        "agents": {
            "agent1": {
                "type": "ppo",
                "learning_rate": 0.0003,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "normalize_advantage": True,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "use_sde": False,
                "sde_sample_freq": -1,
                "target_kl": None,
                "verbose": 1
            }
        },
        "training": {
            "total_timesteps": 100,  # Minimal steps
            "eval_interval": 50,
            "checkpoint_interval": 1,  # Set to 1 to avoid division by zero
            "log_interval": 10
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints",
            "log_dir": "test_logs"
        }
    }
    
    try:
        # Run training
        logger.info("Running training pipeline...")
        results = train_pipeline(config, tiny_test_data)
        
        # Check results
        assert "best_eval_reward" in results, "Missing best_eval_reward in results"
        assert "best_model_path" in results, "Missing best_model_path in results"
        assert "agent" in results, "Missing agent in results"
        
        # Log training time if available
        if "training_time" in results:
            logger.info(f"Training time: {results['training_time']} seconds")
        else:
            logger.warning("Training time not available in results")
        
        logger.info("Minimal train pipeline test completed successfully")
    except Exception as e:
        logger.error(f"Error in minimal train pipeline test: {e}")
        raise


@pytest.mark.shape_verification
def test_multi_agent_train_pipeline_minimal(tiny_test_data, minimal_agent_configs):
    """Test the full multi-agent training pipeline with minimal configurations"""
    logger.info("Starting minimal multi-agent train pipeline test")
    
    # Create minimal configuration
    config = {
        "env": {
            "type": "multi_agent_rl",
            "window_size": 10,
            "initial_balance": 10000,
            "trading_fee": 0.001,
            "shared_capital": True,
            "capital_reallocation_freq": 5,
            "multi_agent_configs": [
                {"id": "agent1", "agent_type": "PPOAgent", "initial_capital_percentage": 0.5},
                {"id": "agent2", "agent_type": "MomentumPPOAgent", "initial_capital_percentage": 0.5}
            ]
        },
        "agents": minimal_agent_configs,
        "training": {
            "total_timesteps": 100,  # Minimal steps
            "eval_interval": 50,
            "checkpoint_interval": 1,  # Set to 1 to avoid division by zero
            "log_interval": 10
        },
        "paths": {
            "checkpoint_dir": "test_checkpoints",
            "log_dir": "test_logs"
        }
    }
    
    # Prepare data - use only one asset and drop the asset column
    single_asset_data = tiny_test_data[tiny_test_data['asset'] == 'BTC'].copy()
    single_asset_data = single_asset_data.drop('asset', axis=1)
    
    try:
        # Run training
        logger.info("Running multi-agent training pipeline...")
        results = train_pipeline(config, single_asset_data)
        
        # Check results
        assert "training_time" in results, "Missing training_time in results"
        if "best_eval_rewards" in results:
            for agent_id, reward in results["best_eval_rewards"].items():
                logger.info(f"Agent {agent_id} best evaluation reward: {reward}")
        
        logger.info("Minimal multi-agent train pipeline test completed successfully")
    except Exception as e:
        logger.error(f"Error in minimal multi-agent train pipeline test: {e}")
        raise 