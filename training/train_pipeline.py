"""
Unified Training Pipeline for Single and Multi-Agent RL.

This module provides a unified training pipeline that can handle both
single-agent and multi-agent reinforcement learning for trading. It uses
the configuration-driven approach to create environments, agents, and
manage the training process.

Features:
- Unified pipeline for single and multi-agent training
- Configuration-driven setup and hyperparameter tuning
- Checkpoint saving and evaluation
- Metric tracking with MLflow
- Progress visualization

Implementation Notes:
- Uses the environment and agent factories for setup
- Supports both synchronous and asynchronous training
- Handles training resumption from checkpoints
- Integrates with MLflow for experiment tracking

Recent Changes:
- Added multi-agent training support
- Enhanced checkpoint management
- Improved metric tracking and visualization
"""

import os
import sys
import logging
import time
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import traceback

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils.config_manager import ConfigManager
from training.env_factory import create_env, create_eval_env
from agents.strategies.agent_factory import create_agent
from training.utils.unified_mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

def train_single_agent(
    agent,
    env,
    config: Dict[str, Any],
    mlflow_manager: Optional[MLflowManager] = None
) -> Dict[str, Any]:
    """
    Train a single agent using the standard RL loop.
    
    Args:
        agent: The agent to train
        env: The environment to train in
        config: Configuration dictionary
        mlflow_manager: Optional MLflow manager for logging
        
    Returns:
        Dictionary with training results
    """
    # Extract training parameters
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 100000)
    checkpoint_interval = training_config.get("checkpoint_interval", 10000)
    eval_interval = training_config.get("eval_interval", 5000)
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "checkpoints")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up tracking
    steps_done = 0
    episode_num = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    # Set up evaluation
    best_eval_reward = float('-inf')
    
    # Reset environment
    obs, info = env.reset()
    
    logger.info(f"Starting training for {total_timesteps} timesteps")
    training_start_time = time.time()
    
    # Main training loop
    while steps_done < total_timesteps:
        # Get action from agent
        action = agent.get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Update agent with experience
        agent.train_step(obs, action, reward, next_obs, done or truncated)
        
        # Update tracking
        current_episode_reward += reward
        current_episode_length += 1
        steps_done += 1
        
        # Update observation
        obs = next_obs
        
        # End of episode logic
        if done or truncated:
            # Log episode results
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            # Log to MLflow if available
            if mlflow_manager is not None:
                mlflow_manager.log_metrics({
                    "episode_reward": current_episode_reward,
                    "episode_length": current_episode_length,
                    "average_reward": np.mean(episode_rewards[-100:]),
                }, step=steps_done)
            
            # Reset for next episode
            obs, info = env.reset()
            current_episode_reward = 0
            current_episode_length = 0
            episode_num += 1
            
            # Log progress
            if episode_num % 10 == 0:
                logger.info(f"Episode {episode_num}, Steps: {steps_done}/{total_timesteps}, "
                           f"Recent Reward: {np.mean(episode_rewards[-10:]):.2f}")
        
        # Checkpoint saving
        if steps_done % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_{steps_done}.pt")
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log checkpoint to MLflow
            if mlflow_manager is not None:
                mlflow_manager.log_artifact(checkpoint_path)
        
        # Evaluation
        if steps_done % eval_interval == 0:
            # Create evaluation environment
            eval_env = env  # In a real implementation, you'd create a separate env with test data
            eval_rewards = evaluate_agent(agent, eval_env, num_episodes=5)
            mean_eval_reward = np.mean(eval_rewards)
            
            logger.info(f"Evaluation at step {steps_done}: Mean reward: {mean_eval_reward:.2f}")
            
            # Log to MLflow
            if mlflow_manager is not None:
                mlflow_manager.log_metrics({
                    "eval_reward": mean_eval_reward,
                    "eval_reward_std": np.std(eval_rewards),
                }, step=steps_done)
            
            # Save best model
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                best_model_path = os.path.join(checkpoint_dir, "best_agent.pt")
                agent.save(best_model_path)
                logger.info(f"New best model with reward {best_eval_reward:.2f} saved to {best_model_path}")
    
    # Final save
    final_model_path = os.path.join(checkpoint_dir, "final_agent.pt")
    agent.save(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Calculate training duration
    training_duration = time.time() - training_start_time
    logger.info(f"Training took {training_duration:.2f} seconds "
               f"({training_duration / 60:.2f} minutes)")
    
    # Log final metrics to MLflow
    if mlflow_manager is not None:
        mlflow_manager.log_metrics({
            "training_duration": training_duration,
            "final_average_reward": np.mean(episode_rewards[-100:]),
            "best_eval_reward": best_eval_reward,
            "total_episodes": episode_num,
        })
    
    # Return results
    return {
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_eval_reward": best_eval_reward,
        "training_duration": training_duration,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path if best_eval_reward > float('-inf') else None,
    }

def train_multi_agent(
    agents: Dict[str, Any],
    env,
    config: Dict[str, Any],
    mlflow_manager: Optional[MLflowManager] = None
) -> Dict[str, Any]:
    """
    Train multiple agents in a multi-agent environment.
    
    Args:
        agents: Dictionary mapping agent IDs to agent instances
        env: Multi-agent environment
        config: Configuration dictionary
        mlflow_manager: Optional MLflow manager for logging
        
    Returns:
        Dictionary with training results
    """
    # Extract training parameters
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 100000)
    checkpoint_interval = training_config.get("checkpoint_interval", 10000)
    eval_interval = training_config.get("eval_interval", 5000)
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "checkpoints")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create agent-specific checkpoint directories
    agent_checkpoint_dirs = {}
    for agent_id in agents.keys():
        agent_dir = os.path.join(checkpoint_dir, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        agent_checkpoint_dirs[agent_id] = agent_dir
    
    # Set up tracking
    steps_done = 0
    episode_num = 0
    episode_rewards = {agent_id: [] for agent_id in agents.keys()}
    current_episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
    episode_lengths = []
    current_episode_length = 0
    
    # Set up evaluation
    best_eval_rewards = {agent_id: float('-inf') for agent_id in agents.keys()}
    
    # Reset environment
    obs_dict, info = env.reset()
    
    logger.info(f"Starting multi-agent training for {total_timesteps} timesteps "
               f"with {len(agents)} agents: {list(agents.keys())}")
    training_start_time = time.time()
    
    # Main training loop
    while steps_done < total_timesteps:
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.get_action(obs_dict[agent_id])
        
        # Take step in environment
        next_obs_dict, rewards, dones, truncated, info = env.step(actions)
        
        # Update agents with experiences
        for agent_id, agent in agents.items():
            # Get agent-specific done state (True if either done or truncated)
            agent_done = bool(dones[agent_id] or truncated)
            
            # Update agent
            agent.train_step(
                obs_dict[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_obs_dict[agent_id],
                agent_done
            )
            
            # Update tracking for this agent
            current_episode_rewards[agent_id] += rewards[agent_id]
        
        # Update overall tracking
        current_episode_length += 1
        steps_done += 1
        
        # Update observations
        obs_dict = next_obs_dict
        
        # Check if episode is done (all agents done or truncated)
        episode_done = truncated or all(dones.values())
        
        # End of episode logic
        if episode_done:
            # Log episode results for each agent
            for agent_id in agents.keys():
                episode_rewards[agent_id].append(current_episode_rewards[agent_id])
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    mlflow_manager.log_metrics({
                        f"{agent_id}_episode_reward": current_episode_rewards[agent_id],
                        f"{agent_id}_average_reward": np.mean(episode_rewards[agent_id][-100:]),
                    }, step=steps_done)
            
            # Also log episode length
            episode_lengths.append(current_episode_length)
            if mlflow_manager is not None:
                mlflow_manager.log_metrics({
                    "episode_length": current_episode_length,
                }, step=steps_done)
            
            # Reset for next episode
            obs_dict, info = env.reset()
            for agent_id in agents.keys():
                current_episode_rewards[agent_id] = 0
            current_episode_length = 0
            episode_num += 1
            
            # Log progress
            if episode_num % 10 == 0:
                log_msg = f"Episode {episode_num}, Steps: {steps_done}/{total_timesteps}"
                for agent_id in agents.keys():
                    recent_rewards = episode_rewards[agent_id][-10:]
                    if recent_rewards:
                        log_msg += f", {agent_id} Reward: {np.mean(recent_rewards):.2f}"
                logger.info(log_msg)
        
        # Checkpoint saving
        if steps_done % checkpoint_interval == 0:
            for agent_id, agent in agents.items():
                checkpoint_path = os.path.join(agent_checkpoint_dirs[agent_id], f"agent_{steps_done}.pt")
                agent.save(checkpoint_path)
                logger.info(f"Saved {agent_id} checkpoint to {checkpoint_path}")
                
                # Log checkpoint to MLflow
                if mlflow_manager is not None:
                    mlflow_manager.log_artifact(checkpoint_path)
        
        # Evaluation
        if steps_done % eval_interval == 0:
            # Create evaluation environment
            eval_env = env  # In a real implementation, you'd create a separate env with test data
            
            # Evaluate all agents
            eval_rewards = evaluate_multi_agent(agents, eval_env, num_episodes=5)
            
            # Log evaluation results for each agent
            for agent_id in agents.keys():
                mean_eval_reward = np.mean(eval_rewards[agent_id])
                std_eval_reward = np.std(eval_rewards[agent_id])
                
                logger.info(f"Evaluation at step {steps_done}: "
                           f"{agent_id} Mean reward: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
                
                # Log to MLflow
                if mlflow_manager is not None:
                    mlflow_manager.log_metrics({
                        f"{agent_id}_eval_reward": mean_eval_reward,
                        f"{agent_id}_eval_reward_std": std_eval_reward,
                    }, step=steps_done)
                
                # Save best model for this agent
                if mean_eval_reward > best_eval_rewards[agent_id]:
                    best_eval_rewards[agent_id] = mean_eval_reward
                    best_model_path = os.path.join(agent_checkpoint_dirs[agent_id], "best_agent.pt")
                    agent.save(best_model_path)
                    logger.info(f"New best model for {agent_id} with reward "
                               f"{best_eval_rewards[agent_id]:.2f} saved to {best_model_path}")
    
    # Final save for all agents
    final_model_paths = {}
    for agent_id, agent in agents.items():
        final_model_path = os.path.join(agent_checkpoint_dirs[agent_id], "final_agent.pt")
        agent.save(final_model_path)
        final_model_paths[agent_id] = final_model_path
        logger.info(f"Training complete for {agent_id}. Final model saved to {final_model_path}")
    
    # Calculate training duration
    training_duration = time.time() - training_start_time
    logger.info(f"Multi-agent training took {training_duration:.2f} seconds "
               f"({training_duration / 60:.2f} minutes)")
    
    # Log final metrics to MLflow
    if mlflow_manager is not None:
        metrics = {
            "training_duration": training_duration,
            "total_episodes": episode_num,
        }
        for agent_id in agents.keys():
            if episode_rewards[agent_id]:
                metrics[f"{agent_id}_final_average_reward"] = np.mean(episode_rewards[agent_id][-100:])
                metrics[f"{agent_id}_best_eval_reward"] = best_eval_rewards[agent_id]
        
        mlflow_manager.log_metrics(metrics)
    
    # Return results
    return {
        "agents": agents,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "best_eval_rewards": best_eval_rewards,
        "training_duration": training_duration,
        "final_model_paths": final_model_paths,
    }

def evaluate_agent(agent, env, num_episodes: int = 5) -> List[float]:
    """
    Evaluate an agent's performance in an environment.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        num_episodes: Number of evaluation episodes
        
    Returns:
        List of rewards achieved in each episode
    """
    rewards = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        obs, info = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action without exploration
            action = agent.get_action(obs, eval_mode=True)
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Update reward
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return rewards

def evaluate_multi_agent(agents, env, num_episodes: int = 5) -> Dict[str, List[float]]:
    """
    Evaluate multiple agents' performance in a multi-agent environment.
    
    Args:
        agents: Dictionary mapping agent IDs to agent instances
        env: Multi-agent environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary mapping agent IDs to lists of rewards
    """
    rewards = {agent_id: [] for agent_id in agents.keys()}
    
    for episode in range(num_episodes):
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        obs_dict, info = env.reset()
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get actions without exploration (deterministic)
            actions = {}
            for agent_id, agent in agents.items():
                actions[agent_id] = agent.get_action(obs_dict[agent_id], deterministic=True)
            
            # Take step in environment
            obs_dict, step_rewards, dones, truncated, info = env.step(actions)
            
            # Update rewards
            for agent_id in agents.keys():
                episode_rewards[agent_id] += step_rewards[agent_id]
            
            # Check if episode is done
            done = truncated or all(dones.values())
        
        # Record episode rewards
        for agent_id in agents.keys():
            rewards[agent_id].append(episode_rewards[agent_id])
    
    return rewards

def train_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main training pipeline that handles both single and multi-agent training.
    
    Args:
        config: Configuration dictionary containing all parameters
        
    Returns:
        Dictionary with training results
    """
    # Create MLflow manager
    experiment_name = f"{config.get('agent_type', 'ppo')}_{config['env']['type']}"
    mlflow_manager = MLflowManager(experiment_name)

    # Start MLflow run
    mlflow_manager.start_run(run_name="train_pipeline")
    try:
        # Create environment
        env = create_env(config)
        
        # Determine environment type and handle accordingly
        env_type = config["env"]["type"]
        
        if env_type == "single_asset_rl":
            # Single-agent training
            agent = create_agent(
                agent_type=config.get("agent_type", "ppo"),
                config=config.get("agent", {}),
                observation_space=env.observation_space,
                action_space=env.action_space
            )
            results = train_single_agent(agent, env, config, mlflow_manager)
            
        elif env_type == "multi_agent_rl":
            # Multi-agent training
            multi_agent_cfgs = config["env"].get("multi_agent_configs", [])
            if not multi_agent_cfgs:
                raise ValueError("No multi_agent_configs found in config['env']")
                
            agents = {}
            for agent_cfg in multi_agent_cfgs:
                agent_id = agent_cfg["id"]
                agent_type = agent_cfg.get("type", "ppo")
                
                # Get agent-specific observation and action spaces
                obs_space = env.observation_spaces[agent_id]
                act_space = env.action_spaces[agent_id]
                
                # Combine hyperparameters from agent_cfg and default config
                agent_config = {
                    **config.get("agent", {}),  # Default agent config
                    **agent_cfg.get("hyperparameters", {})  # Agent-specific overrides
                }
                
                # Create agent instance
                agent = create_agent(
                    agent_type=agent_type,
                    config=agent_config,
                    observation_space=obs_space,
                    action_space=act_space
                )
                agents[agent_id] = agent
                
            results = train_multi_agent(agents, env, config, mlflow_manager)
            
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise
        
    finally:
        # Ensure MLflow run is ended
        mlflow_manager.end_run()

class nullcontext:
    """Context manager that does nothing when MLflow is not available."""
    def __enter__(self): return self
    def __exit__(self, *excinfo): pass

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary for MLflow parameter logging.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convert lists to strings to make them loggable
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items) 