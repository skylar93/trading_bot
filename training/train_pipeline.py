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
                    "episode_reward": float(current_episode_reward),
                    "episode_length": float(current_episode_length),
                    "average_reward": float(np.mean(episode_rewards[-100:])),
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
    
    Features:
    - Handles agents with different strategies and configurations
    - Supports independent agent rewards, observations, and done states
    - Properly manages episode termination and environment reset
    - Tracks and logs metrics per agent
    - Periodic evaluation and checkpoint saving
    
    Implementation Notes:
    - Follows the standard RL loop adapted for multi-agent scenarios
    - Each agent receives its own observations and calculates its own actions
    - Agents train independently on their own experiences
    - Supports both shared and independent capital modes via the environment
    
    Recent Changes:
    - Improved handling of agent-specific done states
    - Enhanced checkpoint management with agent-specific directories
    - Added support for deterministic evaluation
    - Optimized training loop to match the required structure
    - Fixed shared buffer format to be compatible with agent implementations
    
    Args:
        agents: Dictionary mapping agent_id to agent instances
        env: Multi-agent environment (must support dict-based interface)
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
    log_interval = training_config.get("log_interval", 10)
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
    
    # Check if shared experience buffer is enabled
    shared_experience_config = config.get("shared_experience", {})
    use_shared_buffer = shared_experience_config.get("enabled", False)
    shared_buffer = [] if use_shared_buffer else None
    max_buffer_size = shared_experience_config.get("buffer_size", 10000)
    
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
        
        # Take step in environment - agent_id keyed dictionaries
        next_obs_dict, rewards, dones, truncated, info = env.step(actions)
        
        # Episode termination logic - either all agents are done or environment truncated
        episode_done = all(dones.values()) or truncated
        
        # Update agents with experiences
        for agent_id, agent in agents.items():
            # Get agent-specific done flag - either this agent is done or the episode is truncated
            agent_done = bool(dones[agent_id] or truncated)
            
            # Update agent with its own experience
            agent.train_step(
                obs_dict[agent_id],                   # Current observation
                actions[agent_id],                    # Action taken
                rewards[agent_id],                    # Reward received
                next_obs_dict[agent_id],              # Next observation
                done=agent_done                       # Done flag for this agent
            )
            
            # Track reward for this agent
            current_episode_rewards[agent_id] += rewards[agent_id]
            
            # Add to shared experience buffer if enabled
            if use_shared_buffer:
                # Format as tuple to match agent expectations: (state, action, reward, next_state, done)
                experience = (
                    obs_dict[agent_id],              # state
                    actions[agent_id],               # action
                    rewards[agent_id],               # reward
                    next_obs_dict[agent_id],         # next_state
                    agent_done                       # done
                )
                shared_buffer.append(experience)
                
                # Trim buffer if it exceeds max size
                if len(shared_buffer) > max_buffer_size:
                    shared_buffer.pop(0)
        
        # Update tracking
        current_episode_length += 1
        steps_done += 1
        
        # Update observations for next step
        obs_dict = next_obs_dict
        
        # End of episode logic
        if episode_done:
            # Log episode results for each agent
            for agent_id in agents.keys():
                episode_rewards[agent_id].append(current_episode_rewards[agent_id])
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    # Ensure values are native Python float for MLflow
                    reward = float(current_episode_rewards[agent_id])
                    avg_reward = float(np.mean(episode_rewards[agent_id][-100:]) 
                                     if episode_rewards[agent_id] else 0.0)
                    mlflow_manager.log_metric(f"{agent_id}/episode_reward", reward, step=episode_num)
                    mlflow_manager.log_metric(f"{agent_id}/avg_reward_100", avg_reward, step=episode_num)
            
            # Add episode length to tracking
            episode_lengths.append(current_episode_length)
            
            # Reset environment and trackers
            obs_dict, info = env.reset()
            current_episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
            current_episode_length = 0
            episode_num += 1
            
            # Log episode stats
            if episode_num % log_interval == 0:
                episode_avg_rewards = {
                    agent_id: np.mean(episode_rewards[agent_id][-log_interval:])
                    for agent_id in agents.keys()
                }
                
                logger.info(
                    f"Episode {episode_num} | "
                    f"Steps: {steps_done} | "
                    f"Avg rewards: {episode_avg_rewards} | "
                    f"Time elapsed: {time.time() - training_start_time:.2f}s"
                )
        
        # Periodically share experiences between agents if enabled
        if use_shared_buffer and len(shared_buffer) > 0 and steps_done % 10 == 0:
            for agent_id, agent in agents.items():
                if hasattr(agent, "learn_from_shared_experience"):
                    agent.learn_from_shared_experience(shared_buffer)
        
        # Periodic evaluation
        if steps_done % eval_interval == 0:
            eval_rewards = evaluate_multi_agent(agents, env)
            
            # Log evaluation results
            for agent_id, rewards in eval_rewards.items():
                avg_eval_reward = np.mean(rewards)
                
                logger.info(
                    f"Agent {agent_id} | "
                    f"Step {steps_done} | "
                    f"Evaluation avg reward: {avg_eval_reward:.4f}"
                )
                
                # Save best model
                if avg_eval_reward > best_eval_rewards[agent_id]:
                    best_eval_rewards[agent_id] = avg_eval_reward
                    best_model_path = os.path.join(agent_checkpoint_dirs[agent_id], f"best_model.pt")
                    if hasattr(agents[agent_id], "save"):
                        agents[agent_id].save(best_model_path)
                        logger.info(f"Saved best model for agent {agent_id} with reward {avg_eval_reward:.4f}")
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    for i, reward in enumerate(rewards):
                        mlflow_manager.log_metric(f"{agent_id}/eval_episode_{i+1}_reward", reward, step=steps_done)
                    mlflow_manager.log_metric(f"{agent_id}/eval_avg_reward", avg_eval_reward, step=steps_done)
        
        # Checkpoint saving
        if steps_done % checkpoint_interval == 0:
            for agent_id, agent in agents.items():
                if hasattr(agent, "save"):
                    checkpoint_path = os.path.join(
                        agent_checkpoint_dirs[agent_id], 
                        f"checkpoint_{steps_done}.pt"
                    )
                    agent.save(checkpoint_path)
            
            logger.info(f"Saved checkpoints at step {steps_done}")
    
    # Final evaluation
    final_eval_rewards = evaluate_multi_agent(agents, env, num_episodes=10)
    final_avg_rewards = {
        agent_id: np.mean(rewards) 
        for agent_id, rewards in final_eval_rewards.items()
    }
    
    logger.info(f"Training completed. Final evaluation rewards: {final_avg_rewards}")
    
    # Save final models
    final_model_paths = {}
    best_model_paths = {}
    for agent_id, agent in agents.items():
        if hasattr(agent, "save"):
            # Save final model
            final_model_path = os.path.join(agent_checkpoint_dirs[agent_id], "final_model.pt")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            agent.save(final_model_path)
            final_model_paths[agent_id] = final_model_path
            
            # Save path to best model (for compatibility with tests)
            best_model_path = os.path.join(agent_checkpoint_dirs[agent_id], "best_model.pt")
            best_model_paths[agent_id] = best_model_path if best_eval_rewards[agent_id] > float('-inf') else None
    
    # Compile and return results
    results = {
        "episode_rewards": episode_rewards,
        "best_eval_rewards": best_eval_rewards,
        "final_eval_rewards": final_avg_rewards,
        "final_model_paths": final_model_paths,
        "best_model_paths": best_model_paths,
        "training_time": time.time() - training_start_time,
        "episode_lengths": episode_lengths,
    }
    
    return results

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

def train_pipeline(config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Main training pipeline that handles both single and multi-agent training.
    
    Args:
        config: Configuration dictionary containing all parameters
        data: Optional DataFrame with OHLCV data. If provided, will be used instead of loading from config
        
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
        if data is not None:
            logger.info(f"Using provided data with shape {data.shape}")
            env = create_env(config, data)
        else:
            env = create_env(config)
        
        # Determine environment type and handle accordingly
        env_type = config["env"]["type"]
        
        if env_type == "single_asset_rl":
            # Single-agent training
            agent = create_agent(
                agent_type=config.get("agent_type", "ppo"),
                strategy=config.get("strategy", None),  # Add strategy parameter
                config=config.get("agent", {}),
                observation_space=env.observation_space,
                action_space=env.action_space
            )
            results = train_single_agent(agent, env, config, mlflow_manager)
            
        elif env_type == "multi_asset_rl":
            # Single agent handling multiple assets
            agent = create_agent(
                agent_type=config.get("agent_type", "ppo"),
                strategy=config.get("strategy", None),
                config=config.get("agent", {}),
                observation_space=env.observation_space,
                action_space=env.action_space
            )
            results = train_single_agent(agent, env, config, mlflow_manager)
            
        elif env_type in ["multi_agent_rl", "multi_asset_multi_agent_rl"]:
            # Get multi-agent configuration
            use_manager = config["env"].get("use_manager", False)  # Flag to use MultiAgentManager
            ensemble_method = config["env"].get("ensemble_method", "weighted")  # Default to weighted ensemble
            multi_agent_cfgs = config["env"].get("multi_agent_configs", [])
            
            if not multi_agent_cfgs:
                raise ValueError("No multi_agent_configs found in config['env']")
            
            # Check if we should use the MultiAgentManager for training
            if use_manager:
                logger.info(f"Using MultiAgentManager with ensemble method: {ensemble_method}")
                
                # Create meta-agent config if needed
                meta_config = None
                if ensemble_method == "meta":
                    meta_config = config["env"].get("meta_config", None)
                    if meta_config is None:
                        logger.info("No meta_config provided, will create default meta-agent configuration")
                
                # Log the number of agents being managed
                logger.info(f"Training with MultiAgentManager using {len(multi_agent_cfgs)} sub-agents")
                
                # Use the manager-based training function
                results = train_multi_agent_with_manager(
                    env=env,
                    agent_configs=multi_agent_cfgs,
                    meta_config=meta_config,
                    ensemble_method=ensemble_method,
                    config=config,
                    mlflow_manager=mlflow_manager
                )
                
                # Log that manager training completed
                logger.info("Multi-agent training with manager completed successfully")
                
            else:
                # Traditional approach: create agents individually and train
                logger.info("Using traditional multi-agent training (without manager)")
                
                agents = {}
                for agent_cfg in multi_agent_cfgs:
                    agent_id = agent_cfg["id"]
                    
                    # Get agent_type and strategy separately
                    agent_type = agent_cfg.get("agent_type", "ppo")  # Learning algorithm
                    strategy = agent_cfg.get("strategy", None)       # Trading strategy
                    
                    # Log agent configuration
                    logger.info(f"Configuring agent '{agent_id}' with agent_type='{agent_type}', strategy='{strategy}'")
                    
                    # Get agent-specific observation and action spaces
                    obs_space = env.observation_spaces[agent_id]
                    act_space = env.action_spaces[agent_id]
                    
                    # Combine hyperparameters from agent_cfg and default config
                    agent_config = {
                        **config.get("agent", {}),  # Default agent config
                        **agent_cfg.get("hyperparameters", {})  # Agent-specific overrides
                    }
                    
                    # Create agent instance with explicit strategy
                    agent = create_agent(
                        agent_type=agent_type,
                        strategy=strategy,  # Pass strategy separately
                        config=agent_config,
                        observation_space=obs_space,
                        action_space=act_space
                    )
                    
                    # Store agent in dictionary
                    agents[agent_id] = agent
                    
                    # Log successful agent creation
                    logger.info(f"Created {agent_type} agent with {strategy} strategy for agent_id '{agent_id}'")
                
                # Train all agents with the traditional approach
                results = train_multi_agent(agents, env, config, mlflow_manager)
        
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        # Log final metrics
        if mlflow_manager is not None:
            # Log top-level metrics based on results structure
            if "best_eval_rewards" in results:
                for agent_id, reward in results["best_eval_rewards"].items():
                    mlflow_manager.log_metric(f"best_eval_reward/{agent_id}", float(reward))
            
            # Log final metrics to MLflow
            mlflow_manager.log_metrics({
                "total_episodes": float(len(results.get("episode_lengths", []))),
                "training_duration_seconds": float(results.get("training_time", 0)),
            })
            
            # Log model paths as artifacts
            if "final_model_paths" in results:
                for agent_id, path in results["final_model_paths"].items():
                    if path and os.path.exists(path):
                        mlflow_manager.log_artifact(path)
            
            # End MLflow run
            mlflow_manager.end_run()
        
        return results
    
    except Exception as e:
        logger.exception(f"Error in train_pipeline: {e}")
        if mlflow_manager is not None:
            mlflow_manager.log_metric("training_error", 1.0)
            mlflow_manager.end_run()
        raise

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

def train_multi_agent_with_manager(
    env,
    agent_configs: List[Dict[str, Any]],
    meta_config: Optional[Dict[str, Any]] = None,
    ensemble_method: str = "meta",
    config: Dict[str, Any] = None,
    mlflow_manager: Optional[MLflowManager] = None
) -> Dict[str, Any]:
    """
    Train multiple agents using the MultiAgentManager for coordination.
    
    Features:
    - Integrates with MultiAgentManager for meta-agent and shared buffer functionality
    - Supports ensemble methods: weighted, best, meta
    - Handles coordinated decision making across agents
    - Enables experience sharing between agents
    - Manages meta-agent training if ensemble_method is "meta"
    
    Implementation Notes:
    - Uses the manager.act() method for coordinated action selection
    - Leverages manager.train_step() for integrated agent training
    - Automatically manages shared experience buffer
    - Supports both standard sub-agents and meta-agent training
    - Maintains compatibility with MLflow tracking
    
    Recent Changes:
    - Initial implementation integrating with MultiAgentManager
    - Added support for multiple ensemble methods
    - Integrated proper checkpointing for manager and agents
    - Enhanced experience sharing with meta-agent coordination
    - Added compatibility with both 4-value and 5-value returns from env.step()
    
    Args:
        env: Multi-agent environment
        agent_configs: List of sub-agent configurations
        meta_config: Optional meta-agent configuration (auto-created if None and ensemble_method is "meta")
        ensemble_method: Method for action selection ("weighted", "best", "meta")
        config: Configuration dictionary
        mlflow_manager: Optional MLflow manager for logging
        
    Returns:
        Dictionary with training results
    """
    from agents.strategies.multi.multi_agent_manager import MultiAgentManager
    
    # Default config if not provided
    if config is None:
        config = {
            "training": {
                "total_timesteps": 100000,
                "checkpoint_interval": 10000,
                "eval_interval": 5000,
                "log_interval": 10
            },
            "paths": {
                "checkpoint_dir": "checkpoints"
            }
        }
    
    # Extract training parameters
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 100000)
    checkpoint_interval = training_config.get("checkpoint_interval", 10000)
    eval_interval = training_config.get("eval_interval", 5000)
    log_interval = training_config.get("log_interval", 10)
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "checkpoints")
    manager_checkpoint_dir = os.path.join(checkpoint_dir, "manager")
    
    # Ensure checkpoint directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(manager_checkpoint_dir, exist_ok=True)
    
    # If ensemble method is "meta" and no meta_config provided, create a default one
    if ensemble_method == "meta" and meta_config is None:
        # Estimate observation size as sum of all agent observation spaces
        obs_size_estimate = sum(cfg.get("observation_size", env.window_size * 5) for cfg in agent_configs)
        meta_config = {
            "id": "meta_agent",
            "type": "meta",
            "model": "ppo",
            "observation_size": obs_size_estimate,
            "action_dim": len(agent_configs),
            "learning_rate": 3e-4,
            "hidden_dim": 128,
            "continuous_ensemble": True  # Use continuous weights instead of discrete selection
        }
    
    # Add meta_config to agent_configs if using meta ensemble
    all_configs = agent_configs.copy()
    if ensemble_method == "meta" and meta_config is not None:
        all_configs.append(meta_config)
    
    # Create MultiAgentManager
    logger.info(f"Creating MultiAgentManager with ensemble method: {ensemble_method}")
    manager = MultiAgentManager(
        agent_configs=all_configs,
        ensemble_method=ensemble_method,
        min_share_reward=0.2  # Minimum reward threshold for sharing experiences
    )
    
    # Set up tracking
    steps_done = 0
    episode_num = 0
    episode_rewards = {agent_id: [] for agent_id in env.agents}
    current_episode_rewards = {agent_id: 0 for agent_id in env.agents}
    episode_lengths = []
    current_episode_length = 0
    
    # Set up evaluation
    best_eval_rewards = {agent_id: float('-inf') for agent_id in env.agents}
    
    # Reset environment
    obs_dict, info = env.reset()
    
    logger.info(f"Starting multi-agent training with manager for {total_timesteps} timesteps")
    training_start_time = time.time()
    
    # Main training loop
    while steps_done < total_timesteps:
        # Get actions from manager - handles ensemble logic internally
        actions = manager.act(obs_dict, deterministic=False)
        
        # Take step in environment - handle both 4-value and 5-value returns
        step_result = env.step(actions)
        
        # Check if step_result has 4 or 5 elements (handle different Gym versions)
        if len(step_result) == 5:
            next_obs_dict, rewards, dones, truncated, info = step_result
            # Episode termination logic
            episode_done = all(dones.values()) or truncated
        else:  # Assume 4 elements (older Gym version)
            next_obs_dict, rewards, dones, info = step_result
            truncated = False  # Set default value
            # Episode termination logic
            episode_done = all(dones.values())
        
        # Create experiences dictionary for manager
        experiences = {}
        for agent_id in env.agents:
            # Get agent-specific done flag - either this agent is done or episode is truncated
            agent_done = bool(dones[agent_id] or truncated)
            
            experiences[agent_id] = {
                "observation": obs_dict[agent_id],
                "action": actions[agent_id],
                "reward": rewards[agent_id],
                "next_observation": next_obs_dict[agent_id],
                "done": agent_done,
                "info": info.get(agent_id, {})
            }
            
            # Track reward for this agent
            current_episode_rewards[agent_id] += rewards[agent_id]
        
        # Train agents through manager - handles meta-agent and shared experiences
        metrics = manager.train_step(experiences)
        
        # Update tracking
        current_episode_length += 1
        steps_done += 1
        
        # Update observations for next step
        obs_dict = next_obs_dict
        
        # End of episode logic
        if episode_done:
            # Log episode results for each agent
            for agent_id in env.agents:
                episode_rewards[agent_id].append(current_episode_rewards[agent_id])
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    reward = float(current_episode_rewards[agent_id])
                    avg_reward = float(np.mean(episode_rewards[agent_id][-100:]) 
                                    if episode_rewards[agent_id] else 0.0)
                    mlflow_manager.log_metric(f"{agent_id}/episode_reward", reward, step=episode_num)
                    mlflow_manager.log_metric(f"{agent_id}/avg_reward_100", avg_reward, step=episode_num)
            
            # Add episode length to tracking
            episode_lengths.append(current_episode_length)
            
            # Reset environment and trackers
            obs_dict, info = env.reset()
            current_episode_rewards = {agent_id: 0 for agent_id in env.agents}
            current_episode_length = 0
            episode_num += 1
            
            # Log episode stats
            if episode_num % log_interval == 0:
                episode_avg_rewards = {
                    agent_id: np.mean(episode_rewards[agent_id][-log_interval:])
                    for agent_id in env.agents
                }
                
                logger.info(
                    f"Episode {episode_num} | "
                    f"Steps: {steps_done} | "
                    f"Avg rewards: {episode_avg_rewards} | "
                    f"Time elapsed: {time.time() - training_start_time:.2f}s"
                )
        
        # Periodic evaluation
        if steps_done % eval_interval == 0:
            # Evaluate using manager for proper ensemble coordination
            eval_returns = evaluate_with_manager(env, manager, num_episodes=5)
            
            # Log evaluation results
            for agent_id, returns in eval_returns.items():
                avg_eval_return = np.mean(returns)
                
                logger.info(
                    f"Agent {agent_id} | "
                    f"Step {steps_done} | "
                    f"Evaluation avg return: {avg_eval_return:.4f}"
                )
                
                # Save best model for individual agents
                if avg_eval_return > best_eval_rewards[agent_id]:
                    best_eval_rewards[agent_id] = avg_eval_return
                    
                    # Save individual agent if it's in the manager
                    if agent_id in manager.agents:
                        best_model_path = os.path.join(checkpoint_dir, agent_id, "best_model.pt")
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        if hasattr(manager.agents[agent_id], "save"):
                            manager.agents[agent_id].save(best_model_path)
                            logger.info(f"Saved best model for agent {agent_id} with return {avg_eval_return:.4f}")
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    for i, reward in enumerate(returns):
                        mlflow_manager.log_metric(f"{agent_id}/eval_episode_{i+1}_return", reward, step=steps_done)
                    mlflow_manager.log_metric(f"{agent_id}/eval_avg_return", avg_eval_return, step=steps_done)
            
            # Save entire manager for ensemble functionality
            manager_checkpoint_path = os.path.join(manager_checkpoint_dir, f"manager_best.pt")
            manager.save(manager_checkpoint_path)
            logger.info(f"Saved best manager checkpoint at step {steps_done}")
        
        # Regular checkpoint saving
        if steps_done % checkpoint_interval == 0:
            # Save manager
            manager_checkpoint_path = os.path.join(manager_checkpoint_dir, f"manager_step_{steps_done}.pt")
            manager.save(manager_checkpoint_path)
            
            # Save individual agents
            for agent_id, agent in manager.agents.items():
                if hasattr(agent, "save"):
                    agent_checkpoint_path = os.path.join(checkpoint_dir, agent_id, f"checkpoint_{steps_done}.pt")
                    os.makedirs(os.path.dirname(agent_checkpoint_path), exist_ok=True)
                    agent.save(agent_checkpoint_path)
            
            logger.info(f"Saved checkpoints at step {steps_done}")
    
    # Final evaluation
    final_eval_returns = evaluate_with_manager(env, manager, num_episodes=10)
    final_avg_returns = {
        agent_id: np.mean(returns) 
        for agent_id, returns in final_eval_returns.items()
    }
    
    logger.info(f"Training completed. Final evaluation returns: {final_avg_returns}")
    
    # Save final models
    final_model_paths = {}
    best_model_paths = {}
    
    # Save manager to final path
    final_manager_path = os.path.join(manager_checkpoint_dir, "final_manager.pt")
    manager.save(final_manager_path)
    
    # Save individual agents and track paths for compatibility with existing tests
    for agent_id, agent in manager.agents.items():
        if hasattr(agent, "save"):
            # Save final model
            final_model_path = os.path.join(checkpoint_dir, agent_id, "final_model.pt")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            agent.save(final_model_path)
            final_model_paths[agent_id] = final_model_path
            
            # Save path to best model (for compatibility with tests)
            best_model_path = os.path.join(checkpoint_dir, agent_id, "best_model.pt")
            best_model_paths[agent_id] = best_model_path if best_eval_rewards[agent_id] > float('-inf') else None
    
    # Compile and return results
    results = {
        "episode_rewards": episode_rewards,
        "best_eval_rewards": best_eval_rewards,
        "final_eval_returns": final_avg_returns,
        "training_time": time.time() - training_start_time,
        "episode_lengths": episode_lengths,
        "manager": manager  # Include the manager in results
    }
    
    return results

def evaluate_with_manager(env, manager, num_episodes: int = 5) -> Dict[str, List[float]]:
    """
    Evaluate agents using MultiAgentManager with deterministic policy.
    
    Features:
    - Evaluates all agents coordinated by the manager
    - Uses deterministic policy for stable evaluation
    - Tracks returns for each agent
    - Handles proper environment reset and termination
    
    Implementation Notes:
    - Uses manager.act() with deterministic=True for action selection
    - Resets environment between episodes
    - Properly handles episode termination
    - Compatible with both 4-value and 5-value env.step() returns
    
    Recent Changes:
    - Initial implementation for manager-based evaluation
    - Added compatibility with both gym versions
    
    Args:
        env: Multi-agent environment
        manager: MultiAgentManager instance
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary mapping agent_id to list of episode returns
    """
    returns = {agent_id: [] for agent_id in env.agents}
    
    for episode in range(num_episodes):
        episode_returns = {agent_id: 0 for agent_id in env.agents}
        obs_dict, info = env.reset()
        
        done = False
        while not done:
            # Get actions deterministically
            actions = manager.act(obs_dict, deterministic=True)
            
            # Step environment - handle both gym versions
            step_result = env.step(actions)
            
            # Check if step_result has 4 or 5 elements
            if len(step_result) == 5:
                next_obs_dict, rewards, dones, truncated, info = step_result
                # Check if episode is done
                done = all(dones.values()) or truncated
            else:  # Assume 4 elements (older Gym version)
                next_obs_dict, rewards, dones, info = step_result
                # Check if episode is done
                done = all(dones.values())
            
            # Update returns
            for agent_id, reward in rewards.items():
                episode_returns[agent_id] += reward
            
            # Update observations
            obs_dict = next_obs_dict
        
        # Add episode returns for each agent
        for agent_id, return_ in episode_returns.items():
            returns[agent_id].append(return_)
    
    return returns 