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
import torch

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
    For SB3AgentWrapper, delegates to agent.train() (model.learn()).

    Args:
        agent: The agent to train
        env: The environment to train in
        config: Configuration dictionary
        mlflow_manager: Optional MLflow manager for logging

    Returns:
        Dictionary with training results
    """
    from agents.sb3.sb3_agent_wrapper import SB3AgentWrapper
    if isinstance(agent, SB3AgentWrapper):
        training_config = config.get("training", {})
        total_timesteps = training_config.get("total_timesteps", 100_000)
        agent.train(env, total_timesteps=total_timesteps)
        return {
            "episode_rewards": [0.0],
            "episode_lengths": [total_timesteps],
            "best_eval_reward": 0.0,
            "total_timesteps": total_timesteps,
        }

    # Extract training parameters
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 100000)
    checkpoint_interval = training_config.get("checkpoint_interval", 10000)
    eval_interval = training_config.get("eval_interval", 5000)
    update_interval = training_config.get("update_interval", 2048)  # Steps before PPO update
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "checkpoints")
    
    # Check for progress callback
    progress_update_callback = None
    if "callbacks" in config and "progress_update" in config["callbacks"]:
        progress_update_callback = config["callbacks"]["progress_update"]
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up tracking
    steps_done = 0
    episode_num = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    steps_since_update = 0
    
    # Set up evaluation
    best_eval_reward = float('-inf')
    
    # Reset environment
    obs, info = env.reset()
    
    logger.info(f"Starting training for {total_timesteps} timesteps")
    training_start_time = time.time()
    last_progress_update_time = time.time()
    
    # Main training loop
    while steps_done < total_timesteps:
        # Get action from agent
        action = agent.get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Collect experience in buffer
        agent.train_step(obs, action, reward, next_obs, done or truncated)
        steps_since_update += 1
        
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
            
            # Log progress more frequently for better visibility
            if episode_num % 5 == 0:
                recent_rewards = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
                logger.info(f"Episode {episode_num}, Steps: {steps_done}/{total_timesteps} ({steps_done/total_timesteps*100:.1f}%), "
                           f"Recent Reward: {recent_rewards:.2f}")
                
                # If we have evaluation results, include them in log
                if best_eval_reward > float('-inf'):
                    logger.info(f"Best evaluation reward so far: {best_eval_reward:.2f}")
        
        # Update progress callback
        current_time = time.time()
        if progress_update_callback and (current_time - last_progress_update_time > 0.5 or steps_done % 100 == 0):
            # Prepare metrics
            progress = steps_done / total_timesteps
            
            # Simple safe metrics
            metrics = {}
            
            # Only add metrics that exist and can be converted to float
            try:
                metrics["episode_reward"] = float(current_episode_reward) if current_episode_reward else 0.0
                if episode_rewards:
                    metrics["average_reward"] = float(np.mean(episode_rewards[-100:]))
                metrics["episode_length"] = float(current_episode_length)
                metrics["episode_num"] = float(episode_num)
                metrics["steps_done"] = float(steps_done)
            except (ValueError, TypeError) as e:
                # Just log and continue if conversion fails
                print(f"Metrics conversion error (ignoring): {str(e)}")
            
            # Call the progress update callback - wrapped in try/except to prevent training interruption
            try:
                progress_update_callback(steps_done, total_timesteps, metrics)
            except Exception as e:
                # Just log the error but continue training
                print(f"Error in progress callback: {str(e)}")
                logger.error(f"Error in progress callback: {str(e)}")
            
            last_progress_update_time = current_time
        
        # Update policy if we have collected enough steps
        if steps_since_update >= update_interval:
            logger.info(f"Updating policy after collecting {steps_since_update} experiences")
            update_results = agent.update_if_buffer_ready()
            
            if update_results:
                # Log update metrics
                logger.info(f"Policy update: policy_loss={update_results.get('policy_loss', 0):.4f}, "
                           f"value_loss={update_results.get('value_loss', 0):.4f}, "
                           f"entropy={update_results.get('entropy', 0):.4f}")
                
                # Reset counter
                steps_since_update = 0
                
                # Log to MLflow if available
                if mlflow_manager is not None:
                    mlflow_metrics = {
                        k: float(v) for k, v in update_results.items() if v is not None
                    }
                    mlflow_manager.log_metrics(mlflow_metrics, step=steps_done)
        
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
            
            # 평가 실행
            eval_rewards = evaluate_agent(agent, eval_env, num_episodes=5)
            mean_eval_reward = np.mean(eval_rewards)
            
            logger.info(f"Evaluation at step {steps_done}: Mean reward: {mean_eval_reward:.2f}")
            
            # Log to MLflow
            if mlflow_manager is not None:
                mlflow_manager.log_metrics({
                    "eval_reward": mean_eval_reward,
                    "eval_reward_std": np.std(eval_rewards),
                }, step=steps_done)
            
            # Update metrics for progress callback
            if progress_update_callback:
                # Simple safe metrics
                metrics = {}
                
                # Only add metrics that exist and can be converted to float
                try:
                    metrics["episode_reward"] = float(current_episode_reward) if current_episode_reward else 0.0
                    if episode_rewards:
                        metrics["average_reward"] = float(np.mean(episode_rewards[-100:]))
                    metrics["episode_length"] = float(current_episode_length)
                    metrics["eval_reward"] = float(mean_eval_reward)
                    metrics["episode_num"] = float(episode_num)
                    metrics["steps_done"] = float(steps_done)
                except (ValueError, TypeError) as e:
                    # Just log and continue if conversion fails
                    print(f"Metrics conversion error during eval (ignoring): {str(e)}")
                
                try:
                    progress_update_callback(steps_done, total_timesteps, metrics)
                except Exception as e:
                    print(f"Error in evaluation progress callback: {str(e)}")
                    logger.error(f"Error in progress callback during evaluation: {str(e)}")
            
            # Save best model
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                best_model_path = os.path.join(checkpoint_dir, "best_agent.pt")
                agent.save(best_model_path)
                logger.info(f"New best model with reward {best_eval_reward:.2f} saved to {best_model_path}")
            
            # 평가 후 학습 환경 리셋
            obs, info = env.reset()
    
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
    
    # Final progress update
    if progress_update_callback:
        # Simple safe metrics
        metrics = {}
        
        # Only add metrics that exist and can be converted to float
        try:
            if episode_rewards:
                metrics["average_reward"] = float(np.mean(episode_rewards[-100:]))
            metrics["best_eval_reward"] = float(best_eval_reward)
            metrics["training_duration"] = float(training_duration)
            metrics["total_episodes"] = float(episode_num)
        except (ValueError, TypeError) as e:
            # Just log and continue if conversion fails
            print(f"Metrics conversion error for final update (ignoring): {str(e)}")
        
        try:
            progress_update_callback(total_timesteps, total_timesteps, metrics)
        except Exception as e:
            print(f"Error in final progress callback: {str(e)}")
            logger.error(f"Error in final progress callback: {str(e)}")
    
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
    - Uses proper PPO rollouts for experience collection
    
    Implementation Notes:
    - Collects experiences for a fixed number of steps before updating
    - Follows PPO algorithm with multiple epochs of optimization
    - Each agent trains on its own collected experiences
    - Supports shared experience buffer for knowledge transfer
    
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
    update_interval = training_config.get("update_interval", 2048)  # Steps before PPO update
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
    steps_since_update = 0
    
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
        
        # Collect experiences for each agent
        for agent_id, agent in agents.items():
            # Get agent-specific done flag - either this agent is done or the episode is truncated
            agent_done = bool(dones[agent_id] or truncated)
            
            # Store experience in agent's buffer
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
                # Format experience for shared buffer
                experience = {
                    "state": obs_dict[agent_id],
                    "action": actions[agent_id],
                    "reward": rewards[agent_id],
                    "next_state": next_obs_dict[agent_id],
                    "done": agent_done
                }
                shared_buffer.append(experience)
                
                # Trim buffer if it exceeds max size
                if len(shared_buffer) > max_buffer_size:
                    shared_buffer.pop(0)
        
        # Update tracking
        current_episode_length += 1
        steps_done += 1
        steps_since_update += 1
        
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
        
        # Update agents with collected experiences if enough steps
        if steps_since_update >= update_interval:
            logger.info(f"Updating agents after collecting {steps_since_update} steps...")
            
            # Update each agent with its own experiences
            for agent_id, agent in agents.items():
                if hasattr(agent, "update_if_buffer_ready"):
                    update_results = agent.update_if_buffer_ready()
                    
                    if update_results:
                        # Log update metrics for this agent
                        logger.info(
                            f"Agent {agent_id} update: "
                            f"policy_loss={update_results.get('policy_loss', 0):.4f}, "
                            f"value_loss={update_results.get('value_loss', 0):.4f}, "
                            f"entropy={update_results.get('entropy', 0):.4f}"
                        )
                        
                        # Log to MLflow if available
                        if mlflow_manager is not None:
                            mlflow_metrics = {
                                f"{agent_id}/{k}": float(v) 
                                for k, v in update_results.items() if v is not None
                            }
                            mlflow_manager.log_metrics(mlflow_metrics, step=steps_done)
            
            # Reset steps counter
            steps_since_update = 0
            
            # After updating, let agents learn from shared experiences if enabled
            if use_shared_buffer and len(shared_buffer) > 0:
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
    Unified training pipeline for single/multi agent RL for trading.
    
    This function handles creation of environments, agents, and manages the training
    process based on the provided configuration.
    
    Args:
        config: Configuration dictionary
        data: Optional data to use for training (if not provided, will load from config)
        
    Returns:
        Dictionary with training results
    """
    # Setup logging
    logger.info("Starting training pipeline")
    
    # Extract paths from config
    paths_config = config.get("paths", {})
    checkpoint_dir = paths_config.get("checkpoint_dir", "checkpoints")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set random seed if provided
    if "seed" in config.get("training", {}):
        seed = config["training"]["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    # Get MLflow manager if provided
    mlflow_manager = config.get("mlflow_manager", None)
    
    # Create environment based on config
    env = create_env(config, data)
    
    # Determine agent type
    env_type = config["env"]["type"]
    agent_config = config.get("agent", {})
    agent_type = agent_config.get("type", "ppo")
    
    # Create or use provided agent
    if "pre_created_agent" in config:
        logger.info(f"Using pre-created agent: {config['pre_created_agent'].__class__.__name__}")
        agent = config["pre_created_agent"]
    else:
        # Create agent based on environment type and agent configuration
        logger.info(f"Creating agent with type: {agent_type}")
        agent = create_agent(
            agent_type=agent_type,
            config=agent_config,
            observation_space=env.observation_space,
            action_space=env.action_space
        )
    
    # Select training method based on environment type
    if env_type == "single_asset_rl" or env_type == "multi_asset_rl":
        # Single agent training for both single and multi-asset environments
        logger.info(f"Starting single agent training for environment type: {env_type}")
        results = train_single_agent(agent, env, config, mlflow_manager)
    elif env_type == "multi_agent_rl" or env_type == "multi_asset_multi_agent_rl":
        # Build agents dict from multi_agent_configs
        multi_agent_cfgs = config["env"].get("multi_agent_configs", [])
        agents = {}
        for agent_cfg in multi_agent_cfgs:
            agent_id = agent_cfg["id"]
            strategy = agent_cfg.get("strategy", "momentum")
            obs_space = env.observation_spaces.get(agent_id, env.observation_space)
            act_space = env.action_spaces.get(agent_id, env.action_space)
            agents[agent_id] = create_agent(
                agent_type=strategy,
                config=agent_cfg,
                observation_space=obs_space,
                action_space=act_space,
            )
        # Multi-agent training
        logger.info(f"Starting multi-agent training for environment type: {env_type}")
        results = train_multi_agent(agents, env, config, mlflow_manager)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")
    
    logger.info("Training pipeline completed")
    return results

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
    update_interval = training_config.get("update_interval", 2048)  # Steps before PPO update
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
    
    # Initialize tracking variables
    episode_rewards = {agent_id: [] for agent_id in env.agents}
    episode_lengths = []
    best_eval_rewards = {agent_id: float('-inf') for agent_id in env.agents}
    
    # 메타 에이전트 ID 추적 (manager에서 가져옴)
    meta_agent_id = manager.meta_agent_id
    if meta_agent_id:
        best_eval_rewards[meta_agent_id] = float('-inf')
        logger.info(f"Meta agent ID detected: {meta_agent_id}")
    
    # Create checkpoint directories
    checkpoint_dir = os.path.join(config.get("checkpoint_dir", "checkpoints"))
    manager_checkpoint_dir = os.path.join(checkpoint_dir, "manager")
    os.makedirs(manager_checkpoint_dir, exist_ok=True)
    
    # Training loop
    training_start_time = time.time()
    total_steps = 0
    episode = 0
    
    # 환경 초기화
    obs_dict, info = env.reset()
    
    # 에피소드 보상 초기화
    current_episode_rewards = {agent_id: 0 for agent_id in env.agents}
    current_episode_length = 0
    
    # Main training loop
    while total_steps < total_timesteps:
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
        total_steps += 1
        
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
                    mlflow_manager.log_metric(f"{agent_id}/episode_reward", reward, step=episode)
                    mlflow_manager.log_metric(f"{agent_id}/avg_reward_100", avg_reward, step=episode)
            
            # Add episode length to tracking
            episode_lengths.append(current_episode_length)
            
            # Reset environment and trackers
            obs_dict, info = env.reset()
            current_episode_rewards = {agent_id: 0 for agent_id in env.agents}
            current_episode_length = 0
            episode += 1
            
            # Log episode stats
            if episode % log_interval == 0:
                episode_avg_rewards = {
                    agent_id: np.mean(episode_rewards[agent_id][-log_interval:])
                    for agent_id in env.agents
                }
                
                logger.info(
                    f"Episode {episode} | "
                    f"Steps: {total_steps} | "
                    f"Avg rewards: {episode_avg_rewards} | "
                    f"Time elapsed: {time.time() - training_start_time:.2f}s"
                )
        
        # Periodic evaluation
        if total_steps % eval_interval == 0:
            # Evaluate using manager for proper ensemble coordination
            eval_returns = evaluate_with_manager(env, manager, num_episodes=5)
            
            # Log evaluation results
            for agent_id, returns in eval_returns.items():
                avg_eval_return = np.mean(returns)
                
                logger.info(
                    f"Agent {agent_id} | "
                    f"Step {total_steps} | "
                    f"Evaluation avg return: {avg_eval_return:.4f}"
                )
                
                # best_eval_rewards에 agent_id가 없으면 초기화
                if agent_id not in best_eval_rewards:
                    logger.warning(f"Agent {agent_id} not found in best_eval_rewards. Initializing.")
                    best_eval_rewards[agent_id] = float('-inf')
                
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
                        mlflow_manager.log_metric(f"{agent_id}/eval_episode_{i+1}_return", reward, step=total_steps)
                    mlflow_manager.log_metric(f"{agent_id}/eval_avg_return", avg_eval_return, step=total_steps)
            
            # Save entire manager for ensemble functionality
            manager_checkpoint_path = os.path.join(manager_checkpoint_dir, f"manager_best.pt")
            manager.save(manager_checkpoint_path)
            logger.info(f"Saved best manager checkpoint at step {total_steps}")
        
        # Regular checkpoint saving
        if total_steps % checkpoint_interval == 0:
            # Save manager
            manager_checkpoint_path = os.path.join(manager_checkpoint_dir, f"manager_step_{total_steps}.pt")
            manager.save(manager_checkpoint_path)
            
            # Save individual agents
            for agent_id, agent in manager.agents.items():
                if hasattr(agent, "save"):
                    agent_checkpoint_path = os.path.join(checkpoint_dir, agent_id, f"checkpoint_{total_steps}.pt")
                    os.makedirs(os.path.dirname(agent_checkpoint_path), exist_ok=True)
                    agent.save(agent_checkpoint_path)
            
            logger.info(f"Saved checkpoints at step {total_steps}")
    
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


# ---------------------------------------------------------------------------
# SB3-native training pipeline
# ---------------------------------------------------------------------------

def train_sb3_agent(
    sb3_agent,
    train_env,
    config: Dict[str, Any],
    eval_env=None,
    mlflow_manager=None,
) -> Dict[str, Any]:
    """
    Train an SB3AgentWrapper using model.learn() with proper callbacks.

    Args:
        sb3_agent: SB3AgentWrapper instance (wraps PPO/SAC/TD3/A2C).
        train_env: Vectorised training environment (VecEnv).
        config: Full training config dict.
        eval_env: Optional separate vectorised environment for EvalCallback.
        mlflow_manager: Optional MLflowManager for metric logging.

    Returns:
        Dict with 'agent', 'model_path', 'best_model_path', 'total_timesteps'.
    """
    from stable_baselines3.common.callbacks import CallbackList
    from training.callbacks.sb3_callbacks import (
        MLflowLoggingCallback,
        SB3CheckpointCallback,
        SB3EvalCallback,
    )

    training_cfg = config.get("training", {})
    total_timesteps = training_cfg.get("total_timesteps", 100_000)
    checkpoint_interval = training_cfg.get("checkpoint_interval", 50_000)
    eval_interval = training_cfg.get("eval_interval", 10_000)
    log_interval = training_cfg.get("log_interval", 1_000)
    n_eval_episodes = training_cfg.get("n_eval_episodes", 5)

    paths_cfg = config.get("paths", {})
    checkpoint_dir = paths_cfg.get("checkpoint_dir", "checkpoints")
    best_model_dir = os.path.join(checkpoint_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Build callback list
    callbacks = [
        MLflowLoggingCallback(
            mlflow_manager=mlflow_manager,
            log_interval=log_interval,
        ),
        SB3CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=checkpoint_dir,
            mlflow_manager=mlflow_manager,
        ),
    ]

    if eval_env is not None:
        callbacks.append(
            SB3EvalCallback(
                eval_env=eval_env,
                mlflow_manager=mlflow_manager,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_interval,
                best_model_save_path=best_model_dir,
                verbose=1,
            )
        )

    callback = CallbackList(callbacks)

    logger.info(
        f"Starting SB3 training: {total_timesteps:,} timesteps, "
        f"algo={sb3_agent.algo_type}"
    )

    sb3_agent.train(train_env, total_timesteps=total_timesteps, callbacks=callback)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    sb3_agent.save(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}.zip")

    if mlflow_manager is not None:
        try:
            mlflow_manager.log_artifact(f"{final_path}.zip")
        except Exception:
            pass

    best_path = os.path.join(best_model_dir, "best_model")
    return {
        "agent": sb3_agent,
        "model_path": final_path,
        "best_model_path": best_path if os.path.exists(f"{best_path}.zip") else None,
        "total_timesteps": total_timesteps,
    }


def train_ensemble_agent(
    ensemble,
    train_env,
    config: Dict[str, Any],
    eval_env=None,
    mlflow_manager=None,
) -> Dict[str, Any]:
    """
    Train an EnsembleManager (PPO + SAC + TD3) sequentially on train_env.

    After training, evaluates each agent on eval_env (if provided) and
    updates the ensemble weights based on rolling Sharpe.

    Args:
        ensemble: EnsembleManager instance.
        train_env: Gymnasium-compatible or VecEnv training environment.
        config: Full training config dict. Reads ``training.total_timesteps``
                and optionally ``ensemble.rebalance_interval``.
        eval_env: Optional evaluation environment for weight updates.
        mlflow_manager: Optional MLflowManager for experiment tracking.

    Returns:
        {
            "agent_results"   : {agent_id: train_result_dict},
            "final_weights"   : {agent_id: float},
            "ensemble_metrics": dict,
        }
    """
    from agents.ensemble.ensemble_manager import EnsembleManager

    training_cfg = config.get("training", {})
    total_timesteps = training_cfg.get("total_timesteps", 100_000)
    ensemble_cfg = config.get("ensemble", {})
    rebalance_interval = ensemble_cfg.get(
        "rebalance_interval", getattr(ensemble, "rebalance_interval", 1000)
    )

    paths_cfg = config.get("paths", {})
    checkpoint_dir = paths_cfg.get("checkpoint_dir", "checkpoints")
    ensemble_save_dir = os.path.join(checkpoint_dir, "ensemble")

    logger.info(
        "Starting ensemble training: %d agents, %s total timesteps",
        len(ensemble),
        f"{total_timesteps:,}",
    )

    if mlflow_manager is not None:
        try:
            mlflow_manager.log_params({
                "ensemble_method": ensemble.method,
                "ensemble_n_agents": len(ensemble),
                "ensemble_total_timesteps": total_timesteps,
            })
        except Exception:
            pass

    agent_results = ensemble.train_all(train_env, total_timesteps=total_timesteps)

    # Post-training evaluation and weight update
    if eval_env is not None:
        logger.info("Evaluating ensemble agents on eval_env …")
        eval_metrics = ensemble.evaluate_agents(eval_env, n_eval_episodes=5)
        ensemble.update_weights(eval_metrics)

        if mlflow_manager is not None:
            try:
                for agent_id, m in eval_metrics.items():
                    mlflow_manager.log_metrics({
                        f"ensemble_{agent_id}_mean_reward": m["mean_reward"],
                        f"ensemble_{agent_id}_std_reward": m["std_reward"],
                    })
                for agent_id, w in ensemble.get_weights().items():
                    mlflow_manager.log_metrics({f"ensemble_weight_{agent_id}": w})
            except Exception:
                pass

    # Save ensemble checkpoint
    ensemble.save(ensemble_save_dir)
    logger.info("Ensemble training complete. Saved to %s", ensemble_save_dir)

    return {
        "agent_results": agent_results,
        "final_weights": ensemble.get_weights(),
        "ensemble_metrics": ensemble.get_ensemble_metrics(),
        "ensemble_save_dir": ensemble_save_dir,
    }