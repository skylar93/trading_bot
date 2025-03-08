"""
Meta-agent training module with enhanced architecture that uses sub-agent hidden states.

This module provides functionality to train a meta-agent that coordinates decisions 
from multiple sub-agents, utilizing both their observations and internal hidden states.
"""

import os
import logging
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.multi.multi_agent_manager import MultiAgentManager
from utils.data_loader import load_and_prepare_data
from agents.agent_factory import create_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_meta_agent(
    data: pd.DataFrame,
    agent_configs: List[Dict[str, Any]],
    meta_config: Dict[str, Any],
    window_size: int = 60,
    episodes: int = 100,
    eval_interval: int = 10,
    save_path: str = "./models",
    use_hidden_states: bool = True,
    seed: int = 42
):
    """
    Train a meta-agent with sub-agent hidden states.
    
    Args:
        data: DataFrame with prepared data
        agent_configs: List of sub-agent configurations
        meta_config: Meta-agent configuration
        window_size: Window size for observations
        episodes: Number of episodes to train
        eval_interval: Interval for evaluation and model saving
        save_path: Path to save models
        use_hidden_states: Whether to use sub-agent hidden states
        seed: Random seed
        
    Returns:
        Trained MultiAgentManager and training metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("Initializing multi-agent environment...")
    
    # Create environment
    env = MultiAgentTradingEnv(
        data=data,
        agent_configs=agent_configs,
        window_size=window_size,
        trading_fee=0.001
    )
    
    # Add meta-agent configuration
    all_configs = agent_configs.copy()
    if meta_config not in all_configs:
        all_configs.append(meta_config)
    
    # Set meta-agent observation dimension based on whether hidden states are used
    if use_hidden_states:
        # Calculate approximate hidden state dimensions
        # This will be refined dynamically during training
        hidden_dim_estimate = 128  # Typical hidden dimension
        meta_obs_dim = sum(config.get("observation_size", window_size * 5) for config in agent_configs)
        meta_obs_dim += len(agent_configs) * hidden_dim_estimate  # Add space for hidden states
        
        meta_config["observation_size"] = meta_obs_dim
        logger.info(f"Meta-agent observation dimension (with hidden states): {meta_obs_dim}")
    
    # Create manager
    logger.info("Creating multi-agent manager with meta-agent...")
    manager = MultiAgentManager(
        agent_configs=all_configs,
        ensemble_method="meta"
    )
    
    # Training metrics
    metrics = {
        "episode_returns": [],
        "agent_returns": {agent_id: [] for agent_id in env.agents},
        "meta_rewards": [],
        "training_losses": [],
    }
    
    # Training loop
    logger.info(f"Starting training for {episodes} episodes...")
    
    for episode in tqdm(range(episodes), desc="Training Progress"):
        # Reset environment
        observations, info = env.reset()
        done = False
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        meta_episode_reward = 0.0
        
        # Episode loop
        while not done:
            # Get actions from manager (includes meta-agent decision)
            actions = manager.act(observations, deterministic=False)
            
            # Get hidden states if using them - using the new method
            hidden_states = {}
            if use_hidden_states:
                for agent_id, agent in manager.agents.items():
                    if agent_id != manager.meta_agent_id and hasattr(agent, 'get_action_with_hidden_state'):
                        # Use the agent's observation
                        _, hidden_state = agent.get_action_with_hidden_state(observations[agent_id], deterministic=False)
                        hidden_states[agent_id] = hidden_state
            
            # Take step in environment
            next_observations, rewards, dones, truncated, infos = env.step(actions)
            
            # Collect experiences for all agents
            experiences = {}
            for agent_id in env.agents:
                experiences[agent_id] = {
                    "observation": observations[agent_id],
                    "action": actions[agent_id],
                    "reward": rewards[agent_id],
                    "next_observation": next_observations[agent_id],
                    "done": dones[agent_id]
                }
                
                # If available, add hidden states to experience
                if agent_id in hidden_states:
                    experiences[agent_id]["hidden_state"] = hidden_states[agent_id]
                
                # Track rewards
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Create meta-agent experience explicitly if not created by manager
            meta_id = manager.meta_agent_id
            if meta_id not in experiences:
                # Get meta-observations with or without hidden states
                if use_hidden_states:
                    # Get hidden states from infos
                    hidden_states = {}
                    for agent_id in env.agents:
                        if "hidden_state" in infos[agent_id]:
                            hidden_states[agent_id] = infos[agent_id]["hidden_state"]
                    
                    # Create enhanced meta observation
                    meta_obs = manager.get_meta_observation(observations)
                    meta_next_obs = manager.get_meta_observation(next_observations)
                else:
                    # Standard meta observation without hidden states
                    meta_obs = manager.get_meta_observation(observations)
                    meta_next_obs = manager.get_meta_observation(next_observations)
                
                # Use the meta-agent's action from the manager
                meta_action = actions.get(meta_id, np.array([0.0]))
                
                # Calculate meta-agent reward (can be customized)
                if hasattr(manager, "continuous_ensemble") and manager.continuous_ensemble:
                    # For continuous weights, reward is weighted sum of sub-agent rewards
                    weights = meta_action
                    meta_reward = sum(weights[i] * rewards[agent_id] 
                                     for i, agent_id in enumerate(sorted(env.agents)))
                else:
                    # For discrete selection, reward is that of the selected agent
                    selected_idx = int(meta_action[0]) % len(env.agents)
                    selected_agent = sorted(env.agents)[selected_idx]
                    meta_reward = rewards[selected_agent]
                
                # Create meta experience
                experiences[meta_id] = {
                    "observation": meta_obs,
                    "action": meta_action,
                    "reward": meta_reward,
                    "next_observation": meta_next_obs,
                    "done": any(dones.values())
                }
                
                # Track meta reward
                meta_episode_reward += meta_reward
            
            # Train all agents including meta-agent
            train_metrics = manager.train_step(experiences)
            if train_metrics.get(meta_id, {}).get("policy_loss"):
                metrics["training_losses"].append(train_metrics[meta_id]["policy_loss"])
            
            # Update observations for next step
            observations = next_observations
            
            # Check if episode is done
            done = all(dones.values())
        
        # Track episode returns
        metrics["episode_returns"].append(sum(episode_rewards.values()) / len(episode_rewards))
        for agent_id, reward in episode_rewards.items():
            metrics["agent_returns"][agent_id].append(reward)
        metrics["meta_rewards"].append(meta_episode_reward)
        
        # Log progress
        if (episode + 1) % 5 == 0:
            logger.info(f"Episode {episode+1}/{episodes} - "
                        f"Avg Return: {metrics['episode_returns'][-1]:.2f}, "
                        f"Meta Reward: {meta_episode_reward:.2f}")
        
        # Evaluate and save model
        if (episode + 1) % eval_interval == 0:
            eval_returns = evaluate_meta_agent(env, manager, num_episodes=5)
            logger.info(f"Evaluation after episode {episode+1} - "
                        f"Avg Return: {eval_returns:.2f}")
            
            # Save model
            save_dir = os.path.join(save_path, f"meta_ep{episode+1}")
            os.makedirs(save_dir, exist_ok=True)
            manager.save(save_dir)
            logger.info(f"Saved model to {save_dir}")
    
    # Final save
    final_save_dir = os.path.join(save_path, "meta_final")
    os.makedirs(final_save_dir, exist_ok=True)
    manager.save(final_save_dir)
    logger.info(f"Training completed. Saved final model to {final_save_dir}")
    
    # Plot training metrics
    plot_training_metrics(metrics, save_path)
    
    return manager, metrics

def evaluate_meta_agent(
    env: MultiAgentTradingEnv,
    manager: MultiAgentManager,
    num_episodes: int = 10
) -> float:
    """
    Evaluate a trained meta-agent.
    
    Args:
        env: MultiAgentTradingEnv to evaluate in
        manager: Trained MultiAgentManager
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average return across episodes
    """
    total_returns = []
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        done = False
        episode_rewards = {agent_id: 0.0 for agent_id in env.agents}
        
        while not done:
            # Get actions from manager (deterministic for evaluation)
            actions = manager.act(observations, deterministic=True)
            
            # Take step in environment
            next_observations, rewards, dones, _, _ = env.step(actions)
            
            # Track rewards
            for agent_id, reward in rewards.items():
                episode_rewards[agent_id] += reward
            
            # Update observations
            observations = next_observations
            
            # Check if episode is done
            done = all(dones.values())
        
        # Calculate average return
        avg_return = sum(episode_rewards.values()) / len(episode_rewards)
        total_returns.append(avg_return)
    
    return sum(total_returns) / len(total_returns)

def plot_training_metrics(metrics: Dict[str, Any], save_path: str):
    """
    Plot training metrics and save figures.
    
    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save figures
    """
    plt.figure(figsize=(12, 8))
    
    # Plot episode returns
    plt.subplot(2, 2, 1)
    plt.plot(metrics["episode_returns"])
    plt.title("Average Episode Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    
    # Plot agent returns
    plt.subplot(2, 2, 2)
    for agent_id, returns in metrics["agent_returns"].items():
        if agent_id != "meta_agent":  # Don't plot meta-agent returns
            plt.plot(returns, label=agent_id)
    plt.title("Individual Agent Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    
    # Plot meta rewards
    plt.subplot(2, 2, 3)
    plt.plot(metrics["meta_rewards"])
    plt.title("Meta-Agent Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot training losses
    if metrics["training_losses"]:
        plt.subplot(2, 2, 4)
        plt.plot(metrics["training_losses"])
        plt.title("Meta-Agent Policy Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(save_path, "training_metrics.png")
    plt.savefig(fig_path)
    logger.info(f"Saved training metrics plot to {fig_path}")

def main():
    """Main entry point for meta-agent training."""
    parser = argparse.ArgumentParser(description="Train a meta-agent with sub-agent hidden states")
    
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data file")
    parser.add_argument("--config_path", type=str, required=True, 
                        help="Path to agent configurations")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes to train")
    parser.add_argument("--window_size", type=int, default=60, 
                        help="Window size for observations")
    parser.add_argument("--use_hidden_states", action="store_true", 
                        help="Use sub-agent hidden states in meta-agent")
    parser.add_argument("--save_path", type=str, default="./models/meta", 
                        help="Path to save models")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = load_and_prepare_data(args.data_path)
    
    # Load agent configurations
    logger.info(f"Loading agent configurations from {args.config_path}")
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    agent_configs = config["agent_configs"]
    meta_config = config["meta_config"]
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_path, f"meta_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump({
            "agent_configs": agent_configs,
            "meta_config": meta_config,
            "training_params": {
                "episodes": args.episodes,
                "window_size": args.window_size,
                "use_hidden_states": args.use_hidden_states,
                "seed": args.seed
            }
        }, f, indent=4)
    
    # Train meta-agent
    manager, metrics = train_meta_agent(
        data=data,
        agent_configs=agent_configs,
        meta_config=meta_config,
        window_size=args.window_size,
        episodes=args.episodes,
        save_path=save_path,
        use_hidden_states=args.use_hidden_states,
        seed=args.seed
    )
    
    # Save metrics
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {
            "episode_returns": [float(r) for r in metrics["episode_returns"]],
            "agent_returns": {
                agent_id: [float(r) for r in returns]
                for agent_id, returns in metrics["agent_returns"].items()
            },
            "meta_rewards": [float(r) for r in metrics["meta_rewards"]],
            "training_losses": [float(l) for l in metrics["training_losses"]] if metrics["training_losses"] else []
        }
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Training completed! Results saved to {save_path}")

if __name__ == "__main__":
    main() 