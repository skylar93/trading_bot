import os
import numpy as np
import pandas as pd
from typing import Dict, List
import mlflow
from datetime import datetime, timedelta
import torch
import logging

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.ppo_agent import PPOAgent
from training.train import fetch_training_data

logger = logging.getLogger(__name__)

def train_multi_agent_system(
    env: MultiAgentTradingEnv,
    agents: Dict[str, PPOAgent],
    num_episodes: int = 1000,
    save_path: str = 'models/multi_agent',
    save_freq: int = 100
) -> Dict[str, Dict[str, List[float]]]:
    """Train multiple agents simultaneously"""
    
    # Create directory for saving models
    try:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Created or verified model save directory: {save_path}")
    except Exception as e:
        logger.error(f"Failed to create model save directory: {str(e)}")
        raise
    
    # Initialize metrics tracking for each agent
    metrics = {
        agent_id: {
            'episode_rewards': [],
            'portfolio_values': [],
            'policy_losses': [],
            'value_losses': []
        }
        for agent_id in agents.keys()
    }
    
    # Set up MLflow tracking
    mlflow.set_experiment('trading_bot_multi_agent_training')
    
    # Training loop
    for episode in range(num_episodes):
        observations, _ = env.reset()
        dones = {agent_id: False for agent_id in agents.keys()}
        done = False
        
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        replay_buffers = {agent_id: [] for agent_id in agents.keys()}
        
        # Episode loop
        while not done:
            # Get actions from all agents
            actions = {}
            for agent_id, agent in agents.items():
                if not dones[agent_id]:
                    action = agent.get_action(observations[agent_id])
                    actions[agent_id] = action
            
            # Take step in environment
            next_observations, rewards, dones, truncated, infos = env.step(actions)
            
            # Store experiences for each agent
            for agent_id in agents.keys():
                if not dones[agent_id]:
                    experience = {
                        'state': observations[agent_id],
                        'action': actions[agent_id],
                        'reward': rewards[agent_id],
                        'next_state': next_observations[agent_id],
                        'done': dones[agent_id]
                    }
                    replay_buffers[agent_id].append(experience)
                    episode_rewards[agent_id] += rewards[agent_id]
            
            # Update observations
            observations = next_observations
            done = all(dones.values())
        
        # Train each agent
        loss_infos = {}
        for agent_id, agent in agents.items():
            if len(replay_buffers[agent_id]) > 0:
                loss_info = agent.train(replay_buffers[agent_id])
                loss_infos[agent_id] = loss_info
                
                # Update metrics
                metrics[agent_id]['episode_rewards'].append(episode_rewards[agent_id])
                metrics[agent_id]['portfolio_values'].append(infos[agent_id]['portfolio_value'])
                metrics[agent_id]['policy_losses'].append(loss_info['policy_loss'])
                metrics[agent_id]['value_losses'].append(loss_info['value_loss'])
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    f"{agent_id}_reward": episode_rewards[agent_id],
                    f"{agent_id}_portfolio_value": infos[agent_id]['portfolio_value'],
                    f"{agent_id}_policy_loss": loss_info['policy_loss'],
                    f"{agent_id}_value_loss": loss_info['value_loss']
                }, step=episode)
        
        # Save models periodically
        if episode > 0 and episode % save_freq == 0:
            for agent_id, agent in agents.items():
                try:
                    save_file = os.path.join(save_path, f'{agent_id}_episode_{episode}.pt')
                    agent.save(save_file)
                    logger.info(f"Saved model for agent {agent_id} at episode {episode}")
                except Exception as e:
                    logger.error(f"Failed to save model for agent {agent_id}: {str(e)}")
        
        # Print progress
        if episode % 10 == 0:
            logger.info(f"\nEpisode {episode}/{num_episodes}")
            for agent_id in agents.keys():
                logger.info(f"\n{agent_id}:")
                logger.info(f"Episode Reward: {episode_rewards[agent_id]:.2f}")
                logger.info(f"Portfolio Value: {infos[agent_id]['portfolio_value']:.2f}")
                if agent_id in loss_infos:
                    logger.info(f"Policy Loss: {loss_infos[agent_id]['policy_loss']:.4f}")
                    logger.info(f"Value Loss: {loss_infos[agent_id]['value_loss']:.4f}")
            logger.info("\n-------------------")
    
    return metrics

if __name__ == "__main__":
    # Fetch training data
    df = fetch_training_data()
    
    # Define agent configurations
    agent_configs = [
        {
            'id': 'momentum_trader',
            'strategy': 'momentum',
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        },
        {
            'id': 'mean_reversion_trader',
            'strategy': 'mean_reversion',
            'initial_balance': 10000.0,
            'fee_multiplier': 1.0
        },
        {
            'id': 'market_maker',
            'strategy': 'market_making',
            'initial_balance': 10000.0,
            'fee_multiplier': 0.8  # Market makers often get fee discounts
        }
    ]
    
    # Create environment
    env = MultiAgentTradingEnv(df, agent_configs)
    
    # Create agents
    agents = {
        config['id']: PPOAgent(
            env.observation_spaces[config['id']], 
            env.action_spaces[config['id']]
        )
        for config in agent_configs
    }
    
    # Train agents
    metrics = train_multi_agent_system(env, agents)
    
    # Save final models
    for agent_id, agent in agents.items():
        agent.save(f'models/multi_agent/{agent_id}_final.pt')
    
    # Plot results
    import matplotlib.pyplot as plt
    
    metrics_to_plot = ['episode_rewards', 'portfolio_values', 'policy_losses', 'value_losses']
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 4*len(metrics_to_plot)))
    
    for i, metric in enumerate(metrics_to_plot):
        for agent_id in agents.keys():
            axes[i].plot(metrics[agent_id][metric], label=agent_id)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('multi_agent_training_results.png')
    plt.close()