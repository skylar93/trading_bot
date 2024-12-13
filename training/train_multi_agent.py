import os
import numpy as np
import pandas as pd
from typing import Dict, List
import mlflow
from datetime import datetime, timedelta
import torch

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.ppo_agent import PPOAgent
from training.train import fetch_training_data

def train_multi_agent_system(
    env: MultiAgentTradingEnv,
    agents: Dict[str, PPOAgent],
    num_episodes: int = 1000,
    save_path: str = 'models/multi_agent',
    save_freq: int = 100
) -> Dict[str, Dict[str, List[float]]]:
    """Train multiple agents simultaneously"""
    
    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    
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
            policy_outputs = {}
            for agent_id, agent in agents.items():
                if not dones[agent_id]:
                    # Get action and policy outputs
                    state_tensor = torch.FloatTensor(observations[agent_id]).reshape(1, -1).to(agent.device)
                    mean, log_std = agent.network(state_tensor)[0:2]
                    action = agent.get_action(observations[agent_id])
                    actions[agent_id] = action
                    policy_outputs[agent_id] = (mean.detach().cpu().numpy(), log_std.detach().cpu().numpy())
            
            # Take step in environment
            next_observations, rewards, dones, _, infos = env.step(actions)
            
            # Store experiences for each agent
            for agent_id in agents.keys():
                if not dones[agent_id]:
                    replay_buffers[agent_id].append((
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id],
                        dones[agent_id],
                        policy_outputs[agent_id][0],
                        policy_outputs[agent_id][1]
                    ))
                    episode_rewards[agent_id] += rewards[agent_id]
            
            # Update observations
            observations = next_observations
            done = all(dones.values())
        
        # Train each agent
        loss_infos = {}
        for agent_id, agent in agents.items():
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
                agent.save(os.path.join(save_path, f'{agent_id}_episode_{episode}.pt'))
        
        # Print progress
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            for agent_id in agents.keys():
                print(f"\n{agent_id}:")
                print(f"Episode Reward: {episode_rewards[agent_id]:.2f}")
                print(f"Portfolio Value: {infos[agent_id]['portfolio_value']:.2f}")
                print(f"Policy Loss: {loss_infos[agent_id]['policy_loss']:.4f}")
                print(f"Value Loss: {loss_infos[agent_id]['value_loss']:.4f}")
            print("\n-------------------")
    
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