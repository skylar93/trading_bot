import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import mlflow
from datetime import datetime, timedelta
import torch
import logging
from pathlib import Path

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.single.ppo_agent import PPOAgent
from training.train import fetch_training_data

logger = logging.getLogger(__name__)

def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def evaluate_agents(
    env: MultiAgentTradingEnv,
    agents: Dict[str, PPOAgent],
    num_episodes: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate agents' performance.
    
    Args:
        env: Trading environment
        agents: Dictionary of agents to evaluate
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary containing evaluation metrics for each agent
    """
    eval_metrics = {
        agent_id: {
            "mean_reward": 0.0,
            "mean_portfolio_value": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        for agent_id in agents.keys()
    }
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        dones = {agent_id: False for agent_id in agents.keys()}
        done = False
        
        episode_rewards = {agent_id: 0 for agent_id in agents.keys()}
        portfolio_values = {agent_id: [] for agent_id in agents.keys()}
        
        while not done:
            actions = {}
            for agent_id, agent in agents.items():
                if not dones[agent_id]:
                    action = agent.get_action(observations[agent_id], deterministic=True)
                    actions[agent_id] = action
            
            next_observations, rewards, dones, truncated, infos = env.step(actions)
            
            for agent_id in agents.keys():
                if not dones[agent_id]:
                    episode_rewards[agent_id] += rewards[agent_id]
                    portfolio_values[agent_id].append(infos[agent_id]["portfolio_value"])
            
            observations = next_observations
            done = all(dones.values())
        
        # Update metrics
        for agent_id in agents.keys():
            eval_metrics[agent_id]["mean_reward"] += episode_rewards[agent_id]
            eval_metrics[agent_id]["mean_portfolio_value"] += portfolio_values[agent_id][-1]
            
            # Calculate Sharpe ratio
            returns = np.diff(portfolio_values[agent_id]) / portfolio_values[agent_id][:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
            eval_metrics[agent_id]["sharpe_ratio"] += sharpe
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_values[agent_id])
            drawdown = (peak - portfolio_values[agent_id]) / peak
            max_drawdown = np.max(drawdown)
            eval_metrics[agent_id]["max_drawdown"] = max(
                eval_metrics[agent_id]["max_drawdown"],
                max_drawdown
            )
    
    # Average metrics
    for agent_id in agents.keys():
        eval_metrics[agent_id]["mean_reward"] /= num_episodes
        eval_metrics[agent_id]["mean_portfolio_value"] /= num_episodes
        eval_metrics[agent_id]["sharpe_ratio"] /= num_episodes
    
    return eval_metrics

def train_multi_agent_system(
    env: MultiAgentTradingEnv,
    agents: Dict[str, PPOAgent],
    num_episodes: int = 1000,
    save_path: str = "models/multi_agent",
    save_freq: int = 100,
    eval_freq: int = 50,
    experiment_name: str = "trading_bot_multi_agent_training",
) -> Dict[str, Dict[str, List[float]]]:
    """Train multiple agents simultaneously"""

    # Setup logging
    setup_logging()
    
    # Create directory for saving models
    save_path = Path(save_path)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created or verified model save directory: {save_path}")
    except Exception as e:
        logger.error(f"Failed to create model save directory: {str(e)}")
        raise

    # Initialize metrics tracking for each agent
    metrics = {
        agent_id: {
            "episode_rewards": [],
            "portfolio_values": [],
            "policy_losses": [],
            "value_losses": [],
        }
        for agent_id in agents.keys()
    }

    # Set up MLflow tracking
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log configurations
        mlflow.log_params({
            "num_episodes": num_episodes,
            "num_agents": len(agents),
            "save_freq": save_freq,
            "eval_freq": eval_freq,
        })
        
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
                            "state": observations[agent_id],
                            "action": actions[agent_id],
                            "reward": rewards[agent_id],
                            "next_state": next_observations[agent_id],
                            "done": dones[agent_id],
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
                    metrics[agent_id]["episode_rewards"].append(episode_rewards[agent_id])
                    metrics[agent_id]["portfolio_values"].append(infos[agent_id]["portfolio_value"])
                    metrics[agent_id]["policy_losses"].append(loss_info["policy_loss"])
                    metrics[agent_id]["value_losses"].append(loss_info["value_loss"])

                    # Log metrics to MLflow
                    mlflow.log_metrics(
                        {
                            f"{agent_id}_reward": episode_rewards[agent_id],
                            f"{agent_id}_portfolio_value": infos[agent_id]["portfolio_value"],
                            f"{agent_id}_policy_loss": loss_info["policy_loss"],
                            f"{agent_id}_value_loss": loss_info["value_loss"],
                        },
                        step=episode,
                    )

            # Evaluate periodically
            if episode > 0 and episode % eval_freq == 0:
                eval_metrics = evaluate_agents(env, agents)
                for agent_id, agent_metrics in eval_metrics.items():
                    for metric_name, value in agent_metrics.items():
                        mlflow.log_metric(
                            f"{agent_id}_eval_{metric_name}",
                            value,
                            step=episode
                        )
                logger.info(f"\nEvaluation metrics at episode {episode}:")
                for agent_id, agent_metrics in eval_metrics.items():
                    logger.info(f"\n{agent_id}:")
                    for metric_name, value in agent_metrics.items():
                        logger.info(f"{metric_name}: {value:.4f}")

            # Save models periodically
            if (episode > 0 and episode % save_freq == 0) or episode == num_episodes - 1:
                for agent_id, agent in agents.items():
                    model_path = save_path / f"{agent_id}_episode_{episode}.pt"
                    torch.save({
                        "policy_state_dict": agent.network.state_dict(),
                        "value_state_dict": agent.value_network.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict()
                    }, model_path)

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
            "id": "momentum_trader",
            "strategy": "momentum",
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
        },
        {
            "id": "mean_reversion_trader",
            "strategy": "mean_reversion",
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
        },
        {
            "id": "market_maker",
            "strategy": "market_making",
            "initial_balance": 10000.0,
            "fee_multiplier": 0.8,  # Market makers often get fee discounts
        },
    ]

    # Create environment
    env = MultiAgentTradingEnv(df, agent_configs)

    # Create agents
    agents = {
        config["id"]: PPOAgent(
            env.observation_spaces[config["id"]],
            env.action_spaces[config["id"]],
        )
        for config in agent_configs
    }

    # Train agents
    metrics = train_multi_agent_system(env, agents)

    # Save final models
    for agent_id, agent in agents.items():
        agent.save(f"models/multi_agent/{agent_id}_final.pt")

    # Plot results
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        "episode_rewards",
        "portfolio_values",
        "policy_losses",
        "value_losses",
    ]
    fig, axes = plt.subplots(
        len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot))
    )

    for i, metric in enumerate(metrics_to_plot):
        for agent_id in agents.keys():
            axes[i].plot(metrics[agent_id][metric], label=agent_id)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("multi_agent_training_results.png")
    plt.close()
