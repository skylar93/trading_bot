import os
import numpy as np
import torch
import logging
from typing import Dict, Any, List
import mlflow
from datetime import datetime

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.multi.multi_agent_manager import MultiAgentManager
from agents.strategies.multi.momentum_ppo_agent import MomentumPPOAgent
from agents.strategies.multi.mean_reversion_ppo_agent import MeanReversionPPOAgent

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

def create_env_config() -> Dict[str, Any]:
    """Create environment configuration."""
    return {
        "num_agents": 2,  # Momentum and Mean Reversion
        "initial_balance": 10000,
        "trading_fee": 0.001,
        "window_size": 20,
        "reward_scaling": 1.0,
        "max_position_size": 1.0
    }

def create_agent_configs() -> List[Dict[str, Any]]:
    """Create configurations for each agent."""
    base_config = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "c1": 1.0,  # Value loss coefficient
        "c2": 0.01,  # Entropy coefficient
        "batch_size": 64,
        "n_epochs": 10,
        "target_kl": 0.015
    }
    
    momentum_config = base_config.copy()
    momentum_config.update({
        "id": "momentum",
        "strategy": "momentum",
        "momentum_window": 10,
        "momentum_threshold": 0.02
    })
    
    mean_reversion_config = base_config.copy()
    mean_reversion_config.update({
        "id": "mean_reversion",
        "strategy": "mean_reversion",
        "rsi_window": 14,
        "bb_window": 20,
        "bb_std": 2.0
    })
    
    return [momentum_config, mean_reversion_config]

def train_multi_agent(
    num_episodes: int = 1000,
    save_freq: int = 100,
    eval_freq: int = 50,
    model_dir: str = "models/multi_agent",
    experiment_name: str = "multi_agent_training"
) -> None:
    """
    Train multiple agents simultaneously.
    
    Args:
        num_episodes: Number of episodes to train
        save_freq: Frequency of saving model checkpoints
        eval_freq: Frequency of evaluation
        model_dir: Directory to save model checkpoints
        experiment_name: Name of the MLflow experiment
    """
    # Setup logging and model directory
    setup_logging()
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    
    # Create environment and agents
    env_config = create_env_config()
    env = MultiAgentTradingEnv(env_config)
    
    agent_configs = create_agent_configs()
    manager = MultiAgentManager(agent_configs)
    
    with mlflow.start_run():
        # Log configurations
        mlflow.log_params({
            "num_episodes": num_episodes,
            "num_agents": env_config["num_agents"],
            "initial_balance": env_config["initial_balance"],
            "trading_fee": env_config["trading_fee"]
        })
        
        # Training loop
        for episode in range(num_episodes):
            obs = env.reset()
            episode_rewards = {agent_id: 0.0 for agent_id in manager.agents.keys()}
            done = False
            step = 0
            
            while not done:
                # Get actions from all agents
                actions = manager.act(obs)
                
                # Environment step
                next_obs, rewards, done, info = env.step(actions)
                
                # Prepare experiences for each agent
                experiences = {}
                for agent_id in actions.keys():
                    experiences[agent_id] = {
                        "state": obs[agent_id],
                        "action": actions[agent_id],
                        "reward": rewards[agent_id],
                        "next_state": next_obs[agent_id],
                        "done": done
                    }
                    episode_rewards[agent_id] += rewards[agent_id]
                
                # Train agents
                metrics = manager.train_step(experiences)
                
                # Update observations
                obs = next_obs
                step += 1
            
            # Log episode metrics
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            for agent_id, reward in episode_rewards.items():
                logger.info(f"{agent_id.capitalize()} Agent - Total Reward: {reward:.2f}")
                mlflow.log_metric(f"{agent_id}_reward", reward, step=episode)
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                checkpoint_dir = os.path.join(model_dir, f"episode_{episode + 1}")
                manager.save(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                eval_metrics = manager.evaluate(env, num_episodes=5)
                for agent_id, metrics in eval_metrics.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(
                            f"{agent_id}_{metric_name}_eval",
                            value,
                            step=episode
                        )
                logger.info(f"Evaluation metrics: {eval_metrics}")

if __name__ == "__main__":
    train_multi_agent()
