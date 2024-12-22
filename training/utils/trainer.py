"""Distributed training utilities"""

import os
import ray
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from training.utils.mlflow_manager import MLflowManager
from training.utils.ray_manager import RayManager, RayConfig
from envs.trading_env import TradingEnvironment
from agents.strategies.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for distributed training"""

    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    hidden_size: int = 256
    num_parallel: int = 4
    initial_balance: float = 10000.0
    trading_fee: float = 0.001
    window_size: int = 20
    tune_trials: int = 10
    experiment_name: str = "trading_bot"


class DistributedTrainer:
    """Distributed trainer for trading agents"""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize Ray
        ray_config = RayConfig(num_cpus=config.num_parallel)
        self.ray_manager = RayManager(ray_config)

        # Initialize MLflow
        self.mlflow_manager = MLflowManager(config.experiment_name)

        # Create storage directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def create_env(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment

        Args:
            data: Market data

        Returns:
            Trading environment instance
        """
        return TradingEnvironment(
            df=data,
            initial_balance=self.config.initial_balance,
            trading_fee=self.config.trading_fee,
            window_size=self.config.window_size,
        )

    def create_agent(self, env: TradingEnvironment) -> PPOAgent:
        """Create PPO agent

        Args:
            env: Trading environment

        Returns:
            PPO agent instance
        """
        return PPOAgent(
            env=env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            hidden_size=self.config.hidden_size,
        )

    def train(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Train agent with distributed processing

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Dictionary of training metrics
        """
        try:
            with self.mlflow_manager:
                # Log parameters
                self.mlflow_manager.log_params(self.config.__dict__)

                # Create environment and agent
                env = self.create_env(train_data)
                agent = self.create_agent(env)

                # Train in parallel
                metrics = []
                for epoch in range(self.config.num_epochs):
                    # Train one epoch
                    train_metrics = agent.train(
                        env,
                        total_timesteps=len(train_data),
                        batch_size=self.config.batch_size,
                    )

                    # Evaluate on validation set
                    val_env = self.create_env(val_data)
                    val_metrics = self.evaluate(agent, val_env)

                    # Combine metrics
                    epoch_metrics = {
                        **train_metrics,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    metrics.append(epoch_metrics)

                    # Log metrics
                    self.mlflow_manager.log_metrics(epoch_metrics, step=epoch)

                # Calculate final metrics
                final_metrics = {
                    k: np.mean([m[k] for m in metrics])
                    for k in metrics[0].keys()
                }

                return final_metrics

        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def evaluate(
        self, agent: PPOAgent, env: TradingEnvironment
    ) -> Dict[str, float]:
        """Evaluate agent performance

        Args:
            agent: PPO agent
            env: Trading environment

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            obs, _ = env.reset()
            done = False
            total_reward = 0
            portfolio_values = []

            while not done:
                action = agent.get_action(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                portfolio_values.append(info["portfolio_value"])

            return {
                "reward": total_reward,
                "portfolio_value": portfolio_values[-1],
                "total_trades": len(env.trades),
                "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_values),
            }

        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise

    def tune(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run hyperparameter tuning

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Best hyperparameter configuration
        """
        try:
            from ray import tune

            def objective(config):
                """Objective function for tuning"""
                # Update trainer config with trial config
                self.config.__dict__.update(config)

                # Train and evaluate
                metrics = self.train(train_data, val_data)

                # Report metrics
                tune.report(
                    reward=metrics["val_reward"],
                    portfolio_value=metrics["val_portfolio_value"],
                    sharpe_ratio=metrics["val_sharpe_ratio"],
                )

            # Define search space
            search_space = {
                "learning_rate": tune.loguniform(1e-4, 1e-2),
                "hidden_size": tune.choice([32, 64, 128, 256]),
                "gamma": tune.uniform(0.9, 0.999),
            }

            # Run tuning
            analysis = tune.run(
                objective,
                config=search_space,
                num_samples=self.config.tune_trials,
                resources_per_trial={"cpu": 1},
                metric="sharpe_ratio",
                mode="max",
            )

            return analysis.best_config

        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save model to disk

        Args:
            path: Path to save model
        """
        try:
            if hasattr(self, "agent"):
                torch.save(
                    {
                        "network_state_dict": self.agent.network.state_dict(),
                        "value_network_state_dict": self.agent.value_network.state_dict(),
                        "optimizer_state_dict": self.agent.optimizer.state_dict(),
                        "config": self.config.__dict__,
                    },
                    path,
                )

                # Log model to MLflow
                self.mlflow_manager.log_artifact(path)

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> PPOAgent:
        """Load model from disk

        Args:
            path: Path to load model from

        Returns:
            Loaded PPO agent
        """
        try:
            checkpoint = torch.load(path)

            # Create environment and agent with saved config
            config = TrainingConfig(**checkpoint["config"])
            env = self.create_env(pd.DataFrame())  # Empty DataFrame for now
            agent = self.create_agent(env)

            # Load state dicts
            agent.network.load_state_dict(checkpoint["network_state_dict"])
            agent.value_network.load_state_dict(
                checkpoint["value_network_state_dict"]
            )
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return agent

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "ray_manager"):
            self.ray_manager.shutdown()

    def _calculate_sharpe_ratio(self, portfolio_values: List[float]) -> float:
        """Calculate Sharpe ratio

        Args:
            portfolio_values: List of portfolio values

        Returns:
            Sharpe ratio
        """
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) < 2:
            return 0.0
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
