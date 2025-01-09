"""Hyperparameter optimization using Ray Tune"""

import os
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.search.optuna import OptunaSearch
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import mlflow
import shutil
import time

from envs.trading_env import TradingEnvironment
from agents.strategies.single.ppo_agent import PPOAgent
from training.train import train_agent
from training.utils.unified_mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)


class MinimalTuner:
    """Minimal hyperparameter tuner using Ray Tune"""

    def __init__(self, df: pd.DataFrame, mlflow_experiment: str = None):
        """Initialize tuner

        Args:
            df: DataFrame containing market data
            mlflow_experiment: Optional MLflow experiment name
        """
        self.df = df

        # Initialize MLflow if experiment provided
        if mlflow_experiment:
            # End any active runs
            try:
                experiment = mlflow.get_experiment_by_name(mlflow_experiment)
                if experiment:
                    # End any active runs
                    for run in mlflow.search_runs([experiment.experiment_id]):
                        if run.info.status == "RUNNING":
                            mlflow.end_run(run_id=run.info.run_id)
                    time.sleep(0.1)  # Give MLflow time to clean up
            except Exception as e:
                logger.warning(f"Error cleaning up active runs: {str(e)}")

            # Initialize MLflow manager
            self.mlflow_manager = MLflowManager(
                experiment_name=mlflow_experiment
            )
        else:
            self.mlflow_manager = None

        # Split data into train/val
        train_size = int(len(df) * 0.8)
        self.train_data = df[:train_size]
        self.val_data = df[train_size:]

        # Default search space
        self.search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-2),
            "hidden_size": tune.choice([32, 64, 128, 256]),
            "gamma": tune.uniform(0.9, 0.999),
            "gae_lambda": tune.uniform(0.9, 0.999),
            "clip_epsilon": tune.loguniform(1e-5, 1e-3),
            "c1": tune.uniform(0.5, 2.0),
            "c2": tune.uniform(0.001, 0.1),
            "initial_balance": tune.choice([10000, 100000]),
            "trading_fee": tune.loguniform(1e-3, 1e-2),
            "total_timesteps": tune.choice([10000, 50000]),
        }

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True)

        # Create storage directory with absolute path
        self.storage_path = os.path.abspath(
            os.path.join(os.getcwd(), "ray_results")
        )
        os.makedirs(self.storage_path, exist_ok=True)

    def objective(
        self, config: Dict[str, Any], session: Optional[Any] = None
    ) -> None:
        """Objective function for Ray Tune"""
        try:
            # Convert flat config to nested config
            config_dict = {
                "env": {
                    "initial_balance": config["initial_balance"],
                    "trading_fee": config["trading_fee"],
                    "window_size": 10,  # Reduced window size
                    "max_position_size": 1.0,
                },
                "model": {
                    "hidden_size": config["hidden_size"],
                    "learning_rate": config["learning_rate"],
                    "gamma": config["gamma"],
                    "gae_lambda": config["gae_lambda"],
                    "clip_epsilon": config["clip_epsilon"],
                    "c1": config["c1"],
                    "c2": config["c2"],
                    "batch_size": 32,  # Reduced batch size
                    "n_epochs": 1,  # Single epoch
                },
                "training": {"total_timesteps": config["total_timesteps"]},
            }

            # Train agent
            agent = train_agent(self.train_data, self.val_data, config_dict)

            # Evaluate on validation set
            metrics = self.evaluate_config(config_dict, agent)

            # Ensure score is always reported
            if "score" not in metrics:
                metrics["score"] = float("-inf")

            # Report metrics to Ray Tune
            if session:
                session.report(metrics)

            # Log to MLflow if available
            if self.mlflow_manager:
                with self.mlflow_manager:
                    # Log hyperparameters with flattened names
                    mlflow_params = {
                        "learning_rate": config["learning_rate"],
                        "hidden_size": config["hidden_size"],
                        "gamma": config["gamma"],
                        "gae_lambda": config["gae_lambda"],
                        "clip_epsilon": config["clip_epsilon"],
                        "c1": config["c1"],
                        "c2": config["c2"],
                        "initial_balance": config["initial_balance"],
                        "trading_fee": config["trading_fee"],
                        "total_timesteps": config["total_timesteps"],
                    }
                    self.mlflow_manager.log_params(mlflow_params)

                    # Log metrics
                    self.mlflow_manager.log_metrics(metrics)

        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            # Report error metrics with score
            error_metrics = {
                "score": float("-inf"),
                "sharpe_ratio": float("-inf"),
                "total_return": float("-inf"),
                "max_drawdown": float("-inf"),
                "error": str(e),
            }
            if session:
                session.report(error_metrics)

    def evaluate_config(
        self, config: Dict[str, Any], agent: Optional[PPOAgent] = None
    ) -> Dict[str, float]:
        """Evaluate a configuration on validation data"""
        try:
            # Create validation environment
            val_env = TradingEnvironment(
                df=self.val_data,
                initial_balance=config["env"]["initial_balance"],
                trading_fee=config["env"]["trading_fee"],
                window_size=config["env"]["window_size"],
            )

            # Create agent if not provided
            if agent is None:
                agent = PPOAgent(
                    observation_space=val_env.observation_space,
                    action_space=val_env.action_space,
                    learning_rate=config["model"]["learning_rate"],
                    gamma=config["model"]["gamma"],
                    gae_lambda=config["model"]["gae_lambda"],
                    clip_epsilon=config["model"]["clip_epsilon"],
                    c1=config["model"]["c1"],
                    c2=config["model"]["c2"],
                )

            # Run evaluation episode
            state, _ = val_env.reset()
            done = False
            truncated = False
            total_reward = 0
            portfolio_values = []
            max_steps = min(
                20, len(self.val_data) - config["env"]["window_size"]
            )  # Limit to 20 steps
            step_count = 0

            try:
                while not (done or truncated) and step_count < max_steps:
                    action = agent.get_action(state)
                    state, reward, done, truncated, info = val_env.step(action)
                    total_reward += reward
                    portfolio_values.append(info["portfolio_value"])
                    step_count += 1

                    # Force truncate if we've reached max steps
                    if step_count >= max_steps:
                        truncated = True
            except Exception as e:
                logger.error(f"Error during evaluation episode: {str(e)}")
                return {
                    "score": float("-inf"),
                    "sharpe_ratio": float("-inf"),
                    "total_return": float("-inf"),
                    "max_drawdown": float("-inf"),
                }

            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
                max_drawdown = (
                    np.min(portfolio_values) / np.max(portfolio_values) - 1
                )
                total_return = portfolio_values[-1] / portfolio_values[0] - 1
            else:
                sharpe_ratio = float("-inf")
                max_drawdown = float("-inf")
                total_return = float("-inf")

            # Calculate score
            score = (
                0.4 * sharpe_ratio
                + 0.4 * total_return
                + 0.2
                * (1 + max_drawdown)  # Convert drawdown to positive score
            )

            return {
                "score": score,
                "sharpe_ratio": float(sharpe_ratio),
                "total_return": float(total_return),
                "max_drawdown": float(max_drawdown),
            }

        except Exception as e:
            logger.error(f"Error in evaluate_config: {str(e)}")
            return {
                "score": float("-inf"),
                "sharpe_ratio": float("-inf"),
                "total_return": float("-inf"),
                "max_drawdown": float("-inf"),
            }

    def optimize(
        self,
        search_space: Dict = None,
        num_samples: int = 10,
        max_concurrent: int = 4,
        max_epochs: int = None,
    ):
        """Run hyperparameter optimization

        Args:
            search_space: Search space dictionary (optional, uses default if not provided)
            num_samples: Number of trials to run
            max_concurrent: Maximum number of concurrent trials
            max_epochs: Maximum number of epochs per trial
        """
        try:
            # Use provided search space or default
            space = (
                search_space if search_space is not None else self.search_space
            )

            # Set up Optuna search algorithm
            search_alg = OptunaSearch(space, metric="score", mode="max")

            # Run optimization
            tuner = tune.Tuner(
                self.objective,
                tune_config=tune.TuneConfig(
                    metric="score",
                    mode="max",
                    num_samples=num_samples,
                    max_concurrent_trials=max_concurrent,
                    search_alg=search_alg,
                    time_budget_s=30,  # 30 second timeout
                ),
                run_config=ray.air.RunConfig(
                    storage_path=self.storage_path,
                    name="ppo_tuning",
                    stop={
                        "training_iteration": (
                            max_epochs if max_epochs else 100
                        ),
                        "done": True,
                    },
                ),
            )

            # Execute trials
            try:
                results = tuner.fit()
            except ValueError as e:
                if (
                    "Trial returned a result which did not include the specified metric(s)"
                    in str(e)
                ):
                    logger.warning(
                        "Metric validation error, returning best trial so far"
                    )
                    return None
                raise

            # Get best trial
            best_trial = results.get_best_result(metric="score", mode="max")
            if best_trial is None:
                logger.warning("No successful trials found")
                return None

            # Log results
            if best_trial.metrics:
                logger.info(f"Best trial config: {best_trial.config}")
                logger.info(f"Best trial final metrics: {best_trial.metrics}")
            else:
                logger.warning("Best trial has no metrics")

            return best_trial

        except Exception as e:
            logger.error(f"Error in optimize: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up Ray
            if ray.is_initialized():
                ray.shutdown()

            # Clean up storage directory
            if os.path.exists(self.storage_path):
                shutil.rmtree(self.storage_path)

            # Clean up MLflow
            if self.mlflow_manager:
                # End any active runs
                if mlflow.active_run():
                    mlflow.end_run()
                    time.sleep(0.1)  # Give MLflow time to clean up

                # End any active runs in the experiment
                try:
                    experiment = mlflow.get_experiment_by_name(
                        self.mlflow_manager.experiment_name
                    )
                    if experiment:
                        for run in mlflow.search_runs(
                            [experiment.experiment_id]
                        ):
                            if run.info.status == "RUNNING":
                                mlflow.end_run(run_id=run.info.run_id)
                except Exception as e:
                    logger.warning(f"Error cleaning up active runs: {str(e)}")

                # Call MLflow manager cleanup
                self.mlflow_manager.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
