"""
SB3 Callbacks for training monitoring, checkpointing, and evaluation.
"""
import logging
import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class MLflowLoggingCallback(BaseCallback):
    """
    Log SB3 training metrics to MLflow every ``log_interval`` steps.

    Captures whatever SB3 puts in ``self.model.logger.name_to_value`` after
    each rollout / gradient update so we get policy_loss, value_loss, entropy,
    approx_kl, explained_variance, etc. without any monkey-patching.
    """

    def __init__(
        self,
        mlflow_manager,
        log_interval: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.mlflow_manager = mlflow_manager
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            metrics: dict = {}

            # Pull whatever SB3 logged internally
            for key, val in self.model.logger.name_to_value.items():
                try:
                    metrics[key] = float(val)
                except (TypeError, ValueError):
                    pass

            # Always log rollout reward stats if available
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                metrics["rollout/ep_rew_mean"] = float(np.mean(ep_rewards))
                metrics["rollout/ep_len_mean"] = float(np.mean(ep_lengths))

            if metrics and self.mlflow_manager is not None:
                try:
                    self.mlflow_manager.log_metrics(metrics, step=self.num_timesteps)
                except Exception as e:
                    logger.warning(f"MLflow logging failed at step {self.num_timesteps}: {e}")

        return True  # True = keep training


class SB3CheckpointCallback(BaseCallback):
    """
    Save the SB3 model every ``save_freq`` steps and optionally log to MLflow.

    Mirrors SB3's built-in CheckpointCallback but also records the checkpoint
    path as an MLflow artifact and tags the run with the step number.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        mlflow_manager=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mlflow_manager = mlflow_manager
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose:
                logger.info(f"Saved checkpoint to {path}.zip")
            if self.mlflow_manager is not None:
                try:
                    self.mlflow_manager.log_artifact(f"{path}.zip")
                    self.mlflow_manager.log_metrics(
                        {"checkpoint/step": self.num_timesteps},
                        step=self.num_timesteps,
                    )
                except Exception as e:
                    logger.warning(f"MLflow checkpoint logging failed: {e}")
        return True


class SB3EvalCallback(EvalCallback):
    """
    Thin wrapper around SB3's built-in EvalCallback that also pipes eval
    metrics into MLflow after each evaluation round.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        mlflow_manager=None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_save_path,
            verbose=verbose,
            **kwargs,
        )
        self.mlflow_manager = mlflow_manager

    def _on_step(self) -> bool:
        result = super()._on_step()
        # After each evaluation (parent sets last_mean_reward), log to MLflow
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.mlflow_manager is not None and self.last_mean_reward is not None:
                try:
                    self.mlflow_manager.log_metrics(
                        {
                            "eval/mean_reward": float(self.last_mean_reward),
                            "eval/is_best": float(self.last_mean_reward >= self.best_mean_reward),
                        },
                        step=self.num_timesteps,
                    )
                except Exception as e:
                    logger.warning(f"MLflow eval logging failed: {e}")
        return result
