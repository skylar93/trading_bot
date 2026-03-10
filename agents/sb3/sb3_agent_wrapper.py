"""
SB3AgentWrapper: adapts Stable-Baselines3 models to the project's BaseAgent interface.

Supported algorithms: PPO, SAC, TD3, A2C
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from agents.strategies.base_agent import BaseAgent
from agents.sb3.feature_extractors import LSTMTradingExtractor, TradingWindowExtractor

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependency at module level
def _get_algo_class(algo_type: str):
    algo_type = algo_type.lower().replace("sb3_", "")
    if algo_type == "ppo":
        from stable_baselines3 import PPO
        return PPO
    elif algo_type == "sac":
        from stable_baselines3 import SAC
        return SAC
    elif algo_type == "td3":
        from stable_baselines3 import TD3
        return TD3
    elif algo_type == "a2c":
        from stable_baselines3 import A2C
        return A2C
    else:
        raise ValueError(f"Unknown SB3 algorithm: '{algo_type}'. Choose from: ppo, sac, td3, a2c")


_EXTRACTOR_MAP = {
    "conv1d": TradingWindowExtractor,
    "lstm": LSTMTradingExtractor,
}


class SB3AgentWrapper(BaseAgent):
    """
    Wraps a Stable-Baselines3 model behind the project's BaseAgent interface.

    Usage:
        agent = SB3AgentWrapper("ppo", obs_space, act_space)
        agent.train(env, total_timesteps=100_000)
        action = agent.get_action(obs)
        agent.save("checkpoints/ppo_model")
        agent.load("checkpoints/ppo_model")
    """

    def __init__(
        self,
        algo_type: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        feature_extractor: Optional[str] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        sb3_params: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        verbose: int = 0,
    ):
        """
        Args:
            algo_type: One of "ppo", "sac", "td3", "a2c" (or "sb3_ppo", etc.)
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            feature_extractor: "conv1d", "lstm", or None (uses SB3 default MlpPolicy)
            feature_extractor_kwargs: Extra kwargs for the feature extractor
            sb3_params: Algorithm-specific SB3 hyperparameters
            device: PyTorch device ("auto", "cpu", "cuda")
            verbose: SB3 verbosity (0=quiet, 1=info, 2=debug)
        """
        super().__init__(observation_space, action_space)

        self.algo_type = algo_type.lower().replace("sb3_", "")
        self._algo_class = _get_algo_class(algo_type)
        self._sb3_params = sb3_params or {}
        self._device = device
        self._verbose = verbose
        self._feature_extractor = feature_extractor
        self._feature_extractor_kwargs = feature_extractor_kwargs or {}

        # Determine policy type: use MlpPolicy by default, or CnnPolicy if using
        # a custom extractor that treats obs as image-like (we still use MlpPolicy
        # since our extractors are custom BaseFeaturesExtractor subclasses).
        self._policy = "MlpPolicy"

        # Build policy_kwargs for custom feature extractor
        self._policy_kwargs: Dict[str, Any] = {}
        if feature_extractor and feature_extractor in _EXTRACTOR_MAP:
            self._policy_kwargs["features_extractor_class"] = _EXTRACTOR_MAP[feature_extractor]
            fe_kwargs = {"features_dim": 128}
            fe_kwargs.update(self._feature_extractor_kwargs)
            self._policy_kwargs["features_extractor_kwargs"] = fe_kwargs

        # Model is created lazily in train() when env is available
        self.model = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call train() or load() first."
            )
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def train(
        self,
        env,
        total_timesteps: int = 100_000,
        callbacks: Optional[Union[List, Any]] = None,
        reset_num_timesteps: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the SB3 model.

        Args:
            env: Gymnasium-compatible environment or VecEnv
            total_timesteps: Total environment steps to train for
            callbacks: SB3 callback(s) for logging, checkpointing, etc.
            reset_num_timesteps: Whether to reset the step counter

        Returns:
            Dict with training info (total_timesteps)
        """
        if self.model is None:
            self._create_model(env)
        else:
            self.model.set_env(env)

        logger.info(
            f"Training {self.algo_type.upper()} for {total_timesteps:,} timesteps "
            f"(device={self._device})"
        )
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
        return {"total_timesteps": total_timesteps, "algo_type": self.algo_type}

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, env=None) -> None:
        self.model = self._algo_class.load(path, env=env, device=self._device)
        logger.info(f"Model loaded from {path}")

    def train_step(self, experience: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError(
            "SB3AgentWrapper does not support step-wise training. "
            "Use train(env, total_timesteps) instead."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_model(self, env) -> None:
        """Instantiate the SB3 model with the provided environment."""
        kwargs = dict(
            policy=self._policy,
            env=env,
            device=self._device,
            verbose=self._verbose,
            **self._sb3_params,
        )
        if self._policy_kwargs:
            kwargs["policy_kwargs"] = self._policy_kwargs

        self.model = self._algo_class(**kwargs)
        logger.info(
            f"Created {self.algo_type.upper()} model "
            f"(feature_extractor={self._feature_extractor or 'default'})"
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ) -> "SB3AgentWrapper":
        """
        Construct an SB3AgentWrapper from a config dict.

        Expected config keys:
            algo_type (str): e.g. "sb3_ppo"
            feature_extractor (str, optional): "conv1d", "lstm", or None
            feature_extractor_kwargs (dict, optional)
            sb3_params (dict, optional): algo-specific params
            device (str, optional): default "auto"
            verbose (int, optional): default 0
        """
        return cls(
            algo_type=config.get("algo_type", "ppo"),
            observation_space=observation_space,
            action_space=action_space,
            feature_extractor=config.get("feature_extractor"),
            feature_extractor_kwargs=config.get("feature_extractor_kwargs", {}),
            sb3_params=config.get("sb3_params", {}),
            device=config.get("device", "auto"),
            verbose=config.get("verbose", 0),
        )
