"""TD3 Agent: Twin Delayed DDPG via Stable-Baselines3 wrapped in BaseAgent.

Wraps SB3's TD3 implementation. TD3 improves on DDPG with:
- Twin Q-networks (takes minimum → reduces overestimation)
- Delayed policy updates (every policy_delay steps)
- Target policy smoothing (adds noise to target actions)

In the ensemble, TD3 serves as the *aggressive* agent with higher
learning rate and larger buffer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from agents.base.base_agent import BaseAgent


class _SpaceEnv(gym.Env):
    """Minimal Gymnasium env that exposes the target spaces for SB3 init."""
    metadata: dict = {}

    def __init__(self, obs_space: gym.spaces.Space, act_space: gym.spaces.Space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

logger = logging.getLogger(__name__)


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG agent backed by stable_baselines3.TD3."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 200_000,
        batch_size: int = 256,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        learning_starts: int = 1000,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.learning_starts = learning_starts
        self._device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Flatten 2D obs for MlpPolicy
        self._obs_is_2d = len(observation_space.shape) == 2
        if self._obs_is_2d:
            flat_dim = int(np.prod(observation_space.shape))
            self._flat_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
            )
        else:
            self._flat_obs_space = observation_space

        self._model = None
        self._step_count = 0
        self._drift_callback = None  # set externally via train_pipeline

        unused = [k for k in kwargs if k not in ("type", "strategy")]
        if unused:
            logger.warning("TD3Agent ignoring unused config keys: %s", unused)

    # ------------------------------------------------------------------
    # Lazy model creation
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from stable_baselines3 import TD3 as SB3_TD3
        except ImportError as e:
            raise ImportError(
                "stable_baselines3 is required for TD3Agent. "
                "Install with: pip install stable-baselines3"
            ) from e

        self._model = SB3_TD3(
            policy="MlpPolicy",
            env=_SpaceEnv(self._flat_obs_space, self.action_space),
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            learning_starts=self.learning_starts,
            device=self._device_str,
        )
        logger.info(
            "TD3Agent initialized — obs=%s, act=%s, device=%s",
            self._flat_obs_space.shape,
            self.action_space.shape,
            self._device_str,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_is_2d and obs.ndim == 2:
            return obs.flatten().astype(np.float32)
        return np.asarray(obs, dtype=np.float32)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self._ensure_model()
        flat = self._flatten_obs(state)
        action, _ = self._model.predict(flat, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def _update(self, state, action, reward, next_state, done, info=None) -> Dict[str, float]:
        self._ensure_model()
        flat_s = self._flatten_obs(state)
        flat_ns = self._flatten_obs(next_state)

        self._model.replay_buffer.add(
            obs=flat_s.reshape(1, -1),
            next_obs=flat_ns.reshape(1, -1),
            action=np.asarray(action).reshape(1, -1),
            reward=np.array([float(reward)]),
            done=np.array([bool(done)]),
            infos=[info or {}],
        )
        self._step_count += 1

        metrics: Dict[str, float] = {}
        if self._step_count >= self.learning_starts and self._step_count % self.batch_size == 0:
            self._model.train(gradient_steps=1, batch_size=self.batch_size)
            metrics["td3_step"] = float(self._step_count)

        return metrics

    def train(self, env, total_timesteps: int = 10000, batch_size: int = 64) -> Dict[str, Any]:
        self._ensure_model()
        self._model.set_env(env)
        callbacks = []
        if self._drift_callback is not None:
            callbacks.append(self._drift_callback)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
        )
        return {"total_timesteps": total_timesteps}

    def update_if_buffer_ready(self) -> Optional[Dict[str, float]]:
        self._ensure_model()
        if self._step_count >= self.learning_starts:
            self._model.train(gradient_steps=1, batch_size=self.batch_size)
            return {"td3_update": 1.0}
        return None

    def save(self, path: str) -> None:
        self._ensure_model()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        logger.info("TD3Agent saved to %s", path)

    def load(self, path: str) -> None:
        try:
            from stable_baselines3 import TD3 as SB3_TD3
        except ImportError as e:
            raise ImportError("stable_baselines3 required") from e
        self._model = SB3_TD3.load(path, device=self._device_str)
        logger.info("TD3Agent loaded from %s", path)
