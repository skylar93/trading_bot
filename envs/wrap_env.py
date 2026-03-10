import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union, List, Callable

try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None  # type: ignore[assignment]


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to range [-1, 1] with support for GPU and NaN handling"""

    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = torch.device(device)

        # Initialize running statistics
        self.is_vector_env = hasattr(env, "num_envs")
        
        # Initialize running statistics for each feature
        obs_shape = self.observation_space.shape
        self.running_mean = np.zeros(obs_shape[-1], dtype=np.float32)
        self.running_std = np.ones(obs_shape[-1], dtype=np.float32)
        self.count = 0
        self.eps = 1e-8

        # Update observation space to reflect normalized values
        self.observation_space = spaces.Box(
            low=-10,  # Allow slightly larger range for stability
            high=10,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def _update_stats(self, obs):
        """Update running statistics for normalization"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
            
        # Handle NaN and infinite values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure obs is 2D for consistent processing
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif obs.ndim == 3:
            # For image-like observations, flatten last two dimensions
            obs = obs.reshape(obs.shape[0], -1)
            
        batch_mean = obs.mean(axis=0)
        batch_std = obs.std(axis=0)
        batch_count = obs.shape[0]
        
        # Update running statistics using Welford's online algorithm
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / (self.count + batch_count)
        
        # Update variance
        delta2 = batch_mean - self.running_mean
        m_a = self.running_std * self.running_std * self.count
        m_b = batch_std * batch_std * batch_count
        M2 = m_a + m_b + delta * delta2 * self.count * batch_count / (self.count + batch_count)
        self.running_std = np.sqrt(M2 / (self.count + batch_count))
        
        self.count += batch_count

    def observation(self, obs):
        """Normalize observation"""
        # Update statistics
        self._update_stats(obs)
        
        if isinstance(obs, np.ndarray):
            # Handle NaN values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize using running statistics
            obs_normalized = (obs - self.running_mean) / (self.running_std + self.eps)
            
            # Clip to reasonable range
            obs_normalized = np.clip(obs_normalized, -10, 10)
            
            return obs_normalized.astype(np.float32)
            
        elif isinstance(obs, torch.Tensor):
            # Handle NaN values
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Convert running statistics to tensors
            running_mean = torch.FloatTensor(self.running_mean).to(obs.device)
            running_std = torch.FloatTensor(self.running_std).to(obs.device)
            
            # Normalize using running statistics
            obs_normalized = (obs - running_mean) / (running_std + self.eps)
            
            # Clip to reasonable range
            obs_normalized = torch.clamp(obs_normalized, -10, 10)
            
            return obs_normalized.to(dtype=torch.float32)
            
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    def reset(self, **kwargs):
        """Reset the environment and running statistics"""
        obs, info = self.env.reset(**kwargs)

        # Reset running statistics
        obs_shape = self.observation_space.shape
        self.running_mean = np.zeros(obs_shape[-1], dtype=np.float32)
        self.running_std = np.ones(obs_shape[-1], dtype=np.float32)
        self.count = 0

        return self.observation(obs), info


class StackObservation(gym.ObservationWrapper):
    """Stack observations to create a history of observations"""

    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size

        # Calculate new observation space shape
        old_shape = env.observation_space.shape
        if len(old_shape) != 2:
            raise ValueError(
                f"Expected 2D observation shape (window_size, features), got {old_shape}"
            )

        # New shape will be (window_size, features)
        # Keep original feature dimension
        new_shape = (old_shape[0], old_shape[1])

        # Update observation space
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=np.float32,
        )

        # Initialize observation stack
        self.obs_stack = None

    def reset(self, **kwargs):
        """Reset observation stack"""
        obs, info = self.env.reset(**kwargs)

        # Initialize stack with copies of the initial observation
        self.obs_stack = obs  # Just use the initial observation as is
        return self.obs_stack, info

    def observation(self, obs):
        """Process observation to maintain correct feature dimension"""
        if self.obs_stack is None:
            self.obs_stack = obs
        else:
            # Update the observation while maintaining the feature dimension
            self.obs_stack = obs
        return self.obs_stack


class ClipActions(gym.ActionWrapper):
    """Clip actions to valid range"""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class RecordEpisodeStats(gym.Wrapper):
    """Record episode statistics"""

    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_return = 0
        self.current_length = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        self.current_return += reward
        self.current_length += 1

        if terminated or truncated:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            info["episode"] = {
                "r": self.current_return,
                "l": self.current_length,
                "returns": self.episode_returns,
                "lengths": self.episode_lengths,
            }

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_return = 0
        self.current_length = 0
        return self.env.reset(**kwargs)


class MLflowLoggingWrapper(gym.Wrapper):
    """Log environment metrics to MLflow"""

    def __init__(self, env, experiment_name="trading_bot"):
        super().__init__(env)
        self.experiment_name = experiment_name
        if _mlflow is not None:
            _mlflow.set_experiment(experiment_name)
        self.episode_count = 0
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset with MLflow logging"""
        obs, info = self.env.reset(**kwargs)

        # Log reset metrics
        if _mlflow is not None:
            _mlflow.log_metrics(
                {
                    "initial_balance": info.get("balance", 0),
                    "initial_price": info.get("current_price", 0),
                },
                step=self.episode_count,
            )

        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        self.step_count += 1

        # Log step metrics
        if _mlflow is not None:
            _mlflow.log_metrics(
                {
                    "step_reward": reward,
                    "portfolio_value": info.get("portfolio_value", 0),
                    "position_size": info.get("position_size", 0),
                },
                step=self.step_count,
            )

        if terminated or truncated:
            self.episode_count += 1
            # Log episode metrics
            if _mlflow is not None:
                _mlflow.log_metrics(
                    {
                        "episode_return": info["episode"]["r"],
                        "episode_length": info["episode"]["l"],
                        "total_trades": info.get("total_trades", 0),
                        "win_rate": info.get("win_rate", 0),
                    },
                    step=self.episode_count,
                )

        return observation, reward, terminated, truncated, info


def make_env(env, normalize=True, stack_size=4):
    """Create environment with specified wrappers (legacy)."""
    env = ClipActions(env)
    if normalize:
        env = NormalizeObservation(env)
    if stack_size > 1:
        env = StackObservation(env, stack_size=stack_size)
    env = RecordEpisodeStats(env)
    return env


# ---------------------------------------------------------------------------
# SB3-compatible wrappers  (Week 2)
# ---------------------------------------------------------------------------

class SB3CompatWrapper(gym.Wrapper):
    """Thin wrapper ensuring Gymnasium/SB3 API compatibility.

    - Clips actions to action-space bounds (replaces ClipActions).
    - Records episode statistics under the ``episode`` info key,
      which SB3 monitors require (replaces RecordEpisodeStats).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_length += 1
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
            }
        return obs, reward, terminated, truncated, info


def make_sb3_env(
    data,
    n_envs: int = 1,
    use_vec_normalize: bool = True,
    vec_normalize_kwargs: Optional[dict] = None,
    **env_kwargs,
):
    """Create a vectorized, SB3-ready environment.

    Args:
        data:                  DataFrame with OHLCV data.
        n_envs:                Number of parallel environments.
        use_vec_normalize:     Whether to wrap with VecNormalize.
        vec_normalize_kwargs:  Override VecNormalize defaults.
        **env_kwargs:          Passed to SingleAssetRLTradingEnv.

    Returns:
        DummyVecEnv or VecNormalize-wrapped DummyVecEnv.

    Example::

        from envs.wrap_env import make_sb3_env
        vec_env = make_sb3_env(df, n_envs=4, use_vec_normalize=True)
        model = PPO("MlpPolicy", vec_env, verbose=1)
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.single_asset_rl_env import SingleAssetRLTradingEnv

    def _make() -> gym.Env:
        env = SingleAssetRLTradingEnv(data=data, **env_kwargs)
        return SB3CompatWrapper(env)

    vec_env = DummyVecEnv([_make] * n_envs)

    if use_vec_normalize:
        _vn_defaults = {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": env_kwargs.get("gamma", 0.99),
        }
        if vec_normalize_kwargs:
            _vn_defaults.update(vec_normalize_kwargs)
        vec_env = VecNormalize(vec_env, **_vn_defaults)

    return vec_env
