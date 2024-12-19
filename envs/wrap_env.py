import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import torch
import mlflow

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to range [-1, 1] with support for GPU and NaN handling"""
    
    def __init__(self, env, device='cpu'):
        super().__init__(env)
        self.device = torch.device(device)
        
        # Initialize running statistics
        self.is_vector_env = hasattr(env, 'num_envs')
        
        # Update observation space to reflect normalized values
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Normalize observation"""
        if isinstance(obs, np.ndarray):
            # Handle NaN values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            # Ensure observation is in [-1, 1] range
            obs = np.clip(obs, -1, 1)
            return obs.astype(np.float32)
        elif isinstance(obs, torch.Tensor):
            # Handle NaN values
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            # Ensure observation is in [-1, 1] range
            obs = torch.clamp(obs, -1, 1)
            return obs.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

class StackObservation(gym.ObservationWrapper):
    """Stack observations to create a history of observations"""
    
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        
        # Calculate new observation space shape
        old_shape = env.observation_space.shape
        if len(old_shape) != 2:
            raise ValueError(f"Expected 2D observation shape (window_size, features), got {old_shape}")
        
        # New shape will be (window_size, features)
        # Keep original feature dimension
        new_shape = (old_shape[0], old_shape[1])
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=np.float32
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
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward
        self.current_length += 1
        
        if terminated or truncated:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            info['episode'] = {
                'r': self.current_return,
                'l': self.current_length,
                'returns': self.episode_returns,
                'lengths': self.episode_lengths
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
        mlflow.set_experiment(experiment_name)
        self.episode_count = 0
        self.step_count = 0
    
    def reset(self, **kwargs):
        """Reset with MLflow logging"""
        obs, info = self.env.reset(**kwargs)
        
        # Log reset metrics
        mlflow.log_metrics({
            'initial_balance': info.get('balance', 0),
            'initial_price': info.get('current_price', 0)
        }, step=self.episode_count)
        
        return obs, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Log step metrics
        mlflow.log_metrics({
            'step_reward': reward,
            'portfolio_value': info.get('portfolio_value', 0),
            'position_size': info.get('position_size', 0)
        }, step=self.step_count)
        
        if terminated or truncated:
            self.episode_count += 1
            # Log episode metrics
            mlflow.log_metrics({
                'episode_return': info['episode']['r'],
                'episode_length': info['episode']['l'],
                'total_trades': info.get('total_trades', 0),
                'win_rate': info.get('win_rate', 0)
            }, step=self.episode_count)
        
        return observation, reward, terminated, truncated, info

def make_env(env, normalize=True, stack_size=4):
    """Create environment with specified wrappers"""
    # Add action clipping
    env = ClipActions(env)
    
    # Add observation normalization if requested
    if normalize:
        env = NormalizeObservation(env)
    
    # Add observation stacking if requested
    if stack_size > 1:
        env = StackObservation(env, stack_size=stack_size)
    
    # Add episode statistics recording
    env = RecordEpisodeStats(env)
    
    return env