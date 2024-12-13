import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to range [-1, 1]"""
    
    def __init__(self, env):
        super().__init__(env)
        self.running_mean = None
        self.running_std = None
        
    def observation(self, obs):
        # Initialize running statistics if not already done
        if self.running_mean is None:
            self.running_mean = np.zeros_like(obs)
            self.running_std = np.ones_like(obs)
        
        # Update running statistics
        self.running_mean = 0.99 * self.running_mean + 0.01 * obs
        self.running_std = 0.99 * self.running_std + 0.01 * np.abs(obs - self.running_mean)
        
        # Normalize
        normalized_obs = (obs - self.running_mean) / (self.running_std + 1e-8)
        return np.clip(normalized_obs, -1, 1)

class StackObservation(gym.ObservationWrapper):
    """Stack multiple observations"""
    
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.stacked_obs = None
        
        # Update observation space
        low = np.repeat(self.observation_space.low, stack_size)
        high = np.repeat(self.observation_space.high, stack_size)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stacked_obs = np.tile(obs, (self.stack_size, 1))
        return self.stacked_obs.flatten(), info
    
    def observation(self, obs):
        if self.stacked_obs is None:
            self.stacked_obs = np.tile(obs, (self.stack_size, 1))
        else:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-1, axis=0)
            self.stacked_obs[-1] = obs
        return self.stacked_obs.flatten()

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

def make_env(env, normalize=True, stack_size=4):
    """Create wrapped environment"""
    env = ClipActions(env)
    if normalize:
        env = NormalizeObservation(env)
    if stack_size > 1:
        env = StackObservation(env, stack_size)
    env = RecordEpisodeStats(env)
    return env