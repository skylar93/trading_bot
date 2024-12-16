import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
from dataclasses import dataclass

@dataclass
class Position:
    """Position information"""
    type: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        if self.type == 'long':
            return (current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - current_price) * self.size

class TradingEnvironment(gym.Env):
    """Basic trading environment"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 trading_fee: float = 0.001,
                 window_size: int = 30,
                 max_position_size: float = 1.0):
        """
        Args:
            df: DataFrame with OHLCV and feature data
            initial_balance: Initial account balance
            trading_fee: Trading fee as a fraction
            window_size: Number of time steps to include in state
            max_position_size: Maximum position size as a fraction of balance
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.max_position_size = max_position_size
        
        # Get feature columns (excluding datetime and instrument)
        self.feature_columns = [col for col in df.columns 
                              if col not in ['datetime', 'instrument']]
        
        # Action space: -1 (sell/short) to 1 (buy/long)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: (window_size, features)
        # Features include price data, technical indicators, and position info
        self.n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.n_features), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment
        
        Args:
            seed: Random seed
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset internal state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = None
        self.position_history = []
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': self.df['$close'].iloc[self.current_step]
        }
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        # Get window of feature data
        obs = self.df[self.feature_columns].iloc[
            self.current_step - self.window_size:self.current_step].values
        
        # Normalize each feature independently
        for i in range(obs.shape[1]):
            col = obs[:, i]
            min_val = np.min(col)
            max_val = np.max(col)
            if min_val != max_val:
                obs[:, i] = 2 * (col - min_val) / (max_val - min_val) - 1
            else:
                obs[:, i] = 0
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: float) -> float:
        """Calculate reward for the current step"""
        # Get current price
        current_price = self.df['$close'].iloc[self.current_step]
        
        # Calculate profit/loss if we have a position
        reward = 0.0
        if self.position is not None:
            reward = self.position.calculate_pnl(current_price) / self.initial_balance
        
        # Penalize for trading fees
        if abs(action) > 0:
            reward -= self.trading_fee
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        action_value = action[0]  # Extract scalar value from action array
        
        # Calculate reward before taking action
        reward = self._calculate_reward(action_value)
        
        # Process the action
        current_price = self.df['$close'].iloc[self.current_step]
        current_time = self.df.index[self.current_step]
        
        # Close existing position if action is in opposite direction
        if self.position is not None:
            if (self.position.type == 'long' and action_value < 0) or \
               (self.position.type == 'short' and action_value > 0):
                # Record position
                self.position_history.append(self.position)
                self.position = None
        
        # Open new position
        if self.position is None and abs(action_value) > 0.1:  # Threshold to prevent tiny positions
            position_type = 'long' if action_value > 0 else 'short'
            position_size = abs(action_value) * self.max_position_size * self.balance / current_price
            self.position = Position(
                type=position_type,
                size=position_size,
                entry_price=current_price,
                entry_time=current_time
            )
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        truncated = False  # For gymnasium compatibility
        
        # Get new observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'timestamp': current_time
        }
        
        return obs, reward, done, truncated, info