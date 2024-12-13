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
        
        # Observation space: price data + features + position info
        num_features = len(self.feature_columns) * window_size + 4  # +4 for position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed: Optional[int] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = None
        self.done = False
        self.position_history = []
        
        return self._get_observation(), {}  # Return observation and info dict
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        # Get window of feature data
        features = self.df[self.feature_columns].iloc[
            self.current_step - self.window_size:self.current_step].values.flatten()
        
        # Add position information
        position_size = 0.0
        position_type = 0.0  # -1 for short, 1 for long
        position_profit = 0.0
        position_hold_time = 0.0
        
        if self.position is not None:
            position_size = self.position.size / self.max_position_size
            position_type = 1.0 if self.position.type == 'long' else -1.0
            current_price = self.df['$close'].iloc[self.current_step]
            position_profit = self.position.calculate_pnl(current_price) / self.initial_balance
            position_hold_time = (self.df['datetime'].iloc[self.current_step] - 
                                self.position.entry_time).total_seconds() / (24 * 3600)  # in days
        
        position_info = np.array([
            position_size,
            position_type,
            position_profit,
            position_hold_time
        ])
        
        return np.concatenate([features, position_info])
    
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
                entry_time=self.df['datetime'].iloc[self.current_step]
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
            'timestamp': self.df['datetime'].iloc[self.current_step]
        }
        
        return obs, reward, done, truncated, info