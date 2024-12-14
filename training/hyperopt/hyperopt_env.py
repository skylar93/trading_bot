"""
Simplified environment for hyperparameter optimization.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class HyperoptEnv(gym.Env):
    """
    Simplified trading environment for hyperparameter optimization.
    Stripped down version of TradingEnvironment for faster iteration during tuning.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001,
                 window_size: int = 20):
        
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # Action space: continuous value between -1 (full sell) and 1 (full buy)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: price data + position info
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 5),  # OHLCV
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.portfolio_values = [self.initial_balance]
        
        return self._get_observation(), {}
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute trade
        if abs(action) > 0.05:  # Small threshold to prevent tiny trades
            if action > 0:  # Buy
                shares_to_buy = (self.balance * abs(action)) / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.position += shares_to_buy
                    self.balance -= cost
                    
            else:  # Sell
                shares_to_sell = self.position * abs(action)
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.position -= shares_to_sell
                self.balance += revenue
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (portfolio value change)
        portfolio_value = self.balance + (self.position * current_price)
        self.portfolio_values.append(portfolio_value)
        reward = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation of the environment"""
        # Get price data window
        df_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Normalize features
        price_mean = df_window['close'].mean()
        price_std = df_window['close'].std()
        volume_mean = df_window['volume'].mean()
        volume_std = df_window['volume'].std()
        
        # Construct features
        obs = np.zeros((self.window_size, 5))
        for i, col in enumerate(['open', 'high', 'low', 'close']):
            obs[:, i] = (df_window[col] - price_mean) / price_std
        obs[:, 4] = (df_window['volume'] - volume_mean) / volume_std
        
        return obs.astype(np.float32)
        
    def get_portfolio_stats(self) -> Dict:
        """Calculate key portfolio statistics"""
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # Calculate Sortino ratio (using only negative returns)
        negative_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'total_return': float(total_return),
            'final_value': float(portfolio_values[-1])
        }