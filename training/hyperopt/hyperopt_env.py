"""
Simplified environment for hyperparameter optimization.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class SimplifiedTradingEnv(gym.Env):
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
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 5),  # OHLCV
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.returns = []
        self.portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_price = self.df.iloc[self.current_step]['close']
        prev_value = self.portfolio_value
        
        # Execute trade
        if abs(action) > 0.05:  # Threshold to prevent tiny trades
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
        
        # Update step and values
        self.current_step += 1
        self.portfolio_value = self.balance + (self.position * current_price)
        
        # Calculate return and add to history
        returns = (self.portfolio_value - prev_value) / prev_value
        self.returns.append(returns)
        
        # Calculate reward as Sharpe-like ratio for recent returns
        recent_returns = self.returns[-self.window_size:] if len(self.returns) > self.window_size else self.returns
        reward = self._calculate_sharpe_ratio(recent_returns) if recent_returns else 0
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'return': returns
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get normalized OHLCV observation"""
        df_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Normalize prices together
        price_mean = df_window[['open', 'high', 'low', 'close']].mean().mean()
        price_std = df_window[['open', 'high', 'low', 'close']].std().mean()
        
        # Normalize volume separately
        volume_mean = df_window['volume'].mean()
        volume_std = df_window['volume'].std()
        
        obs = np.zeros((self.window_size, 5))
        
        # Normalize OHLC using same stats
        for i, col in enumerate(['open', 'high', 'low', 'close']):
            obs[:, i] = (df_window[col] - price_mean) / (price_std + 1e-8)
            
        # Normalize volume
        obs[:, 4] = (df_window['volume'] - volume_mean) / (volume_std + 1e-8)
        
        return obs.astype(np.float32)

    @staticmethod
    def _calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns:
            return 0.0
            
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        if len(excess_returns) < 2:
            return 0.0
            
        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / std
        return float(np.sqrt(252) * sharpe)  # Annualized