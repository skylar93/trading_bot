import gymnasium as gym
import logging

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001, window_size: int = 60):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # Action space: continuous action between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: OHLCV data + technical indicators
        n_features = 10  # price, volume, position, balance, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation(), {}
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action (float): Value between -1 and 1 indicating the trading action
                          -1: full sell, 0: hold, 1: full buy
        """
        # Get current price and next price
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute trade
        if action > 0:  # Buy
            shares_to_buy = (self.balance * abs(action)) / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_fee)
            if cost <= self.balance:
                self.position += shares_to_buy
                self.balance -= cost
                self.trades.append(('buy', shares_to_buy, current_price))
        
        elif action < 0:  # Sell
            shares_to_sell = self.position * abs(action)
            revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
            self.position -= shares_to_sell
            self.balance += revenue
            self.trades.append(('sell', shares_to_sell, current_price))
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        portfolio_value = self.balance + (self.position * current_price)
        prev_portfolio_value = self.balance + (self.position * self.df.iloc[self.current_step-1]['close'])
        reward = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance
        }
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation"""
        # Get the price data for the current window
        logger.info(f"Current step: {self.current_step}, Window size: {self.window_size}")
        logger.info(f"DataFrame length: {len(self.df)}")
        df_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        logger.info(f"Window data shape: {df_window.shape}")
        
        # Normalize the data
        price_mean = df_window['close'].mean()
        price_std = df_window['close'].std()
        volume_mean = df_window['volume'].mean()
        volume_std = df_window['volume'].std()
        
        # Construct features
        features = []
        # Price features
        for col in ['open', 'high', 'low', 'close']:
            features.append((df_window[col] - price_mean) / price_std)
        # Volume
        features.append((df_window['volume'] - volume_mean) / volume_std)
        # Returns and changes
        features.append(df_window['close'].pct_change().fillna(0))
        features.append(df_window['volume'].pct_change().fillna(0))
        # Portfolio info
        features.append(pd.Series([self.position] * len(df_window)))
        features.append(pd.Series([self.balance / self.initial_balance] * len(df_window)))
        features.append(pd.Series([(self.balance + self.position * df_window['close'].iloc[-1]) / self.initial_balance] * len(df_window)))
        
        obs = np.array(features).T
        logger.info(f"Observation shape: {obs.shape}")
        
        return obs.astype(np.float32)