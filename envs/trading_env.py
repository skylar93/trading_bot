import gymnasium as gym
import logging
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple

logger = logging.getLogger('trading_bot.env')

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001, window_size: int = 60,
                 max_position_size: float = 1.0):
        super(TradingEnvironment, self).__init__()
        
        logger.info(f"Initializing TradingEnvironment with window_size={window_size}, "
                   f"initial_balance={initial_balance}, transaction_fee={transaction_fee}")

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.max_position_size = max_position_size
        
        # Action space: continuous action between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Calculate number of features
        self.feature_columns = [col for col in df.columns if col not in ['datetime', 'instrument']]
        self.n_features = len(self.feature_columns)
        
        # Observation space: price data + technical indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )
        
        logger.debug(f"Environment initialized with observation space shape: {self.observation_space.shape}")
        logger.debug(f"Available features: {self.feature_columns}")
        self.reset()
    
    @property
    def portfolio_value(self):
        """Calculate current portfolio value"""
        current_price = self.df.iloc[self.current_step]['$close']
        value = self.balance + (self.position * current_price)
        logger.debug(f"Portfolio value: {value:.2f} (balance: {self.balance:.2f}, "
                    f"position: {self.position:.4f} @ {current_price:.2f})")
        return value
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        logger.info("Resetting environment")
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.trades = []
        
        logger.debug(f"Reset state - step: {self.current_step}, balance: {self.balance}, "
                    f"position: {self.position}")
        
        # Get observation
        obs = self._get_observation()
        
        # Update info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': self.df.iloc[self.current_step]['$close']
        }
        
        logger.debug(f"Reset complete - observation shape: {obs.shape}, info: {info}")
        return obs, info
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        logger.debug(f"Step {self.current_step} - Executing action: {action[0]:.4f}")
        
        # Get current price
        current_price = float(self.df.iloc[self.current_step]['$close'])
        prev_portfolio_value = self.portfolio_value
        
        # Calculate maximum position value allowed
        max_position_value = self.initial_balance * self.max_position_size
        
        # Execute trade
        if action[0] > 0:  # Buy
            # Calculate maximum shares that can be bought considering position limit
            current_position_value = self.position * current_price
            remaining_position_value = max_position_value - current_position_value
            max_cost = min(self.balance, remaining_position_value) / (1 + self.transaction_fee)
            max_shares = max_cost / current_price
            shares_to_buy = max_shares * abs(float(action[0]))
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.position += shares_to_buy
                    self.balance -= cost
                    self.trades.append(('buy', shares_to_buy, current_price))
                    logger.info(f"Bought {shares_to_buy:.4f} shares at {current_price:.2f}")
                else:
                    logger.warning("Insufficient balance for buy order")
            
        elif action[0] < 0:  # Sell
            shares_to_sell = self.position * abs(float(action[0]))
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.position -= shares_to_sell
                self.balance += revenue
                self.trades.append(('sell', shares_to_sell, current_price))
                logger.info(f"Sold {shares_to_sell:.4f} shares at {current_price:.2f}")
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (relative change in portfolio value)
        current_portfolio_value = self.portfolio_value
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        obs = self._get_observation()
        info = {
            'portfolio_value': float(current_portfolio_value),
            'position': float(self.position),
            'balance': float(self.balance),
            'current_price': float(current_price),
            'action': float(action[0])
        }
        
        if done:
            logger.info(f"Episode complete - Final portfolio value: {current_portfolio_value:.2f}")
        
        return obs, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation"""
        try:
            # Get the price data for the current window
            df_window = self.df[self.feature_columns].iloc[self.current_step-self.window_size:self.current_step]
            
            if len(df_window) < self.window_size:
                logger.warning(f"Insufficient data for window size {self.window_size}, padding with zeros")
                # Pad with zeros if we don't have enough data
                pad_size = self.window_size - len(df_window)
                pad_shape = (pad_size, self.n_features)
                padding = np.zeros(pad_shape, dtype=np.float32)
                obs = np.vstack([padding, df_window.values])
            else:
                obs = df_window.values
            
            # Normalize each feature independently
            for i in range(obs.shape[1]):
                col = obs[:, i]
                min_val = np.min(col)
                max_val = np.max(col)
                if min_val != max_val:
                    obs[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    obs[:, i] = 0
            
            logger.debug(f"Generated observation with shape {obs.shape}")
            return obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating observation: {str(e)}", exc_info=True)
            raise