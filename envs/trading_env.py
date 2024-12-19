import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning"""
    
    def __init__(self,
                 df: Optional[pd.DataFrame] = None,
                 initial_balance: float = 10000.0,
                 trading_fee: float = 0.001,
                 window_size: int = 20,
                 max_position_size: float = 1.0):
        """Initialize environment
        
        Args:
            df: DataFrame with OHLCV data (optional)
            initial_balance: Initial account balance
            trading_fee: Trading fee as fraction of trade value
            window_size: Number of time steps to include in state
            max_position_size: Maximum position size as fraction of balance
        """
        super().__init__()
        
        # Initialize data
        if df is not None:
            # Convert column names if needed
            rename_map = {
                'open': '$open',
                'high': '$high',
                'low': '$low',
                'close': '$close',
                'volume': '$volume'
            }
            self.df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            # Verify required columns
            required_columns = ['$open', '$high', '$low', '$close', '$volume']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        else:
            # Create empty DataFrame with required columns
            self.df = pd.DataFrame(columns=['$open', '$high', '$low', '$close', '$volume'])
        
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.max_position_size = max_position_size
        
        # Calculate number of features
        self.n_features = 5  # OHLCV features
        
        # Set up spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),  # 2D observation space
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info(
            f"Initialized TradingEnvironment with window_size={window_size}, "
            f"initial_balance={initial_balance}, trading_fee={trading_fee}"
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = self.window_size if len(self.df) > self.window_size else 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take step in environment
        
        Args:
            action: Action to take (-1 to 1, scaled to position size)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if len(self.df) <= self.current_step:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Get current price and calculate target position
        current_price = self.df.iloc[self.current_step]['$close']
        target_position = float(action[0]) * self.max_position_size
        
        # Calculate trade size
        trade_size = target_position - self.position
        
        # Execute trade if non-zero
        if abs(trade_size) > 0:
            # Calculate trade cost
            trade_value = abs(trade_size * current_price)
            fee = trade_value * self.trading_fee
            
            # Update balance and position
            self.balance -= fee
            if trade_size > 0:  # Buy
                self.balance -= trade_value
            else:  # Sell
                self.balance += trade_value
            
            self.position = target_position
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'type': 'buy' if trade_size > 0 else 'sell',
                'size': abs(trade_size),
                'price': current_price,
                'fee': fee
            })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        old_portfolio_value = self._calculate_portfolio_value(self.current_step - 1)
        new_portfolio_value = self._calculate_portfolio_value(self.current_step)
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation
        
        Returns:
            2D array of shape (window_size, n_features) with OHLCV data
        """
        if len(self.df) <= self.window_size:
            # Return zero array if not enough data
            return np.zeros((self.window_size, self.n_features), dtype=np.float32)
        
        # Get window of data
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # Extract OHLCV values
        features = []
        for col in ['$open', '$high', '$low', '$close', '$volume']:
            values = window_data[col].values
            # Normalize volume separately
            if col == '$volume':
                values = values / values.mean() if values.mean() != 0 else values
            else:
                # Normalize prices relative to first value
                values = values / values[0] - 1 if values[0] != 0 else values
            features.append(values)
        
        # Stack features into 2D array
        obs = np.stack(features, axis=1)
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info
        
        Returns:
            Dictionary with current state information
        """
        portfolio_value = self._calculate_portfolio_value(self.current_step)
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_trades': len(self.trades)
        }
    
    def _calculate_portfolio_value(self, step: int) -> float:
        """Calculate total portfolio value at given step
        
        Args:
            step: Time step to calculate value for
            
        Returns:
            Total portfolio value (balance + position value)
        """
        if len(self.df) <= step:
            return self.balance
        
        price = self.df.iloc[step]['$close']
        position_value = self.position * price
        return self.balance + position_value
    
    def render(self, mode: str = 'human'):
        """Render environment"""
        pass  # Not implemented
    
    def close(self):
        """Clean up environment"""
        pass