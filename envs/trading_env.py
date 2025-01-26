import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning"""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 20,
        max_position: float = 1.0,
    ):
        """Initialize environment

        Args:
            data: DataFrame with OHLCV data (optional)
            initial_capital: Initial account capital
            trading_fee: Trading fee as fraction of trade value
            window_size: Number of time steps to include in state
            max_position: Maximum position size as fraction of capital
        """
        super().__init__()

        # Initialize data
        if data is not None:
            # Convert column names if needed
            rename_map = {
                "open": "$open",
                "high": "$high",
                "low": "$low",
                "close": "$close",
                "volume": "$volume",
            }
            self.data = data.rename(
                columns={
                    k: v for k, v in rename_map.items() if k in data.columns
                }
            )

            # Verify required columns
            required_columns = ["$open", "$high", "$low", "$close", "$volume"]
            missing_cols = [
                col for col in required_columns if col not in self.data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}"
                )
        else:
            self.data = None

        # Store parameters
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.max_position = max_position

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: OHLCV data for window_size steps
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, 5),  # OHLCV
            dtype=np.float32
        )

        # Initialize state variables
        self.current_step = None
        self.current_position = None
        self.current_capital = None
        self.portfolio_value = None
        self.done = None
        self.trades = []

        logger.info(
            f"Initialized TradingEnvironment with window_size={window_size}, "
            f"initial_capital={initial_capital}, trading_fee={trading_fee}"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if self.data is None:
            raise ValueError("No data provided to environment")

        # Reset state
        self.current_step = self.window_size
        self.current_position = 0.0
        self.current_capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.done = False
        self.trades = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.done:
            raise RuntimeError("Environment is done, call reset() first")

        # Get current price data
        current_price = self.data.iloc[self.current_step]["$close"]
        
        # Calculate target position change
        position_change = float(action[0]) * self.max_position
        target_position = self.current_position + position_change
        
        # Apply position limits
        target_position = np.clip(
            target_position, 
            -self.max_position,
            self.max_position
        )
        
        # Calculate actual position change
        actual_change = target_position - self.current_position
        
        # Execute trade if there is a position change
        if abs(actual_change) > 0:
            # Calculate trade cost
            trade_value = abs(actual_change * current_price)
            trade_cost = trade_value * self.trading_fee
            
            # Update capital and position
            self.current_capital -= trade_cost
            if actual_change > 0:  # Buy
                self.current_capital -= trade_value
            else:  # Sell
                self.current_capital += trade_value
            
            self.current_position = target_position
            
            # Record trade
            self.trades.append({
                "step": self.current_step,
                "price": current_price,
                "size": actual_change,
                "cost": trade_cost,
                "type": "buy" if actual_change > 0 else "sell"
            })

        # Update portfolio value
        self.portfolio_value = self._calculate_portfolio_value(self.current_step)
        
        # Calculate reward (change in portfolio value)
        reward = (self.portfolio_value / self.initial_capital) - 1.0
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.done, False, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation (window of OHLCV data)"""
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract OHLCV data
        observation = np.column_stack([
            window_data["$open"].values,
            window_data["$high"].values,
            window_data["$low"].values,
            window_data["$close"].values,
            window_data["$volume"].values,
        ])
        
        return observation.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "step": self.current_step,
            "position": self.current_position,
            "capital": self.current_capital,
            "portfolio_value": self.portfolio_value,
            "total_trades": len(self.trades)
        }

    def _calculate_portfolio_value(self, step: int) -> float:
        """Calculate total portfolio value at current step"""
        if step >= len(self.data):
            return self.portfolio_value
            
        current_price = self.data.iloc[step]["$close"]
        position_value = self.current_position * current_price
        return self.current_capital + position_value

    def render(self, mode: str = "human"):
        """Render the environment"""
        pass  # Not implemented

    def close(self):
        """Clean up environment"""
        pass  # Not implemented
