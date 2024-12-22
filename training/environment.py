"""Trading Environment for Reinforcement Learning"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        trading_fee: float = 0.001,
    ):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee

        # Define action space (0: sell, 1: hold, 2: buy)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                len(data.columns) + 4,
            ),  # features + [balance, position, unrealized_pnl, current_price]
            dtype=np.float32,
        )

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # Current position in the asset
        self.position_value = 0
        self.trades = []

        return self.get_state(), {}

    def get_state(self) -> np.ndarray:
        """Get current state observation"""
        features = self.data.iloc[self.current_step].values
        current_price = self.data.iloc[self.current_step]["close"]
        unrealized_pnl = (
            (current_price - self.position_value) * self.position
            if self.position != 0
            else 0
        )

        # Combine features with trading information
        state = np.concatenate(
            [
                features,
                [self.balance, self.position, unrealized_pnl, current_price],
            ]
        )

        return state.astype(np.float32)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self.get_state(), 0, True, False, self._get_info()

        # Get current price
        current_price = self.data.iloc[self.current_step]["close"]
        next_price = self.data.iloc[self.current_step + 1]["close"]

        # Execute action
        reward = 0

        if action == 0:  # Sell
            if self.position > 0:
                sell_value = current_price * self.position
                fee = sell_value * self.trading_fee
                self.balance += sell_value - fee
                self.trades.append(
                    {
                        "type": "sell",
                        "price": current_price,
                        "quantity": self.position,
                        "fee": fee,
                    }
                )
                self.position = 0
                self.position_value = 0

        elif action == 2:  # Buy
            if self.position == 0:
                max_quantity = self.balance / (
                    current_price * (1 + self.trading_fee)
                )
                self.position = max_quantity
                self.position_value = current_price
                buy_value = current_price * self.position
                fee = buy_value * self.trading_fee
                self.balance -= buy_value + fee
                self.trades.append(
                    {
                        "type": "buy",
                        "price": current_price,
                        "quantity": self.position,
                        "fee": fee,
                    }
                )

        # Calculate reward
        if self.position > 0:
            price_change = (next_price - current_price) / current_price
            reward = price_change * self.position * current_price

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self.get_state(), reward, done, False, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment"""
        current_price = self.data.iloc[self.current_step]["close"]
        portfolio_value = self.balance
        if self.position > 0:
            portfolio_value += current_price * self.position

        return {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "current_price": current_price,
            "trades": self.trades,
        }

    def render(self):
        """Render the environment"""
        pass  # We'll use Streamlit for visualization instead
