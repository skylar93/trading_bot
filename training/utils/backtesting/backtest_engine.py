"""
Multi-Asset Backtesting Engine
=============================

This module implements a generalized backtesting engine supporting multiple assets
and portfolio management. It provides a foundation for building complex trading
systems with sophisticated position management and risk controls.

File Structure:
--------------
- Class: BacktestEngine
  - Core engine for multi-asset backtesting
  - Portfolio management and trade execution
  - Performance tracking and analysis

Key Components:
--------------
1. Portfolio Management
   - Multi-asset position tracking
   - Position size limits per asset
   - Portfolio value calculation
   - Cash balance management

2. Trade Execution
   - Action processing (-1 to 1 per asset)
   - Transaction cost handling
   - Position updates and tracking
   - Trade history logging

3. Performance Analysis
   - Portfolio value history
   - Return calculation
   - Position history tracking
   - Trade history management

Dependencies:
------------
- numpy: Numerical operations
- pandas: Data handling and analysis
- typing: Type hints and annotations
- datetime: Timestamp processing

Implementation Notes:
-------------------
1. Position Management
   - Uses dictionary for multi-asset positions
   - Implements position limits per asset
   - Handles cash balance separately
   - Tracks entry prices for PnL

2. Trade Processing
   - Supports simultaneous trades across assets
   - Validates position limits before execution
   - Updates portfolio state after each trade
   - Maintains detailed trade history

3. Data Requirements
   - Timestamp-indexed price data
   - Asset identifiers as dictionary keys
   - Price data for all tracked assets

Example Usage:
-------------
```python
# Initialize engine
engine = BacktestEngine(
    initial_capital=100000.0,
    transaction_cost=0.001,  # 0.1% fee
    max_position=0.2        # 20% max per asset
)

# Prepare multi-asset data
prices = {
    'BTC': 50000.0,
    'ETH': 3000.0
}

# Define actions (-1 to 1 for each asset)
actions = {
    'BTC': 0.5,   # Buy 50% of allowed size
    'ETH': -0.3   # Sell 30% of allowed size
}

# Update portfolio
engine.update(
    timestamp=pd.Timestamp.now(),
    prices=prices,
    actions=actions
)

# Get results
print(f"Portfolio Value: {engine.get_portfolio_value(prices)}")
print(f"Returns: {engine.get_returns()}")
print(f"Positions:\\n{engine.get_position_history()}")
```

Portfolio Tracking:
-----------------
1. Value Components
   - Cash balance
   - Asset positions
   - Total portfolio value

2. History Tracking
   - Portfolio value over time
   - Position sizes per asset
   - Trade execution details
   - Cash balance changes

3. Performance Metrics
   - Returns calculation
   - Position exposure
   - Trading activity

Recent Changes:
--------------
- Added position history tracking
- Improved transaction cost handling
- Enhanced portfolio value calculation
- Added detailed trade logging

See Also:
---------
- Backtester: Single-asset implementation
- RiskAwareBacktester: Risk-managed version
- ExperimentalBacktester: Advanced features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
    ):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as percentage
            max_position: Maximum allowed position size as percentage of portfolio
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reset()

    def reset(self):
        """Reset backtesting state"""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions: Dict[str, float] = {}  # symbol: amount
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]
        self.current_timestamp = None

    def update(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        actions: Dict[str, float],
    ) -> Dict:
        """
        Update portfolio state based on actions

        Args:
            timestamp: Current timestamp
            prices: Dictionary of asset prices
            actions: Dictionary of trading actions (-1 to 1)

        Returns:
            Dict of current state metrics
        """
        self.current_timestamp = timestamp

        # Execute trades
        for symbol, action in actions.items():
            if symbol not in prices:
                continue

            # Calculate target position
            current_price = prices[symbol]
            portfolio_value = self.get_portfolio_value(prices)

            # For action == 0, fully close the position
            if abs(action) < 1e-6:
                target_position = 0
            else:
                max_position_value = portfolio_value * self.max_position
                target_position_value = action * max_position_value
                target_position = target_position_value / current_price

            # Current position
            current_position = self.positions.get(symbol, 0.0)

            # Calculate trade amount
            trade_amount = target_position - current_position
            trade_value = abs(trade_amount * current_price)
            trade_cost = trade_value * self.transaction_cost

            if abs(trade_amount) > 1e-6:  # Minimum trade threshold
                # Check if we have enough cash for buying
                if trade_amount > 0 and trade_value + trade_cost > self.cash:
                    trade_amount = self.cash / (
                        current_price * (1 + self.transaction_cost)
                    )
                    trade_value = abs(trade_amount * current_price)
                    trade_cost = trade_value * self.transaction_cost

                if abs(trade_amount) > 1e-6:
                    # Record trade
                    trade = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "price": current_price,
                        "amount": trade_amount,
                        "value": trade_value,
                        # Check if we're increasing or decreasing position
                        "type": (
                            "buy"
                            if (
                                (trade_amount > 0 and current_position >= 0)
                                or (trade_amount < 0 and current_position < 0)
                            )
                            else "sell"
                        ),
                        "cost": trade_cost,
                    }
                    self.trades.append(trade)

                    # Update position and cash
                    self.positions[symbol] = current_position + trade_amount
                    self.cash -= (
                        (trade_value + trade_cost)
                        if trade_amount > 0
                        else -(trade_value - trade_cost)
                    )

                    # Remove position if close to zero
                    if abs(self.positions[symbol]) < 1e-6:
                        del self.positions[symbol]

        # Update history
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_history.append(portfolio_value)
        self.cash_history.append(self.cash)

        return {
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "trades": self.trades[-1] if self.trades else None,
        }

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash"""
        value = self.cash

        for symbol, position in self.positions.items():
            if symbol in prices:
                value += position * prices[symbol]

        return value

    def get_returns(self) -> pd.Series:
        """Calculate returns series"""
        returns = pd.Series(self.portfolio_history)
        returns = returns.pct_change()
        returns.index = pd.date_range(
            start=self.current_timestamp - pd.Timedelta(days=len(returns) - 1),
            end=self.current_timestamp,
            periods=len(returns),
        )
        return returns

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_position_history(self) -> pd.DataFrame:
        """Get position value history for each asset"""
        position_values = []
        timestamps = pd.date_range(
            start=self.current_timestamp
            - pd.Timedelta(days=len(self.portfolio_history) - 1),
            end=self.current_timestamp,
            periods=len(self.portfolio_history),
        )

        for timestamp, portfolio_value in zip(
            timestamps, self.portfolio_history
        ):
            values = {"timestamp": timestamp, "total": portfolio_value}
            values.update(self.positions)
            position_values.append(values)

        return pd.DataFrame(position_values).set_index("timestamp")
