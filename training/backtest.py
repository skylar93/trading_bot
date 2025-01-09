"""
Single-Asset Backtesting System
==============================

This module implements a basic backtesting system for single-asset trading strategies.
Focused on simplicity and clarity, it serves as a foundation for more complex backtesting systems.

File Structure:
--------------
- Class: Backtester
  - Core backtesting engine for single-asset trading
  - Handles trade execution and performance tracking
  - Calculates key trading metrics

Key Components:
--------------
1. Trade Execution
   - Position sizing based on action (-1 to 1)
   - Transaction cost consideration
   - Balance and position tracking

2. Performance Tracking
   - Portfolio value history
   - Trade history logging
   - Performance metrics calculation
   - Peak value tracking for drawdown

3. Strategy Integration
   - Window-based strategy execution
   - Action validation and processing
   - Trade result accumulation

Dependencies:
------------
- numpy: Numerical computations and metrics
- pandas: Data handling and time series operations
- logging: Debug and transaction logging
- datetime: Timestamp handling
- typing: Type hints for better code clarity

Implementation Notes:
-------------------
1. Data Requirements
   - OHLCV columns must be prefixed with '$'
   - Required columns: $open, $high, $low, $close, $volume
   - DataFrame index serves as timestamp

2. Position Management
   - Single position tracking (no multi-asset support)
   - Dust position cleanup (< 1e-4)
   - Balance updates include transaction fees

3. Performance Metrics
   - Sharpe Ratio (annualized, risk-free rate = 0)
   - Sortino Ratio (downside deviation)
   - Maximum Drawdown
   - Win Rate calculation

Example Usage:
-------------
```python
# Prepare OHLCV data
data = pd.DataFrame({
    '$open': [...],
    '$high': [...],
    '$low': [...],
    '$close': [...],
    '$volume': [...]
})

# Initialize backtester
backtester = Backtester(
    data=data,
    initial_balance=10000.0,
    trading_fee=0.001  # 0.1% fee
)

# Create strategy (must implement get_action method)
class SimpleStrategy:
    def get_action(self, window_data):
        # Return value between -1 and 1
        return 0.5  # Example: always buy with 50% size

# Run backtest
results = backtester.run(
    strategy=SimpleStrategy(),
    window_size=20,
    verbose=True
)

# Access results
print(f"Final Value: {results['portfolio_values'][-1]}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']}")
```

Logging Structure:
----------------
- ERROR: Critical failures (e.g., strategy errors)
- WARNING: Potential issues (e.g., invalid actions)
- INFO: Trade execution, progress updates
- DEBUG: Detailed calculations, state changes

Recent Changes:
--------------
- Enhanced logging for better debugging
- Improved PnL calculation accuracy
- Added detailed trade history
- Fixed position size validation

See Also:
---------
- BacktestEngine: Multi-asset backtesting system
- RiskAwareBacktester: Risk-managed version
- ExperimentalBacktester: Advanced features testing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Backtester:
    """
    Single-asset backtesting engine for evaluating trading strategies.

    This class implements a basic backtesting system for single-asset trading strategies.
    It handles trade execution, position tracking, and performance measurement.

    Features:
    ---------
    - Single asset position tracking
    - Transaction fee consideration
    - Basic risk management (position limits)
    - Performance metrics calculation (Sharpe, Sortino, Max DD)
    - Detailed trade logging

    Implementation Notes:
    -------------------
    - Uses a position variable to track current holdings
    - Maintains trade history and portfolio value history
    - Implements peak value tracking for drawdown calculation
    - Handles transaction fees for accurate PnL calculation

    Example:
    --------
    >>> data = pd.DataFrame(...)  # OHLCV data
    >>> backtester = Backtester(data, initial_balance=10000, trading_fee=0.001)
    >>> results = backtester.run(strategy, window_size=20)
    >>> print(f"Final portfolio value: {results['portfolio_values'][-1]}")
    """

    REQUIRED_COLUMNS = {"$open", "$high", "$low", "$close", "$volume"}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
    ):
        """
        Initialize the backtester with data and parameters.

        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns: $open, $high, $low, $close, $volume
        initial_balance : float, optional
            Starting balance for the portfolio (default: 10000.0)
        trading_fee : float, optional
            Transaction fee as a decimal (default: 0.001 = 0.1%)

        Notes:
        ------
        - Data columns must be prefixed with '$' (e.g., '$close')
        - Initializes internal state (position, balance, trades list)
        - Sets up logging for trade execution tracking
        """
        # Validate required columns
        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {list(missing_columns)}"
            )

        self.data = data.copy()
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.logger = logging.getLogger(self.__class__.__name__)

        self.reset()

    def reset(self):
        """
        Reset the backtester to initial state.

        This method resets all tracking variables to their initial values:
        - Portfolio values list (starts with initial_balance)
        - Trades list (empty)
        - Current position (0)
        - Current balance (initial_balance)
        - Peak portfolio value (initial_balance)
        """
        self.portfolio_values = [
            self.initial_balance
        ]  # Initialize with starting balance
        self.trades = []
        self.position = 0
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance

    def run(
        self,
        strategy: Union[Any, Any],
        window_size: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run backtest with given strategy.

        Parameters:
        -----------
        strategy : object
            Trading strategy object with get_action method
            Method should return float in [-1, 1] range
        window_size : int, optional
            Size of observation window for strategy (default: 20)
        verbose : bool, optional
            Whether to print progress (default: False)

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - metrics: Performance metrics (Sharpe, Sortino, etc.)
            - trades: List of all executed trades
            - portfolio_values: Historical portfolio values
            - timestamps: Corresponding timestamps

        Raises:
        -------
        ValueError
            If data length is less than window_size
        """
        self.reset()

        if len(self.data) < window_size:
            raise ValueError(
                f"Data length ({len(self.data)}) must be >= window_size ({window_size})"
            )

        try:
            # Run strategy
            for i in range(window_size, len(self.data)):
                # Get current window of data
                window_data = self.data.iloc[i - window_size : i].copy()
                current_data = self.data.iloc[i].copy()
                timestamp = current_data.name

                # Get strategy action
                try:
                    action = strategy.get_action(window_data)
                    if not isinstance(action, (int, float, np.ndarray)):
                        self.logger.warning(
                            f"Invalid action type: {type(action)}, expected float"
                        )
                        continue
                    action = float(action)  # Ensure action is float
                except Exception as e:
                    self.logger.error(
                        f"Error getting action from strategy: {str(e)}"
                    )
                    continue

                # Execute trade
                price_data = {
                    "$open": current_data["$open"],
                    "$high": current_data["$high"],
                    "$low": current_data["$low"],
                    "$close": current_data["$close"],
                    "$volume": current_data["$volume"],
                }
                trade_result = self.execute_trade(
                    timestamp, action, price_data
                )

                # Update portfolio value even if trade was skipped
                if "portfolio_value" not in trade_result:
                    portfolio_value = self._calculate_portfolio_value(
                        price_data["$close"]
                    )
                    self.portfolio_values.append(portfolio_value)
                    self.peak_value = max(self.peak_value, portfolio_value)

                if verbose and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(self.data)}")

        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Ensure portfolio values match the data length
        expected_length = len(self.data) - window_size + 1
        if len(self.portfolio_values) < expected_length:
            last_value = (
                self.portfolio_values[-1]
                if self.portfolio_values
                else self.initial_balance
            )
            self.portfolio_values.extend(
                [last_value] * (expected_length - len(self.portfolio_values))
            )
        elif len(self.portfolio_values) > expected_length:
            self.portfolio_values = self.portfolio_values[:expected_length]

        return {
            "metrics": metrics,
            "trades": self.trades,
            "portfolio_values": self.portfolio_values,
            "timestamps": self.data.index[window_size - 1 :].tolist(),
        }

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: float,
        price_data: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Execute trade based on strategy action.

        Parameters:
        -----------
        timestamp : pd.Timestamp
            Current timestamp for the trade
        action : float
            Strategy action in [-1, 1] range
            - Positive: Buy signal (size proportional to value)
            - Negative: Sell signal (size proportional to value)
        price_data : Dict[str, float]
            Current price data with keys:
            - $open, $high, $low, $close, $volume

        Returns:
        --------
        Dict[str, Any]
            Trade result containing:
            - timestamp: Trade timestamp
            - type: 'buy'/'sell'/'skip'
            - size: Trade size
            - price: Execution price
            - cost/revenue: Trade cost or revenue
            - balance: Updated balance
            - position: Updated position
            - action: Trade action
            - reason: Execution status reason

        Notes:
        ------
        - Implements basic position sizing
        - Handles transaction fees
        - Logs trade execution details
        - Cleans up dust positions
        """
        try:
            # Log trade attempt
            self.logger.info("Attempting trade execution - Timestamp: %s, Action: %.4f", timestamp, action)
            self.logger.debug("Price data: %s", price_data)
            
            # Validate price data
            required_columns = {"$open", "$high", "$low", "$close", "$volume"}
            missing_cols = required_columns - set(price_data.keys())
            if missing_cols:
                self.logger.warning("Missing required columns: %s", missing_cols)
                self.position = 0
                return {
                    "timestamp": timestamp,
                    "position": self.position,
                    "balance": self.balance,
                    "action": "error",
                    "reason": f"Missing required columns: {missing_cols}",
                }

            # Skip very small actions
            if abs(action) < 1e-4:  # Strict threshold for small actions
                self.logger.debug("Skipping trade - action too small: %.6f", action)
                return {
                    "timestamp": timestamp,
                    "position": self.position,
                    "balance": self.balance,
                    "action": "skip",
                    "reason": "action too small",
                }

            # Bound action between -1 and 1
            action = max(min(action, 1.0), -1.0)
            self.logger.debug("Bounded action: %.4f", action)

            current_price = price_data["$close"]
            self.logger.info("Current price: %.2f, Current balance: %.2f", current_price, self.balance)

            # Calculate trade size
            if action > 0:  # Buy
                # Check if balance is too low for any meaningful trade
                if self.balance < 10:  # Minimum balance requirement
                    self.logger.warning("Insufficient balance for minimum trade: %.2f", self.balance)
                    self.position = 0  # Reset position for insufficient balance
                    return {
                        "timestamp": timestamp,
                        "position": self.position,
                        "balance": self.balance,
                        "action": "skip",
                        "reason": "insufficient balance for minimum trade",
                    }

                max_shares = self.balance / (current_price * (1 + self.trading_fee))
                trade_shares = max_shares * abs(action)
                self.logger.debug("Buy calculation - Max shares: %.4f, Trade shares: %.4f", 
                               max_shares, trade_shares)

                # Skip if trade size is too small
                if trade_shares < 1e-6:
                    self.logger.warning("Trade size too small: %.6f", trade_shares)
                    self.position = 0  # Reset position for very small trades
                    return {
                        "timestamp": timestamp,
                        "position": self.position,
                        "balance": self.balance,
                        "action": "skip",
                        "reason": "trade size too small",
                    }

                cost = trade_shares * current_price * (1 + self.trading_fee)
                self.logger.info("Buy trade - Shares: %.4f, Cost: %.2f", trade_shares, cost)

                if cost > self.balance:  # Added balance check
                    self.logger.warning("Insufficient balance for trade - Cost: %.2f, Balance: %.2f",
                                     cost, self.balance)
                    self.position = 0  # Reset position for insufficient balance
                    return {
                        "timestamp": timestamp,
                        "position": self.position,
                        "balance": self.balance,
                        "action": "skip",
                        "reason": "insufficient balance",
                    }

                self.balance -= cost
                self.position += trade_shares

                # Clean up dust after buy
                if self.position < 1e-4:
                    self.logger.debug("Cleaning up dust position: %.6f", self.position)
                    self.position = 0

                trade = {
                    "timestamp": timestamp,
                    "entry_time": timestamp,
                    "type": "buy",
                    "size": trade_shares,
                    "price": current_price,
                    "cost": cost,
                    "balance": self.balance,
                    "position": self.position,
                    "action": "buy",
                    "reason": "trade executed",
                }

            else:  # Sell
                # Skip if no position to sell
                if self.position < 1e-4:
                    self.logger.debug("No position to sell: %.6f", self.position)
                    self.position = 0  # Clean up any dust
                    return {
                        "timestamp": timestamp,
                        "position": self.position,
                        "balance": self.balance,
                        "action": "skip",
                        "reason": "no position to sell",
                    }

                trade_shares = self.position * abs(action)
                self.logger.debug("Sell calculation - Position: %.4f, Trade shares: %.4f",
                               self.position, trade_shares)

                # Skip if trade size is too small
                if trade_shares < 1e-6:
                    self.logger.warning("Trade size too small: %.6f", trade_shares)
                    self.position = 0  # Reset position for very small trades
                    return {
                        "timestamp": timestamp,
                        "position": self.position,
                        "balance": self.balance,
                        "action": "skip",
                        "reason": "trade size too small",
                    }

                proceeds = trade_shares * current_price * (1 - self.trading_fee)
                self.logger.info("Sell trade - Shares: %.4f, Proceeds: %.2f", trade_shares, proceeds)

                self.balance += proceeds
                self.position -= trade_shares

                # Clean up any dust (very small remaining position)
                if self.position < 1e-4:
                    self.logger.debug("Cleaning up dust position: %.6f", self.position)
                    self.position = 0

                trade = {
                    "timestamp": timestamp,
                    "entry_time": timestamp,
                    "type": "sell",
                    "size": trade_shares,
                    "price": current_price,
                    "revenue": proceeds,
                    "balance": self.balance,
                    "position": self.position,
                    "action": "sell",
                    "reason": "trade executed",
                }

            self.trades.append(trade)
            self.logger.info("Trade executed successfully: %s", trade)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            self.peak_value = max(self.peak_value, portfolio_value)
            self.logger.info("Updated portfolio value: %.2f (Peak: %.2f)", 
                          portfolio_value, self.peak_value)

            trade["portfolio_value"] = portfolio_value
            return trade

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}", exc_info=True)
            self.position = 0  # Reset position on error
            return {
                "timestamp": timestamp,
                "error": str(e),
                "position": self.position,
                "balance": self.balance,
                "action": "error",
                "reason": str(e),
            }

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calculate total portfolio value.

        Parameters:
        -----------
        current_price : float
            Current asset price

        Returns:
        --------
        float
            Total portfolio value (balance + position_value)

        Notes:
        ------
        - Position value = position_size * current_price
        - Does not include unrealized fees
        """
        return self.balance + (self.position * current_price)

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate trading performance metrics.

        Returns:
        --------
        Dict[str, float]
            Dictionary containing:
            - total_return: Total portfolio return
            - sharpe_ratio: Sharpe ratio (annualized)
            - sortino_ratio: Sortino ratio (annualized)
            - max_drawdown: Maximum drawdown percentage
            - win_rate: Percentage of profitable trades

        Notes:
        ------
        - Assumes 252 trading days per year
        - Uses 0 as risk-free rate
        - Calculates ratios using daily returns
        """
        try:
            # Calculate returns
            values = np.array(self.portfolio_values)
            returns = np.diff(values) / values[:-1]

            # Total return
            total_return = (
                (values[-1] / values[0]) - 1 if len(values) > 1 else 0
            )

            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if len(returns) > 1
                else 0
            )

            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            sortino_ratio = (
                np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
                if len(negative_returns) > 0
                else 0
            )

            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0

            for value in values[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = min(max_drawdown, -drawdown)

            # Win rate
            profitable_trades = sum(
                1
                for trade in self.trades
                if (
                    "revenue" in trade
                    and trade["revenue"] > trade.get("cost", 0)
                )
                or (
                    "cost" in trade
                    and trade["cost"] < trade.get("revenue", float("inf"))
                )
            )
            total_trades = len(self.trades)
            win_rate = (
                profitable_trades / total_trades if total_trades > 0 else 0
            )

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "final_balance": self.balance,
                "final_portfolio_value": (
                    values[-1] if len(values) > 0 else self.balance
                ),
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "total_trades": len(self.trades),
                "win_rate": 0,
                "final_balance": self.balance,
                "final_portfolio_value": self.balance,
            }
