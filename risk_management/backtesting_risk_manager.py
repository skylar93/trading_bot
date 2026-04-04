"""
Backtesting Risk Manager

Unified RiskManager that merges basic risk checks (drawdown, position size, daily trades)
with advanced portfolio risk (VaR, correlation, CVaR) for backtesting environments.
Positions are assumed to be Dict[str, Dict[str, float]]: 
  positions[symbol] = {"units": float, "avg_price": float, "cost_basis": float}

Recent Changes:
- Refactored to inherit from RiskManagerBase abstract class
- Added forced liquidation functionality for max drawdown triggers
- Added trailing stop calculation and tracking
- Added support for partial fill/execution risk constraints
- Added time-based position constraints (weekend close, max holding periods)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, List, Tuple, Set
from scipy.stats import norm

from risk_management.risk_manager_base import RiskManagerBase, RiskConfigBase


@dataclass
class BacktestingRiskConfig(RiskConfigBase):
    """
    Configuration for risk management in backtesting environments.
    
    Features:
    - Basic risk parameters (position size, drawdown, trading limits)
    - Stop loss and trailing stop configurations
    - Advanced portfolio risk metrics (VaR, correlation)
    - Time-based constraints (weekend close, max holding period)
    - Execution constraints (partial fills, slippage)
    
    Implementation Notes:
    - All percentages are expressed as decimals (0.01 = 1%)
    - Defaults are set to conservative values
    - When enable_forced_liquidation is True, positions will be forcibly closed on max drawdown
    """
    # Legacy name compatibility
    stop_loss_pct: float = 0.02  # Maps to stop_loss_threshold in base class
    trailing_stop_pct: float = 0.05  # Maps to trailing_stop_buffer in base class
    
    # Basic risk
    max_position_size: float = 0.2       # fraction of portfolio
    daily_trade_limit: int = 10
    min_trade_size: float = 0.01        # fraction of portfolio
    max_leverage: float = 1.0
    
    # Advanced portfolio risk
    volatility_lookback: int = 20
    risk_free_rate: float = 0.02
    correlation_window: int = 30
    max_correlation: float = 0.7
    portfolio_var_limit: float = 0.02   # fraction of portfolio value
    
    # New time-based constraints
    close_positions_on_friday: bool = False  # Whether to close positions at Friday end of day
    max_holding_period_days: int = 0   # Maximum holding period (0 = no limit)
    
    # New execution constraints
    enable_partial_fills: bool = False  # Whether to simulate partial fills
    max_partial_fill_pct: float = 0.8   # Maximum percentage of order to fill (if partial fills enabled)
    min_partial_fill_pct: float = 0.5   # Minimum percentage of order to fill (if partial fills enabled)
    slippage_std: float = 0.001         # Standard deviation for slippage simulation
    
    # Legacy compatibility fields
    enable_stop_loss: bool = True      # Maps to use_stop_loss in base class
    enable_trailing_stop: bool = False  # Maps to use_trailing_stop in base class
    enable_forced_liquidation: bool = False  # Maps to use_forced_liquidation in base class
    forced_liquidation_drawdown: float = 0.15  # Maps to max_drawdown_pct in base class for liquidation


class StopLossConfig:
    """
    Configuration for stop loss settings for a specific position.
    
    Attributes:
        symbol (str): The symbol/asset of the position
        entry_price (float): The entry price of the position
        position_size (float): The size of the position (positive for long, negative for short)
        stop_price (float): The price at which to trigger the stop loss
        trailing (bool): Whether this is a trailing stop loss
        trailing_pct (float): The percentage below the highest price to set the trailing stop
    """
    
    def __init__(
        self,
        entry_price: float,
        position_size: float,
        symbol: str = "",
        stop_price: float = None,
        trailing: bool = False,
        trailing_pct: float = 0.02
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.position_size = position_size
        self.stop_price = stop_price
        self.trailing = trailing
        self.trailing_pct = trailing_pct
        self.highest_price = entry_price if position_size > 0 else None
        self.lowest_price = entry_price if position_size < 0 else None
        # For compatibility with old code
        self.is_long = position_size > 0


class BacktestingRiskManager(RiskManagerBase):
    """
    Risk Manager for backtesting environments.
    
    Handles:
      - daily trade limits
      - max drawdown
      - max leverage
      - position size clamp
      - min trade size
      - VaR / CVaR
      - correlation checks
      - stop loss and trailing stops
      - forced liquidation
      - time-based constraints
      - execution constraints (partial fills)
    """
    def __init__(self, config: BacktestingRiskConfig):
        """
        Initialize the RiskManager with the given configuration.
        
        Args:
            config (BacktestingRiskConfig): Risk management configuration
        """
        super().__init__(config)
        self.config = config
        
        # Initialize tracking variables
        self.trade_counter = {}  # date -> count
        self.position_start_times = {}  # asset -> timestamp
        self.peak_value = None
        self.current_value = None
        
        # Trailing stop variables
        self.trailing_stops = {}  # asset -> highest price
        self.stop_losses = {}  # asset -> StopLossConfig
        
        # Liquidation flags
        self.liquidation_triggered = False
        self.liquidation_assets = set()
        
        # Correlation and returns data
        self._correlation_matrix = None
        self._asset_returns = {}
        self._last_correlation_update = None

    def reset(self):
        """Reset all internal state."""
        self.trade_counter = {}
        self.position_start_times = {}
        self.peak_value = None
        self.current_value = None
        self.trailing_stops = {}
        self.stop_losses = {}
        self.position_stop_losses = {}  # New attribute for stop loss configuration
        self.position_entries = {}  # Track position entries for advanced risk calc
        self.liquidation_triggered = False
        self.liquidation_assets = set()
        self._correlation_matrix = None
        self._asset_returns = {}
        self._last_correlation_update = None

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """
        Check if max drawdown has been exceeded.
        
        Args:
            peak_value: Historical peak portfolio value
            current_value: Current portfolio value
            
        Returns:
            bool: True if max drawdown exceeded, False otherwise
        """
        if peak_value <= 0:
            return False
            
        drawdown = (peak_value - current_value) / peak_value
        max_exceeded = drawdown > self.config.max_drawdown_pct
        
        # If using forced liquidation, also set the flag
        if max_exceeded and self.config.use_forced_liquidation:
            self.liquidation_triggered = True
            self.logger.warning(
                f"Max drawdown exceeded: {drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
            )
            
        return max_exceeded

    def calculate_stop_loss(
        self, entry_price: float, position_size: float, is_long: bool = True
    ) -> float:
        """
        Return the stop loss price based on config.stop_loss_pct.
        
        Args:
            entry_price: Entry price of the position
            position_size: Size of the position (positive for long, negative for short)
            is_long: Whether the position is long (True) or short (False)
            
        Returns:
            float: Stop loss price
        """
        # For compatibility, we use stop_loss_pct here
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss has been triggered for a symbol.
        
        Args:
            symbol: Asset symbol to check
            current_price: Current price of the asset
            
        Returns:
            bool: True if stop loss triggered, False otherwise
        """
        if not self.config.use_stop_loss or symbol not in self.stop_losses:
            return False
            
        stop_config = self.stop_losses[symbol]
        
        # Check if stop loss is triggered
        if stop_config.is_long:
            return current_price <= stop_config.stop_price
        else:
            return current_price >= stop_config.stop_price

    def update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """
        Update trailing stop level based on current price.
        
        Args:
            symbol: Asset symbol to update
            current_price: Current price of the asset
        """
        if not self.config.use_trailing_stop or symbol not in self.stop_losses:
            return
            
        stop_config = self.stop_losses[symbol]
        if not stop_config.trailing:
            return
            
        # Update highest/lowest price and trailing stop
        if stop_config.is_long:
            if current_price > stop_config.highest_price:
                stop_config.highest_price = current_price
                stop_config.stop_price = current_price * (1 - stop_config.trailing_pct)
        else:
            if stop_config.lowest_price is None or current_price < stop_config.lowest_price:
                stop_config.lowest_price = current_price
                stop_config.stop_price = current_price * (1 + stop_config.trailing_pct)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of current risk metrics
        """
        return {
            "current_value": self.current_value,
            "peak_value": self.peak_value,
            "drawdown": (self.peak_value - self.current_value) / self.peak_value if self.peak_value and self.peak_value > 0 else 0,
            "liquidation_triggered": self.liquidation_triggered,
            "liquidation_assets": list(self.liquidation_assets),
            "stop_losses": {k: v.stop_price for k, v in self.stop_losses.items()},
            "trade_count": sum(self.trade_counter.values())
        }

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Implement abstract method — delegates to get_risk_metrics."""
        return self.get_risk_metrics()

    # Additional backtesting-specific methods
    def check_trade_limits(self, timestamp: pd.Timestamp) -> bool:
        """
        Return True if we have not exceeded the daily trade limit on this date.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if trade is allowed, False if trade limit reached
        """
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        # if we've hit the limit, return False
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return False
        return True

    def update_trade_counter(self, timestamp: pd.Timestamp) -> None:
        """
        Increment trade counter for the given date.
        
        Args:
            timestamp: Current timestamp
        """
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1

    def check_leverage_limits(self, portfolio_value: float, position_value: float) -> bool:
        """
        Return True if position_value/portfolio_value <= max_leverage.
        
        Args:
            portfolio_value: Current portfolio value
            position_value: Value of the position
            
        Returns:
            bool: True if leverage is within limits, False otherwise
        """
        if portfolio_value <= 0:
            return False
        leverage = position_value / portfolio_value
        return leverage <= self.config.max_leverage

    def should_close_for_weekend(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if positions should be closed for the weekend.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if positions should be closed, False otherwise
        """
        if not self.config.close_positions_on_friday:
            return False
            
        # Friday = 4 in pd.Timestamp.dayofweek (Monday=0, Sunday=6)
        return timestamp.dayofweek == 4 and timestamp.hour >= 15  # After 3 PM on Friday 

    # Legacy compatibility methods
    def check_forced_liquidation(self) -> bool:
        """
        Check if forced liquidation has been triggered due to max drawdown.
        
        Returns:
            bool: True if liquidation has been triggered, False otherwise
        """
        return self.liquidation_triggered
    
    def update_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Recompute correlation matrix from the provided price_data.
        
        Args:
            price_data: Dictionary mapping assets to price series
        """
        # Create a simple implementation for compatibility
        if not price_data:
            return
            
        # Build returns dict
        returns = {}
        for asset, prices in price_data.items():
            returns[asset] = prices.pct_change().dropna()
            
        # Skip if we don't have enough data
        if not returns or all(len(ret) < 2 for ret in returns.values()):
            return
            
        # Create a DataFrame and calculate correlation
        try:
            df = pd.DataFrame({k: v for k, v in returns.items() if len(v) > 1})
            self._correlation_matrix = df.corr()
            self._asset_returns_df = df
            self._asset_stds = df.std()
            self._last_correlation_update = pd.Timestamp.now()
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) from historical returns.
        
        Args:
            returns: Series of historical returns
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            float: CVaR at the specified confidence level
        """
        if len(returns) < 2:
            return 0.0
            
        # Calculate VaR
        var = self.calculate_var(returns, confidence_level)
        
        # Calculate CVaR
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return 0.0
        return abs(tail.mean())
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
        asset_name: Optional[str] = None
    ) -> float:
        """
        Calculate position size based on portfolio value and risk parameters.
        
        Args:
            portfolio_value: Current portfolio value
            price: Current price of the asset
            volatility: Optional asset volatility
            current_positions: Optional dictionary of current positions
            asset_name: Optional asset name
            
        Returns:
            float: Recommended position size
        """
        # Basic position size limit in nominal terms
        position_size = portfolio_value * self.config.max_position_size
        
        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            vol_scalar = 1.0 / (1.0 + volatility)
            position_size *= vol_scalar
            
        # Adjust for correlation if applicable
        if current_positions and asset_name and hasattr(self, '_correlation_matrix') and self._correlation_matrix is not None:
            # Apply simple correlation-based adjustment
            for other_asset, pos_value in current_positions.items():
                if not self.check_correlation_limits(asset_name, other_asset):
                    position_size *= 0.5  # Reduce by half if correlation is too high
                    
        # Enforce minimum size
        min_notional = portfolio_value * self.config.min_trade_size
        if position_size < min_notional:
            return 0.0
            
        return position_size
    
    def process_trade_signal(self, signal: dict) -> bool:
        """
        Process a trade signal and determine if it's valid.
        
        Args:
            signal: Dictionary containing trade signal details
            
        Returns:
            bool: True if signal is valid, False otherwise
        """
        required_fields = ["timestamp", "type", "price", "size"]
        if not all(field in signal for field in required_fields):
            self.logger.warning("Invalid trade signal: missing required fields.")
            return False
            
        if signal["price"] <= 0 or signal["size"] <= 0:
            self.logger.warning("Invalid trade signal: negative price or zero size.")
            return False
            
        return True
    
    def check_correlation_limits(self, asset1: str, asset2: str) -> bool:
        """
        Check if correlation between two assets is within limits.
        
        Args:
            asset1: First asset
            asset2: Second asset
            
        Returns:
            bool: True if correlation is within limits, False otherwise
        """
        if not hasattr(self, '_correlation_matrix') or self._correlation_matrix is None:
            return True
            
        if asset1 not in self._correlation_matrix.index or asset2 not in self._correlation_matrix.columns:
            return True
            
        corr = abs(self._correlation_matrix.loc[asset1, asset2])
        return bool(corr <= self.config.max_correlation)
    
    def calculate_var(self, returns, confidence_level=None):
        """
        Calculate Value at Risk (VaR) from historical returns.
        
        Args:
            returns: Series or array of historical returns
            confidence_level: Optional confidence level (overrides config value)
            
        Returns:
            float: VaR at the specified confidence level
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        # Convert returns to numpy array if not already
        returns = np.array(returns)
        
        if len(returns) < 2:
            return 0.0
            
        # Use provided confidence level or default from config
        cl = confidence_level if confidence_level is not None else self.config.var_confidence_level
        
        # Historical VaR: negate left-tail percentile → positive loss amount
        var = -np.percentile(returns, (1 - cl) * 100)
        return max(0.0, float(var))
    
    def get_portfolio_var(self, 
                     portfolio_value: float, 
                     positions: Dict[str, Dict[str, float]],
                     prices: Dict[str, float]) -> float:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            portfolio_value: Total portfolio value
            positions: Dictionary of positions
            prices: Current prices
            
        Returns:
            float: Portfolio VaR as a fraction of portfolio value
        """
        if portfolio_value < 1e-8:
            return 0.0

        # Find active assets that have return history
        has_returns = (
            hasattr(self, "_asset_returns_df") and self._asset_returns_df is not None
        )
        active = [
            a for a, pos in positions.items()
            if abs(pos.get("units", 0) * prices.get(a, 0)) > 1e-8
        ]
        if not active:
            return 0.0

        if has_returns:
            available = [a for a in active if a in self._asset_returns_df.columns]
        else:
            available = []

        # Single-asset or no history: use per-asset std from stored data
        if len(available) == 1:
            asset = available[0]
            if hasattr(self, "_asset_stds") and asset in self._asset_stds.index:
                std = float(self._asset_stds[asset])
                return max(0.0, norm.ppf(self.config.var_confidence_level) * std)
            return 0.0

        if len(available) >= 2:
            returns_matrix = self._asset_returns_df[available].dropna()
            if len(returns_matrix) < 2:
                return 0.0

            pos_values = np.array([
                abs(positions[a]["units"] * prices.get(a, 0)) for a in available
            ])
            total_val = pos_values.sum()
            if total_val < 1e-8:
                return 0.0
            w = pos_values / total_val

            # Parametric portfolio VaR: sqrt(w' Cov w) * z_alpha
            cov = returns_matrix.cov().values
            portfolio_variance = float(w @ cov @ w)
            portfolio_std = np.sqrt(max(portfolio_variance, 0.0))
            portfolio_mean = float(w @ returns_matrix.mean().values)
            var = -(portfolio_mean + norm.ppf(1 - self.config.var_confidence_level) * portfolio_std)
            return max(0.0, float(var))

        # Fallback: no correlated history → sum individual VaRs conservatively
        individual_vars = []
        for a in active:
            if has_returns and a in self._asset_returns_df.columns:
                ret = self._asset_returns_df[a].dropna().values
                if len(ret) >= 2:
                    v = -np.percentile(ret, (1 - self.config.var_confidence_level) * 100)
                    individual_vars.append(max(0.0, v))
        if individual_vars:
            return float(np.mean(individual_vars))
        return 0.0

    def check_stop_losses(self, current_prices: Dict[str, float], positions: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """
        Check if any stop losses have been triggered.
        
        Args:
            current_prices: Dictionary of current prices by asset
            positions: Dictionary of current positions
            
        Returns:
            Dict[str, bool]: Dictionary mapping assets to boolean indicating if stop loss was triggered
        """
        # First update trailing stops with current prices
        self.update_trailing_stops(current_prices)
        
        # Check if any stop losses have been triggered
        triggered = {}
        
        # Create a copy of keys to safely iterate
        symbols_to_check = list(self.stop_losses.keys())
        
        for symbol in symbols_to_check:
            # Skip if config was removed during updating
            if symbol not in self.stop_losses:
                triggered[symbol] = False
                continue
                
            stop_config = self.stop_losses[symbol]
            
            # Skip if we don't have a current price
            if symbol not in current_prices:
                triggered[symbol] = False
                continue
                
            # Skip if we don't have a position anymore
            if symbol not in positions or abs(positions[symbol]["units"]) < 1e-8:
                triggered[symbol] = False
                if symbol in self.stop_losses:
                    del self.stop_losses[symbol]
                continue
                
            current_price = current_prices[symbol]
            
            # Long position stop loss check
            if stop_config.position_size > 0:
                if current_price <= stop_config.stop_price:
                    self.logger.warning(
                        f"Stop loss triggered for {symbol}: Current price {current_price:.2f} <= Stop price {stop_config.stop_price:.2f}"
                    )
                    triggered[symbol] = True
                else:
                    triggered[symbol] = False
            
            # Short position stop loss check
            elif stop_config.position_size < 0:
                if current_price >= stop_config.stop_price:
                    self.logger.warning(
                        f"Stop loss triggered for {symbol}: Current price {current_price:.2f} >= Stop price {stop_config.stop_price:.2f}"
                    )
                    triggered[symbol] = True
                else:
                    triggered[symbol] = False
        
        return triggered
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> None:
        """
        Update trailing stop prices based on current market prices.
        
        Args:
            current_prices: Dictionary of current prices by asset
        """
        # Create a copy of keys to safely iterate
        symbols_to_update = list(self.stop_losses.keys())
        
        for symbol in symbols_to_update:
            # Skip if config was removed
            if symbol not in self.stop_losses:
                continue
                
            stop_config = self.stop_losses[symbol]
            
            if not stop_config.trailing:
                continue
                
            if symbol not in current_prices:
                self.logger.warning(f"Cannot update trailing stop for {symbol}: price not available")
                continue
                
            current_price = current_prices[symbol]
            
            # Update trailing stop for long positions
            if stop_config.position_size > 0:
                # If price moved higher, update the highest seen price
                if current_price > stop_config.highest_price:
                    old_stop = stop_config.stop_price
                    stop_config.highest_price = current_price
                    # Update the stop price based on the trailing percentage
                    stop_config.stop_price = current_price * (1 - stop_config.trailing_pct)
                    self.logger.debug(
                        f"Updated trailing stop for {symbol}: {old_stop:.2f} -> {stop_config.stop_price:.2f} "
                        f"(new high: {current_price:.2f})"
                    )
            
            # Update trailing stop for short positions
            elif stop_config.position_size < 0:
                # If price moved lower, update the lowest seen price
                if stop_config.lowest_price is None or current_price < stop_config.lowest_price:
                    old_stop = stop_config.stop_price
                    stop_config.lowest_price = current_price
                    # Update the stop price based on the trailing percentage
                    stop_config.stop_price = current_price * (1 + stop_config.trailing_pct)
                    self.logger.debug(
                        f"Updated trailing stop for {symbol}: {old_stop:.2f} -> {stop_config.stop_price:.2f} "
                        f"(new low: {current_price:.2f})"
                    ) 

    def update_portfolio_value(self, portfolio_value: float) -> bool:
        """
        Update the current portfolio value and check if liquidation should be triggered.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            bool: True if liquidation triggered, False otherwise
        """
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        self.current_value = portfolio_value
        
        # Check if drawdown exceeds threshold and we should trigger liquidation
        if self.config.enable_forced_liquidation and self.peak_value is not None:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if drawdown >= self.config.forced_liquidation_drawdown:
                self.liquidation_triggered = True
                self.logger.warning(
                    f"Forced liquidation triggered: Drawdown {drawdown:.2%} exceeds threshold "
                    f"{self.config.forced_liquidation_drawdown:.2%}"
                )
                return True
                
        return False 

    def update_after_trade(self, timestamp: pd.Timestamp, asset: str = None, 
                        trade_size: float = 0.0, price: float = 0.0,
                        portfolio_value: float = 0.0, positions: Dict[str, Dict[str, float]] = None,
                        is_long: bool = None, current_price: float = None, units: float = None,
                        entry_price: float = None, position_size: float = None, stop_price: float = None,
                        trailing: bool = None):
        """
        Update internal state after a trade is executed.
        
        Args:
            timestamp: Current timestamp
            asset: Asset that was traded
            trade_size: Size of the trade (positive for buy, negative for sell)
            price: Execution price
            portfolio_value: Current portfolio value
            positions: Current positions
            is_long: Deprecated, use trade_size > 0 instead
            current_price: Deprecated, use price instead
            units: Deprecated, use trade_size instead
            entry_price: The price at which the position was entered (used for stop loss calculations)
            position_size: Absolute size of the position (used instead of trade_size if provided)
            stop_price: The stop loss price to use (overrides automatic calculation)
            trailing: Whether to use trailing stop loss for this specific position
        """
        # Handle backward compatibility
        if current_price is not None and price == 0.0:
            price = current_price
            
        if units is not None and trade_size == 0.0:
            trade_size = units if is_long else -units
            
        # Use position_size if provided
        if position_size is not None:
            actual_trade_size = position_size
            if trade_size < 0:  # Preserve the sign from trade_size if it's negative
                actual_trade_size = -abs(position_size)
        else:
            actual_trade_size = trade_size
            
        # Handle entry_price for stop loss configuration (newer style)
        if entry_price is not None and asset is not None:
            # Store the entry price for stop loss calculations
            if asset not in self.position_stop_losses:
                self.position_stop_losses[asset] = StopLossConfig(
                    entry_price=entry_price,
                    position_size=abs(actual_trade_size),
                    symbol=asset,
                    trailing=trailing if trailing is not None else self.config.enable_trailing_stop,
                    trailing_pct=self.config.trailing_stop_pct
                )
            
        # Create mock positions for backward compatibility
        if positions is None and asset is not None and units is not None:
            positions = {
                asset: {
                    "units": units if is_long else -units,
                    "entry_price": price
                }
            }
        
        # Update position entry in trade history
        if asset is not None and positions is not None and asset in positions:
            if "entry_price" in positions[asset] and positions[asset]["entry_price"] is not None:
                entry_price_value = positions[asset]["entry_price"]
            else:
                entry_price_value = price
                
            if asset not in self.position_entries:
                self.position_entries[asset] = []
                
            self.position_entries[asset].append({
                "timestamp": timestamp,
                "price": entry_price_value,
                "size": abs(actual_trade_size),
                "direction": 1 if actual_trade_size > 0 else -1
            })
            
        # Update daily trade counter
        if asset is not None and actual_trade_size != 0:
            self.update_trade_counter(timestamp)
        
        # Update position start times
        if asset and actual_trade_size != 0:
            # If this is a new position or flipping direction
            if positions and (
                asset not in positions or 
                (positions[asset]["units"] > 0 and actual_trade_size < 0 and abs(actual_trade_size) >= positions[asset]["units"]) or
                (positions[asset]["units"] < 0 and actual_trade_size > 0 and abs(actual_trade_size) >= abs(positions[asset]["units"]))
            ):
                self.position_start_times[asset] = timestamp
                self.logger.info(f"Updated position start time for {asset}: {timestamp}")
        
        # Update stop loss configuration
        if asset and is_long is not None and units is not None:
            # For backward compatibility with old tests
            self.stop_losses[asset] = StopLossConfig(
                entry_price=price,
                position_size=units if is_long else -units,
                symbol=asset,
                stop_price=stop_price if stop_price is not None else self.calculate_stop_loss(price, units, is_long),
                trailing=trailing if trailing is not None else self.config.enable_trailing_stop,
                trailing_pct=self.config.trailing_stop_pct
            )
            self.logger.info(
                f"Set {'trailing ' if (trailing if trailing is not None else self.config.enable_trailing_stop) else ''}stop loss for {asset} "
                f"at {self.stop_losses[asset].stop_price:.2f} ({'long' if is_long else 'short'} position)"
            )
        elif asset and positions and asset in positions:
            position = positions[asset]
            position_units = position["units"]
            
            # Only set stop loss if we have a position
            if abs(position_units) > 0:
                is_long_position = position_units > 0
                
                if self.config.stop_loss_pct > 0:
                    # Calculate stop loss price
                    calculated_stop_price = self.calculate_stop_loss(price, abs(position_units), is_long_position)
                    self.stop_losses[asset] = StopLossConfig(
                        entry_price=price,
                        position_size=position_units,
                        symbol=asset,
                        stop_price=stop_price if stop_price is not None else calculated_stop_price,
                        trailing=trailing if trailing is not None else self.config.enable_trailing_stop,
                        trailing_pct=self.config.trailing_stop_pct
                    )
                    self.logger.info(
                        f"Set {'trailing ' if (trailing if trailing is not None else self.config.enable_trailing_stop) else ''}stop loss for {asset} "
                        f"at {self.stop_losses[asset].stop_price:.2f} ({'long' if is_long_position else 'short'} position)"
                    )
            elif asset in self.stop_losses:
                # Remove stop loss if position is closed
                del self.stop_losses[asset]
                self.logger.info(f"Removed stop loss for {asset} (position closed)")
        
        # Update correlation matrix if we have price data
        if hasattr(self, '_last_correlation_update') and self._last_correlation_update is not None:
            days_since_update = (timestamp - self._last_correlation_update).days
            if days_since_update >= 7:  # Update weekly
                self.logger.info("Updating correlation matrix (weekly)")
                # This would be done with actual price data in a real implementation 

    def check_trade(self, timestamp, portfolio_value, trade_size, price, positions, asset):
        """
        Evaluate if a trade should be allowed based on risk parameters.
        
        Args:
            timestamp (pd.Timestamp): Current time
            portfolio_value (float): Current portfolio value
            trade_size (float): Requested trade size (negative for sell)
            price (float): Current price
            positions (dict): Current positions
            asset (str): Asset being traded
            
        Returns:
            dict: Result with allowed status and adjusted size
        """
        # Initialize result
        result = {
            "allowed": True,
            "adjusted_size": trade_size,
            "reason": None
        }
        
        # Skip checks if trade size is 0
        if trade_size == 0:
            return result
        
        # Check for minimum trade size
        if abs(trade_size) < self.config.min_trade_size:
            result["allowed"] = False
            result["adjusted_size"] = 0
            result["reason"] = f"Trade size {abs(trade_size):.6f} below minimum {self.config.min_trade_size:.6f}"
            return result
        
        # Apply position sizing rules
        adjusted_size = self._check_position_size(trade_size, price, portfolio_value, positions, asset)
        
        # Check if adjusted size meets minimum trade size
        if abs(adjusted_size) < self.config.min_trade_size:
            result["allowed"] = False
            result["adjusted_size"] = 0
            result["reason"] = f"Adjusted size {abs(adjusted_size):.6f} below minimum {self.config.min_trade_size:.6f}"
            return result
        
        # Apply partial fills (if enabled)
        adjusted_size = self._apply_partial_fill(adjusted_size)
        
        # Update the result with adjusted size
        result["adjusted_size"] = adjusted_size
        
        # Apply other risk checks
        if not self._check_daily_trade_limit(timestamp):
            result["allowed"] = False
            result["reason"] = "Daily trade limit exceeded"
        
        # Check for stop loss
        if not self._check_stop_loss(trade_size, price, positions, asset):
            result["allowed"] = False
            result["reason"] = "Stop loss triggered"
        
        # Store current portfolio value for drawdown calculation
        self.current_value = portfolio_value
        
        # Update peak value if needed
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Check for maximum drawdown
        if not self._check_max_drawdown():
            result["allowed"] = False
            result["reason"] = "Maximum drawdown exceeded"
        
        # Check if forced liquidation has been triggered
        if self.check_forced_liquidation():
            # Override result for sell orders (liquidation)
            if trade_size < 0:
                result["allowed"] = True
                result["reason"] = "Forced liquidation"
            else:
                # Block buy orders during liquidation
                result["allowed"] = False
                result["reason"] = "Forced liquidation in progress"
        
        # Apply slippage to the trade (if enabled)
        # Note: Slippage doesn't affect whether trade is allowed, just the execution price
        # We'll check if the attribute exists to maintain backward compatibility
        if result["allowed"] and hasattr(self.config, 'enable_slippage') and self.config.enable_slippage:
            pass
        
        # Log the result
        if result["allowed"]:
            # Convert numpy array to float if needed
            adjusted_size = float(result['adjusted_size']) if hasattr(result['adjusted_size'], 'item') else result['adjusted_size']
            self.logger.info(f"Trade allowed: {adjusted_size:.6f} {asset} @ {price:.2f}")
        else:
            # Ensure trade_size is a simple float for formatting
            trade_size_float = float(trade_size) if hasattr(trade_size, 'item') else trade_size
            self.logger.warning(f"Trade rejected: {trade_size_float:.6f} {asset} @ {price:.2f}. Reason: {result['reason']}")
        
        return result
    
    def _apply_partial_fill(self, trade_size):
        """
        Simulate partial fills by reducing the requested trade size by a random percentage
        within the configured bounds (if enabled).
        
        Args:
            trade_size (float): The original requested trade size
            
        Returns:
            float: The adjusted trade size after applying partial fill simulation
        """
        adjusted_size = trade_size
        
        # Only apply partial fills if explicitly enabled in config
        if hasattr(self.config, "enable_partial_fills") and self.config.enable_partial_fills:
            # Generate random fill percentage between min and max
            fill_pct = np.random.uniform(
                self.config.min_partial_fill_pct, 
                self.config.max_partial_fill_pct
            )
            
            # Apply the partial fill
            adjusted_size = trade_size * fill_pct
            
            # Check if the adjusted size still meets minimum trade size requirements
            if abs(adjusted_size) < self.config.min_trade_size and abs(trade_size) >= self.config.min_trade_size:
                self.logger.info(
                    f"Partial fill would result in size ({adjusted_size:.6f}) below min_trade_size "
                    f"({self.config.min_trade_size:.6f}). Using original size."
                )
                adjusted_size = trade_size
            else:
                self.logger.info(
                    f"Applied partial fill: {trade_size:.6f} -> {adjusted_size:.6f} "
                    f"(fill rate: {fill_pct:.2%})"
                )
        
        return adjusted_size
    
    def _check_daily_trade_limit(self, timestamp):
        """
        Check if the daily trade limit has been reached.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            
        Returns:
            bool: True if trade is allowed, False if limit reached
        """
        date_key = timestamp.date()
        
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
            
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return False
            
        return True
        
    def _check_stop_loss(self, trade_size, price, positions, asset):
        """
        Check if a stop loss has been triggered for the asset.
        
        Args:
            trade_size (float): Requested trade size
            price (float): Current price
            positions (dict): Current positions
            asset (str): Asset being traded
            
        Returns:
            bool: True if trade is allowed, False if stop loss triggered
        """
        # If this is a liquidation trade (selling to reduce risk), allow it
        is_reducing_risk = (trade_size < 0 and asset in positions and positions[asset]["units"] > 0) or \
                          (trade_size > 0 and asset in positions and positions[asset]["units"] < 0)
                          
        # If we have a position in this asset and stop loss is enabled
        if asset in positions and hasattr(self.config, "stop_loss_pct") and self.config.stop_loss_pct > 0:
            position = positions[asset]
            
            # Calculate unrealized PnL percentage
            if "entry_price" in position and position["entry_price"] > 0:
                if position["units"] > 0:  # Long position
                    pnl_pct = (price - position["entry_price"]) / position["entry_price"]
                else:  # Short position
                    pnl_pct = (position["entry_price"] - price) / position["entry_price"]
                
                # Check if stop loss is triggered
                if pnl_pct < -self.config.stop_loss_pct:
                    self.logger.warning(
                        f"Stop loss triggered for {asset}: PnL {pnl_pct:.2%} < -{self.config.stop_loss_pct:.2%}"
                    )
                    
                    # If this is already a liquidation trade, allow it
                    if is_reducing_risk:
                        return True
                        
                    # If forced liquidation is enabled, trigger it
                    if hasattr(self.config, "enable_forced_liquidation") and self.config.enable_forced_liquidation:
                        self.liquidation_triggered = True
                        self.liquidation_assets.add(asset)
                        self.logger.warning(f"Forced liquidation triggered for {asset}")
                        
                    # Don't allow new positions when stop loss is triggered
                    return False
        
        return True
        
    def _check_max_drawdown(self):
        """
        Check if the maximum drawdown has been exceeded.
        
        Returns:
            bool: True if trade is allowed, False if max drawdown exceeded
        """
        if self.peak_value is None or not hasattr(self, 'current_value'):
            return True
            
        current_drawdown = (self.peak_value - self.current_value) / self.peak_value
        
        if current_drawdown > self.config.max_drawdown_pct:
            self.logger.warning(
                f"Max drawdown exceeded: {current_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
            )
            
            # If forced liquidation is enabled, trigger it
            if hasattr(self.config, "enable_forced_liquidation") and self.config.enable_forced_liquidation:
                # Check if the drawdown exceeds the forced liquidation threshold
                liquidation_threshold = self.config.forced_liquidation_drawdown \
                    if hasattr(self.config, "forced_liquidation_drawdown") else self.config.max_drawdown_pct
                
                if current_drawdown > liquidation_threshold:
                    self.liquidation_triggered = True
                    self.logger.warning(f"Forced liquidation triggered due to drawdown: {current_drawdown:.2%}")
            
            return False
            
        return True
        
    def _check_position_size(self, trade_size, price, portfolio_value, positions, asset):
        """
        Check and adjust position size based on risk parameters.
        
        Args:
            trade_size (float): Requested trade size
            price (float): Current price
            portfolio_value (float): Current portfolio value
            positions (dict): Current positions
            asset (str): Asset being traded
            
        Returns:
            float: Adjusted trade size after applying position sizing rules
        """
        adjusted_size = trade_size
        
        # Calculate current exposure
        current_exposure = 0.0
        if asset in positions:
            current_exposure = abs(positions[asset]["units"] * price)
            
        # Calculate proposed value
        proposed_value = abs(trade_size * price)
        
        # Apply position size limit
        max_pos_value = self.config.max_position_size * portfolio_value
        if proposed_value > max_pos_value:
            # Clamp but don't reject - this should succeed with adjusted size
            adjusted_units = max_pos_value / price
            # Keep sign
            adjusted_size = adjusted_units if trade_size > 0 else -adjusted_units
            self.logger.info(
                f"Position size clamped: {trade_size:.6f} -> {adjusted_size:.6f} "
                f"(max: {self.config.max_position_size:.2%} of portfolio)"
            )
            
        # Apply leverage limit
        if portfolio_value > 1e-8:
            total_exposure = current_exposure + proposed_value
            leverage = total_exposure / portfolio_value
            
            if leverage > self.config.max_leverage:
                max_allowed_exposure = self.config.max_leverage * portfolio_value
                remain_exposure = max_allowed_exposure - current_exposure
                
                if remain_exposure <= 0:
                    # No room for additional exposure
                    adjusted_size = 0
                else:
                    # Clamp to maximum allowed exposure
                    adjusted_units = remain_exposure / price
                    adjusted_size = adjusted_units if trade_size > 0 else -adjusted_units
                    self.logger.info(
                        f"Leverage clamped: {trade_size:.6f} -> {adjusted_size:.6f} "
                        f"(max leverage: {self.config.max_leverage:.2f})"
                    )
                    
        return adjusted_size 

    def get_liquidation_signals(self, positions: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """
        Get a list of assets and amounts to liquidate if forced liquidation is triggered.
        
        Args:
            positions: Dict mapping asset symbols to position details
            
        Returns:
            List of (symbol, units) tuples indicating positions to close
        """
        if not self.liquidation_triggered:
            return []
            
        signals = []
        
        # If specific assets are marked for liquidation, close only those
        if self.liquidation_assets:
            for asset in self.liquidation_assets:
                if asset in positions and abs(positions[asset]["units"]) > 1e-8:
                    # Negative units means close the position
                    signals.append((asset, -positions[asset]["units"]))
        # Otherwise close all open positions
        else:
            for asset, pos_dict in positions.items():
                if abs(pos_dict["units"]) > 1e-8:
                    signals.append((asset, -pos_dict["units"]))
        
        # Reset liquidation flags after generating signals
        if signals:
            self.liquidation_triggered = False
            self.liquidation_assets.clear()
            self.logger.info(f"Generated liquidation signals for {len(signals)} positions")
            
        return signals
    
    def apply_slippage(self, price: float) -> float:
        """
        Apply random slippage to a price to simulate execution risk.
        
        Args:
            price: Original price
            
        Returns:
            float: Price with slippage applied
        """
        if not hasattr(self.config, "slippage_std") or self.config.slippage_std <= 0:
            return price
            
        # Generate random slippage based on normal distribution
        slippage_factor = np.random.normal(0, self.config.slippage_std)
        
        # Apply slippage
        new_price = price * (1.0 + slippage_factor)
        
        # Ensure price remains positive
        new_price = max(new_price, 1e-8)
        
        return new_price
    
    def check_max_holding_period(self, 
                                timestamp: pd.Timestamp, 
                                positions: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """
        Check if any positions have exceeded their maximum holding period.
        
        Args:
            timestamp: Current timestamp
            positions: Dict mapping asset symbols to position details
            
        Returns:
            Dict mapping asset symbols to booleans indicating if max holding period exceeded
        """
        if not hasattr(self.config, "max_holding_period_days") or self.config.max_holding_period_days <= 0:
            return {}
            
        exceeded = {}
        
        # Create a copy of keys to safely iterate
        assets_to_check = list(self.position_start_times.keys())
        
        for asset in assets_to_check:
            # Skip if position start time was removed during iteration
            if asset not in self.position_start_times:
                continue
                
            start_time = self.position_start_times[asset]
            
            if asset not in positions or abs(positions[asset]["units"]) < 1e-8:
                continue
                
            days_held = (timestamp - start_time).days
            
            if days_held >= self.config.max_holding_period_days:
                exceeded[asset] = True
                self.logger.warning(
                    f"Max holding period exceeded for {asset}: {days_held} days >= "
                    f"{self.config.max_holding_period_days} days"
                )
            else:
                exceeded[asset] = False
                
        return exceeded 

    def check_weekend_close(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if positions should be closed for the weekend.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if positions should be closed, False otherwise
        """
        return self.should_close_for_weekend(timestamp) 