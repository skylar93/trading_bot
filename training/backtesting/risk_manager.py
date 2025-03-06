"""
Unified RiskManager that merges basic risk checks (drawdown, position size, daily trades)
with advanced portfolio risk (VaR, correlation, CVaR).
Positions are assumed to be Dict[str, Dict[str, float]]: 
  positions[symbol] = {"units": float, "avg_price": float, "cost_basis": float}
Also includes legacy methods (check_max_drawdown, calculate_stop_loss, etc.)
to satisfy older tests in test_risk_management.py

Recent Changes:
- Added forced liquidation functionality for max drawdown triggers
- Added trailing stop calculation and tracking
- Added support for partial fill/execution risk constraints
- Added time-based position constraints (weekend close, max holding periods)
- Fixed forced liquidation detection to properly trigger on drawdown exceeds
- Fixed position tracking in weekend and max holding period checks
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, List, Tuple, Set


@dataclass
class RiskConfig:
    """
    Merged configuration for risk management.
    
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
    # Basic risk
    max_position_size: float = 0.2       # fraction of portfolio
    stop_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    daily_trade_limit: int = 10
    min_trade_size: float = 0.01        # fraction of portfolio
    max_leverage: float = 1.0
    
    # Advanced portfolio risk
    volatility_lookback: int = 20
    risk_free_rate: float = 0.02
    var_confidence_level: float = 0.95
    correlation_window: int = 30
    max_correlation: float = 0.7
    portfolio_var_limit: float = 0.02   # fraction of portfolio value
    
    # New stop loss parameters
    trailing_stop_pct: float = 0.05    # Default trailing stop percentage
    enable_stop_loss: bool = True      # Whether to enforce stop losses
    enable_trailing_stop: bool = False  # Whether to use trailing stops
    
    # New forced liquidation parameters
    enable_forced_liquidation: bool = False  # Whether to force liquidation on max drawdown
    forced_liquidation_drawdown: float = 0.15  # Drawdown threshold for forced liquidation
    
    # New time-based constraints
    close_positions_on_friday: bool = False  # Whether to close positions at Friday end of day
    max_holding_period_days: int = 0   # Maximum holding period (0 = no limit)
    
    # New execution constraints
    enable_partial_fills: bool = False  # Whether to simulate partial fills
    max_partial_fill_pct: float = 0.8   # Maximum percentage of order to fill (if partial fills enabled)
    min_partial_fill_pct: float = 0.5   # Minimum percentage of order to fill (if partial fills enabled)
    slippage_std: float = 0.001         # Standard deviation for slippage simulation


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


class RiskManager:
    """
    Unified Risk Manager that handles:
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
      - plus legacy methods (check_max_drawdown, calculate_stop_loss, etc.)
        to satisfy older tests

    Recent Changes:
    - Added support for enforcing stop losses directly
    - Added forced liquidation when drawdown exceeds threshold
    - Added trailing stop implementation
    - Added support for partial fills and slippage
    - Added time-based constraints (weekend close, max holding)
    """
    def __init__(self, config: RiskConfig):
        """
        Initialize the RiskManager with the given configuration.
        
        Args:
            config (RiskConfig): Risk management configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        self.liquidation_triggered = False
        self.liquidation_assets = set()
        self._correlation_matrix = None
        self._asset_returns = {}
        self._last_correlation_update = None

    # -----------------------------------------------------------------------
    #  Legacy or Additional methods (for older tests)
    # -----------------------------------------------------------------------
    def check_trade_limits(self, timestamp: pd.Timestamp) -> bool:
        """
        Return True if we have not exceeded the daily trade limit on this date.
        test_trade_limits() calls this.
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
        Legacy method for older tests that directly increment the trade counter.
        test_trade_limits calls this method by name.
        """
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """
        Legacy method: Check if max drawdown has been exceeded.
        
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
        if max_exceeded and self.config.enable_forced_liquidation:
            self.liquidation_triggered = True
            self.logger.warning(
                f"Max drawdown exceeded: {drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
            )
            
        return max_exceeded

    def check_leverage_limits(self, portfolio_value: float, position_value: float) -> bool:
        """
        Return True if position_value/portfolio_value <= max_leverage.
        test_leverage_limits() calls this.
        """
        if portfolio_value <= 0:
            return False
        leverage = position_value / portfolio_value
        return leverage <= self.config.max_leverage

    def calculate_stop_loss(
        self, entry_price: float, position_size: float, is_long: bool = True
    ) -> float:
        """
        Return the stop loss price based on config.stop_loss_pct.
        test_stop_loss() calls this.
        """
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def process_trade_signal(self, signal: dict) -> bool:
        """
        Simple old interface: if trade signal has {timestamp, type, price, size}
        and price>0, size>0 => return True. Otherwise False.
        test_trade_signal_processing() calls this.
        """
        required_fields = ["timestamp", "type", "price", "size"]
        if not all(field in signal for field in required_fields):
            self.logger.warning("Invalid trade signal: missing required fields.")
            return False
        if signal["price"] <= 0 or signal["size"] <= 0:
            self.logger.warning("Invalid trade signal: negative price or zero size.")
            return False
        return True

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
        asset_name: Optional[str] = None
    ) -> float:
        """
        Return a NOTIONAL position size in dollars (not units).
        test_position_sizing() expects "portfolio_value * max_position_size" as a baseline.
        Also scale down if volatility is high or if correlation is too high.
        """
        # Basic position size limit in nominal terms
        position_size = portfolio_value * self.config.max_position_size

        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            vol_scalar = 1.0 / (1.0 + volatility)
            position_size *= vol_scalar

        # If we have correlation data and an asset_name & positions
        if current_positions and asset_name and self._correlation_matrix is not None:
            # If correlation is too high, reduce the size. (Simplistic approach)
            for pos_asset, pos_value in current_positions.items():
                if not self.check_correlation_limits(asset_name, pos_asset):
                    # E.g. reduce by half
                    position_size *= 0.5

        # Enforce minimum size (in notional)
        min_notional = portfolio_value * self.config.min_trade_size
        if position_size < min_notional:
            return 0.0

        return position_size

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
        if self.config.enable_partial_fills:
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

    def update_after_trade(self, timestamp: pd.Timestamp, asset: str = None, 
                       trade_size: float = 0.0, price: float = 0.0,
                       portfolio_value: float = 0.0, positions: Dict[str, Dict[str, float]] = None,
                       is_long: bool = None, current_price: float = None, units: float = None):
        """
        Update internal state after a trade is executed.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            asset (str): Asset that was traded
            trade_size (float): Size of the trade (positive for buy, negative for sell)
            price (float): Execution price
            portfolio_value (float): Current portfolio value
            positions (dict): Current positions
            is_long (bool): Deprecated, use trade_size > 0 instead
            current_price (float): Deprecated, use price instead
            units (float): Deprecated, use trade_size instead
        """
        # Handle backward compatibility
        if current_price is not None and price == 0.0:
            price = current_price
            
        if units is not None and trade_size == 0.0:
            trade_size = units if is_long else -units
            
        # Create mock positions for backward compatibility
        if positions is None and asset is not None and units is not None:
            positions = {
                asset: {
                    "units": units,
                    "entry_price": price,
                    "cost_basis": price * units
                }
            }
            
        # Update trade counter
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1
        
        # Update peak value for drawdown calculation
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Store current value
        self.current_value = portfolio_value
        
        # Update position start times
        if asset and trade_size != 0:
            # If this is a new position or flipping direction
            if positions and (
                asset not in positions or 
                (positions[asset]["units"] > 0 and trade_size < 0 and abs(trade_size) >= positions[asset]["units"]) or
                (positions[asset]["units"] < 0 and trade_size > 0 and abs(trade_size) >= abs(positions[asset]["units"]))
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
                stop_price=self.calculate_stop_loss(price, units, is_long),
                trailing=self.config.enable_trailing_stop,
                trailing_pct=self.config.trailing_stop_pct
            )
            self.logger.info(
                f"Set {'trailing ' if self.config.enable_trailing_stop else ''}stop loss for {asset} "
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
                    stop_price = self.calculate_stop_loss(price, abs(position_units), is_long_position)
                    self.stop_losses[asset] = StopLossConfig(
                        entry_price=price,
                        position_size=position_units,
                        symbol=asset,
                        stop_price=stop_price,
                        trailing=self.config.enable_trailing_stop,
                        trailing_pct=self.config.trailing_stop_pct
                    )
                    self.logger.info(
                        f"Set {'trailing ' if self.config.enable_trailing_stop else ''}stop loss for {asset} "
                        f"at {stop_price:.2f} ({'long' if is_long_position else 'short'} position)"
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

    # -------------------------------------------------------------------
    # NEW METHODS FOR ENHANCED RISK MANAGEMENT
    # -------------------------------------------------------------------
    
    def check_stop_losses(self, current_prices: Dict[str, float], positions: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """
        Check if any stop losses have been triggered.
        
        Args:
            current_prices (dict): Dictionary of current prices by asset
            positions (dict): Dictionary of current positions
            
        Returns:
            dict: Dictionary mapping assets to boolean indicating if stop loss was triggered
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
            current_prices (dict): Dictionary of current prices by asset
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
    
    def check_forced_liquidation(self) -> bool:
        """
        Check if forced liquidation has been triggered due to max drawdown.
        
        Returns:
            bool: True if liquidation has been triggered, False otherwise
        """
        return self.liquidation_triggered
    
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
        if self.config.slippage_std <= 0:
            return price
            
        # Generate random slippage based on normal distribution
        slippage_factor = np.random.normal(0, self.config.slippage_std)
        
        # Apply slippage
        new_price = price * (1.0 + slippage_factor)
        
        # Ensure price remains positive
        new_price = max(new_price, 1e-8)
        
        return new_price
    
    def check_weekend_close(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if positions should be closed due to end of week.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            bool: True if weekend close is in effect, False otherwise
        """
        if not self.config.close_positions_on_friday:
            return False
            
        # Check if it's Friday after 16:00
        return timestamp.dayofweek == 4 and timestamp.hour >= 16
    
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
        if self.config.max_holding_period_days <= 0:
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

    # =====================
    # ADVANCED METHODS
    # =====================

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Value at Risk from historical returns at given confidence level.
        """
        if len(returns) < 2:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        """
        var = self.calculate_var(returns, confidence_level)
        if len(returns) < 2 or var <= 1e-12:
            return 0.0
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return 0.0
        return abs(tail.mean())

    def update_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Recompute correlation matrix from the provided price_data.
        """
        if not price_data:
            return
        
        # Build returns
        for asset, prices in price_data.items():
            self._asset_returns[asset] = prices.pct_change().dropna()

        # Combine into dataframe w/ same window
        returns_df = {}
        for asset, rets in self._asset_returns.items():
            windowed = rets.tail(self.config.correlation_window)
            if len(windowed) > 1:
                returns_df[asset] = windowed
        if not returns_df:
            return
        
        df = pd.DataFrame(returns_df)
        if df.shape[1] < 2:
            return  # only 1 asset => no correlation to compute

        self._correlation_matrix = df.corr()
        self._last_correlation_update = pd.Timestamp.now()

    def check_correlation_limits(self, asset1: str, asset2: str) -> bool:
        """
        True if correlation between asset1 & asset2 is <= max_correlation
        Test expects a bool type (not numpy.bool_).
        """
        if self._correlation_matrix is None:
            return True
        if (asset1 not in self._correlation_matrix.columns
            or asset2 not in self._correlation_matrix.columns):
            return True

        corr = abs(self._correlation_matrix.loc[asset1, asset2])
        return bool(corr <= self.config.max_correlation)

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
        if asset in positions and self.config.stop_loss_pct > 0:
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
                    if self.config.enable_forced_liquidation:
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

    def get_portfolio_var(self, 
                    portfolio_value: float, 
                    positions: Dict[str, Dict[str, float]],
                    prices: Dict[str, float]) -> float:
        """
        Calculate portfolio Value at Risk (VaR) based on weighted positions.
        
        Args:
            portfolio_value: Total portfolio value
            positions: Dictionary of positions
            prices: Current prices
            
        Returns:
            float: Portfolio VaR as a fraction of portfolio value
        """
        if not self._correlation_matrix is not None or not self._asset_returns:
            # Not enough data to calculate VaR
            return 0.0
            
        # Calculate position weights
        weights = {}
        total_position_value = 0.0
        
        for symbol, pos_dict in positions.items():
            if symbol not in prices:
                continue
                
            position_value = pos_dict["units"] * prices[symbol]
            total_position_value += position_value
            weights[symbol] = position_value
            
        if total_position_value < 1e-8:
            return 0.0
            
        # Normalize weights
        for symbol in weights:
            weights[symbol] /= total_position_value
            
        # Calculate portfolio variance
        portfolio_variance = 0.0
        
        # Single asset case
        if len(weights) == 1:
            symbol = list(weights.keys())[0]
            if symbol in self._asset_returns:
                returns = self._asset_returns[symbol]
                variance = returns.var()
                portfolio_variance = variance * (weights[symbol] ** 2)
        # Multi-asset case
        elif len(weights) > 1:
            for i, symbol_i in enumerate(weights.keys()):
                if symbol_i not in self._asset_returns:
                    continue
                    
                weight_i = weights[symbol_i]
                var_i = self._asset_returns[symbol_i].var()
                
                # Add variance term
                portfolio_variance += (weight_i ** 2) * var_i
                
                # Add covariance terms
                for j, symbol_j in enumerate(weights.keys()):
                    if i >= j or symbol_j not in self._asset_returns:
                        continue
                        
                    weight_j = weights[symbol_j]
                    if (symbol_i, symbol_j) in self._correlation_matrix:
                        corr_ij = self._correlation_matrix[(symbol_i, symbol_j)]
                        std_i = self._asset_returns[symbol_i].std()
                        std_j = self._asset_returns[symbol_j].std()
                        cov_ij = corr_ij * std_i * std_j
                        portfolio_variance += 2 * weight_i * weight_j * cov_ij
        
        # Convert to VaR using normal distribution quantile
        import scipy.stats as stats
        z_score = stats.norm.ppf(self.config.var_confidence_level)
        portfolio_var = z_score * np.sqrt(portfolio_variance)
        
        return portfolio_var 