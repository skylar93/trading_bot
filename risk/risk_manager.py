"""Risk management system for trading"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from scipy import stats


@dataclass
class RiskConfig:
    """Configuration for risk management"""

    max_position_size: float
    stop_loss_pct: float
    max_drawdown_pct: float
    daily_trade_limit: int
    min_trade_size: float
    max_leverage: float
    volatility_lookback: int = 20
    risk_free_rate: float = 0.02
    # Portfolio risk parameters
    var_confidence_level: float = 0.95
    correlation_window: int = 30
    max_correlation: float = 0.7  # Maximum allowed correlation between assets
    portfolio_var_limit: float = 0.02  # Maximum portfolio VaR as fraction of portfolio value


class RiskManager:
    """Risk management system"""

    def __init__(self, config: RiskConfig):
        """Initialize risk manager

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.trade_counter = {}  # Dictionary to track daily trades
        self.logger = logging.getLogger(self.__class__.__name__)
        self._asset_returns = {}  # Dictionary to store asset returns
        self._correlation_matrix = None
        self._last_correlation_update = None
        self._trades = {}

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk
        
        Args:
            returns: Historical returns series
            confidence_level: VaR confidence level (e.g. 0.95)
            
        Returns:
            VaR as a fraction of portfolio value
        """
        if len(returns) < 2:
            return 0.0
            
        return abs(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (CVaR)

        Args:
            returns: Historical returns series
            confidence_level: CVaR confidence level

        Returns:
            CVaR value as a positive number (expected loss beyond VaR)
        """
        var = self.calculate_var(returns, confidence_level)
        return abs(returns[returns <= -var].mean())

    def update_correlation_matrix(self, asset_prices: Dict[str, pd.Series]) -> None:
        """Update correlation matrix for portfolio assets
        
        Args:
            asset_prices: Dictionary of asset price series
        """
        # Skip if no prices or single asset
        if not asset_prices or len(asset_prices) < 2:
            self._correlation_matrix = None
            return
            
        # Calculate returns
        returns = {}
        for asset, prices in asset_prices.items():
            if len(prices) > 1:  # Need at least 2 points for returns
                returns[asset] = prices.pct_change().fillna(0)
        
        if not returns:
            self._correlation_matrix = None
            return
            
        # Calculate correlation matrix
        return_df = pd.DataFrame(returns)
        self._correlation_matrix = return_df.corr()
        self._last_correlation_update = pd.Timestamp.now()

    def get_portfolio_var(self, positions: Dict[str, float], portfolio_value: float) -> float:
        """Calculate portfolio VaR
        
        Args:
            positions: Dictionary of asset positions (amount in base currency)
            portfolio_value: Total portfolio value
        
        Returns:
            Portfolio VaR as fraction of portfolio value
        """
        # Skip if no correlation matrix or positions
        if self._correlation_matrix is None or not positions:
            return 0.0
        
        # Get position weights
        weights = {}
        for asset, pos in positions.items():
            if portfolio_value > 0:
                weights[asset] = pos / portfolio_value
            else:
                weights[asset] = 0
        
        # Calculate portfolio variance using correlation matrix
        portfolio_var = 0.0
        assets = list(weights.keys())
        
        if not assets:
            return 0.0
        
        # Use simple historical VaR if correlation data is missing
        if self._correlation_matrix is None:
            return 0.02  # Default to 2% VaR
        
        try:
            # Calculate portfolio variance
            for i, asset_i in enumerate(assets):
                for j, asset_j in enumerate(assets):
                    if asset_i in self._correlation_matrix.index and asset_j in self._correlation_matrix.columns:
                        portfolio_var += (
                            weights[asset_i]
                            * weights[asset_j]
                            * self._correlation_matrix.loc[asset_i, asset_j]
                        )
        except Exception as e:
            self.logger.warning(f"Error calculating portfolio VaR: {str(e)}")
            return 0.02  # Default to 2% VaR
        
        return min(max(np.sqrt(portfolio_var), 0), 1)  # Bound between 0 and 1

    def check_correlation_limits(self, asset1: str, asset2: str) -> bool:
        """Check if correlation between assets exceeds limits

        Args:
            asset1: First asset name
            asset2: Second asset name

        Returns:
            Whether correlation is within limits
        """
        if self._correlation_matrix is None:
            return True
            
        if asset1 not in self._correlation_matrix.index or asset2 not in self._correlation_matrix.columns:
            return True
            
        correlation = abs(self._correlation_matrix.loc[asset1, asset2])
        return bool(correlation <= self.config.max_correlation)

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float] = None,
        asset_name: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate position size based on risk parameters

        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Optional volatility estimate
            asset_name: Optional asset name for correlation checks
            current_positions: Optional dictionary of current positions

        Returns:
            Position size in base currency
        """
        # Base position size
        max_size = portfolio_value * self.config.max_position_size

        # Apply volatility scaling if provided
        if volatility is not None:
            # Scale down position size as volatility increases
            vol_scale = 1.0 / (1.0 + volatility)
            max_size *= vol_scale

        # Check correlation limits if we have portfolio information
        if asset_name and current_positions:
            for existing_asset in current_positions:
                if not self.check_correlation_limits(asset_name, existing_asset):
                    self.logger.warning(f"Correlation limit exceeded between {asset_name} and {existing_asset}")
                    max_size *= 0.5  # Reduce position size for highly correlated assets

        # Check portfolio VaR if we have positions
        if current_positions:
            test_positions = current_positions.copy()
            test_positions[asset_name] = max_size
            portfolio_var = self.get_portfolio_var(test_positions, portfolio_value)
            
            if portfolio_var > self.config.portfolio_var_limit:
                var_scale = self.config.portfolio_var_limit / portfolio_var
                max_size *= var_scale

        # Ensure minimum trade size
        if max_size < portfolio_value * self.config.min_trade_size:
            return 0.0

        return max_size

    def check_trade_limits(self, timestamp: pd.Timestamp) -> bool:
        """Check if trade is allowed based on daily limits"""
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0

        # Return False if limit is reached
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return False

        return True

    def update_trade_counter(self, timestamp: pd.Timestamp) -> None:
        """Update the trade counter for the given date"""
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1

    def calculate_stop_loss(
        self, entry_price: float, position_size: float, is_long: bool = True
    ) -> float:
        """Calculate stop loss price

        Args:
            entry_price: Entry price
            position_size: Position size
            is_long: Whether position is long

        Returns:
            Stop loss price
        """
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def check_max_drawdown(
        self, peak_value: float, current_value: float
    ) -> bool:
        """Check if max drawdown is exceeded

        Args:
            peak_value: Peak portfolio value
            current_value: Current portfolio value

        Returns:
            Whether max drawdown is exceeded
        """
        if peak_value <= 0:
            return False

        drawdown = (peak_value - current_value) / peak_value
        return drawdown > self.config.max_drawdown_pct

    def check_leverage_limits(
        self, portfolio_value: float, position_value: float
    ) -> bool:
        """Check if position is within leverage limits

        Args:
            portfolio_value: Current portfolio value
            position_value: Total position value

        Returns:
            Whether position is within limits
        """
        if portfolio_value <= 0:
            return False

        leverage = position_value / portfolio_value
        return leverage <= self.config.max_leverage

    def process_trade_signal(
        self,
        signal: Union[Dict[str, Any], pd.Timestamp],
        portfolio_value: Optional[float] = None,
        price: Optional[float] = None,
        volatility: Optional[float] = None,
        current_leverage: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Union[bool, Dict[str, Any]]:
        """Process trade signal with risk management
        
        This method supports both old and new interfaces:
        Old: process_trade_signal(signal_dict) -> bool
        New: process_trade_signal(timestamp, portfolio_value, price, volatility, current_leverage) -> Dict
        
        Args:
            signal: Either a trade signal dictionary or timestamp
            portfolio_value: Current portfolio value (new interface)
            price: Current price (new interface)
            volatility: Current volatility (new interface)
            current_leverage: Current leverage (new interface)
            current_positions: Current positions by asset (new interface)
            timestamp: Optional timestamp (for backward compatibility)
        
        Returns:
            Either boolean (old interface) or risk assessment dictionary (new interface)
        """
        # Detect which interface is being used
        if isinstance(signal, dict):
            # Old interface
            # Check for required fields
            required_fields = ["timestamp", "type", "price", "size"]
            if not all(field in signal for field in required_fields):
                self.logger.warning(
                    "Invalid trade signal: missing required fields"
                )
                return False

            # Validate signal values
            if signal["price"] <= 0 or signal["size"] <= 0:
                self.logger.warning(
                    "Invalid trade signal: price or size must be positive"
                )
                return False

            # Check trade limits
            if not self.check_trade_limits(pd.Timestamp(signal["timestamp"])):
                return False

            return True
        
        else:
            # New interface
            # Use timestamp from either signal or timestamp parameter
            ts = timestamp if timestamp is not None else signal
            if not all(v is not None for v in [portfolio_value, price]):
                raise ValueError("Missing required parameters for new interface")
            
            # Check basic risk limits
            if not self.check_trade_limits(ts):
                return {"allowed": False, "reason": "daily trade limit reached"}
            
            if current_leverage and current_leverage >= self.config.max_leverage:
                return {"allowed": False, "reason": "leverage limit reached"}
            
            # Calculate position size
            position_size = self.calculate_position_size(
                portfolio_value=portfolio_value,
                price=price,
                volatility=volatility,
                current_positions=current_positions
            )
            
            # Get current drawdown
            drawdown = 0.0  # Initialize drawdown
            if portfolio_value and hasattr(self, '_peak_value'):
                drawdown = (self._peak_value - portfolio_value) / self._peak_value if self._peak_value > 0 else 0.0
            
            if drawdown > self.config.max_drawdown_pct:
                return {
                    "allowed": False,
                    "reason": "max drawdown exceeded",
                    "current_drawdown": drawdown
                }
            
            return {
                "allowed": True,
                "position_size": position_size,
                "current_drawdown": drawdown
            }

    def update_after_trade(self, trade_id: str, timestamp: pd.Timestamp, entry_price: float, position_type: str) -> None:
        """Update risk metrics after trade execution
        
        Args:
            trade_id: Unique trade identifier
            timestamp: Trade timestamp
            entry_price: Entry price
            position_type: Position type (long/short)
        """
        # Update trade counter
        self.update_trade_counter(timestamp)
        
        # Store trade info for future reference
        if not hasattr(self, '_trades'):
            self._trades = {}
        
        self._trades[trade_id] = {
            'timestamp': timestamp,
            'entry_price': entry_price,
            'position_type': position_type
        }

    def reset(self) -> None:
        """Reset risk manager state"""
        self.trade_counter = {}
        self._asset_returns = {}
        self._correlation_matrix = None
        self._last_correlation_update = None
        self._trades = {}
