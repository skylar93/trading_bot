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
    daily_trade_limit: int = 1000
    min_trade_size: float = 0.01
    max_leverage: float = 1.0
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

    def update_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> None:
        """Update correlation matrix for multi-asset portfolio
        
        Args:
            price_data: Dictionary of price series by asset
        """
        # Calculate returns if not already stored
        for asset, prices in price_data.items():
            if asset not in self._asset_returns:
                self._asset_returns[asset] = prices.pct_change().dropna()
            
        # Create correlation matrix
        returns_df = pd.DataFrame({
            asset: returns.tail(self.config.correlation_window)
            for asset, returns in self._asset_returns.items()
        })
        
        self._correlation_matrix = returns_df.corr()
        self._last_correlation_update = pd.Timestamp.now()

    def check_correlation_limits(self, asset1: str, asset2: str) -> bool:
        """Check if correlation between assets is within limits
        
        Args:
            asset1: First asset name
            asset2: Second asset name
            
        Returns:
            Whether correlation is within limits
        """
        if self._correlation_matrix is None:
            return bool(False)
            
        if asset1 not in self._correlation_matrix.index or asset2 not in self._correlation_matrix.columns:
            return bool(False)
            
        correlation = abs(self._correlation_matrix.loc[asset1, asset2])
        return bool(correlation <= self.config.max_correlation)

    def get_portfolio_var(self, positions: Dict[str, float], portfolio_value: float) -> float:
        """Calculate portfolio Value at Risk

        Args:
            positions: Dictionary of positions {asset: size}
            portfolio_value: Total portfolio value

        Returns:
            Portfolio VaR
        """
        if not positions or portfolio_value <= 0:
            return 0.0

        # Calculate position weights as fraction of portfolio value
        weights = {}
        for asset, size in positions.items():
            if asset in self._asset_returns:
                # Use latest price from returns data
                latest_price = (1 + self._asset_returns[asset].iloc[-1]) * (1 / (1 + self._asset_returns[asset].iloc[-2]))
                position_value = size * latest_price
                weights[asset] = position_value / portfolio_value
            else:
                weights[asset] = 0.0

        # Calculate individual VaRs
        asset_vars = []
        assets = []  # Keep track of assets in the same order
        for asset, weight in weights.items():
            if asset in self._asset_returns and weight != 0:
                var = self.calculate_var(
                    self._asset_returns[asset],
                    self.config.var_confidence_level
                )
                asset_vars.append(var)
                assets.append(asset)

        if not asset_vars:  # No valid assets with VaR
            return 0.0

        # Create variance-covariance matrix
        var_matrix = np.diag(asset_vars)
        weights_array = np.array([weights[asset] for asset in assets])
        
        # Calculate portfolio VaR
        if self._correlation_matrix is None or len(assets) == 1:
            # If no correlation matrix or single asset, use simple sum of weighted VaRs
            portfolio_var = np.sqrt(weights_array.dot(var_matrix).dot(weights_array.T))
        else:
            # Use full variance-covariance matrix with matching assets
            correlation_subset = self._correlation_matrix.loc[assets, assets].values
            portfolio_var = np.sqrt(
                weights_array.dot(var_matrix).dot(correlation_subset).dot(weights_array.T)
            )
            
        return min(portfolio_var, 1.0)  # Cap at 100% to avoid unrealistic values

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

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
        asset_name: Optional[str] = None
    ) -> float:
        """Calculate position size considering all risk factors"""
        # Basic position size limit based on config
        position_size = portfolio_value * self.config.max_position_size

        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            vol_scalar = 1.0 / (1.0 + volatility)
            position_size *= vol_scalar

        # Adjust for correlation if relevant
        if current_positions and asset_name and self._correlation_matrix is not None:
            # Reduce position size based on correlation with existing positions
            for pos_asset, pos_value in current_positions.items():
                if not self.check_correlation_limits(asset_name, pos_asset):
                    position_size *= 0.5  # Reduce size for highly correlated assets

        # Ensure minimum size
        if position_size < portfolio_value * self.config.min_trade_size:
            return 0.0

        return min(position_size, portfolio_value * self.config.max_position_size)

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
        """Process trade signal and check all risk limits

        Args:
            signal: Trade signal or timestamp
            portfolio_value: Optional portfolio value
            price: Optional current price
            volatility: Optional volatility measure
            current_leverage: Optional current leverage
            current_positions: Optional current positions
            timestamp: Optional timestamp

        Returns:
            Trade permission and risk assessment
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
