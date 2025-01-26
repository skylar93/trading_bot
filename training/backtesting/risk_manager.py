"""Risk management system for trading"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Any
import logging


@dataclass
class RiskConfig:
    """Configuration for risk management"""
    
    max_position_size: float = 0.2
    stop_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    daily_trade_limit: int = 10
    max_leverage: float = 1.0
    portfolio_var_limit: float = 0.02
    max_correlation: float = 0.7
    min_trade_size: float = 0.01
    volatility_lookback: int = 20
    correlation_window: int = 30
    var_confidence_level: float = 0.95
    risk_free_rate: float = 0.02


class RiskManager:
    """Risk management system"""

    def __init__(self, config: Optional[RiskConfig] = None):
        """Initialize risk manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    def reset(self):
        """Reset risk manager state"""
        self.trade_counter = {}  # Daily trade counts
        self.peak_value = None
        self.current_drawdown = 0.0
        self._asset_returns = {}
        self._correlation_matrix = None

    def check_trade(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        trade_size: float,
        price: float,
        positions: Dict[str, float],
        asset: str,
        volatility: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check if trade is allowed and calculate adjusted size
        
        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
            trade_size: Proposed trade size
            price: Current price
            positions: Current positions dictionary
            asset: Asset being traded
            volatility: Optional volatility estimate
            
        Returns:
            Dict containing:
            - allowed: Whether trade is allowed
            - adjusted_size: Adjusted trade size
            - reason: Reason for adjustment/rejection
        """
        # Check daily trade limit
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
            
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return {
                "allowed": False,
                "adjusted_size": 0.0,
                "reason": "Daily trade limit reached"
            }

        # Check drawdown
        if self.peak_value is not None:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(
                    f"Max drawdown exceeded: {current_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
                )
                # Reject all trades when drawdown limit is exceeded
                return {
                    "allowed": False,
                    "adjusted_size": 0.0,
                    "reason": f"Max drawdown exceeded: {current_drawdown:.2%}"
                }

        # Calculate position value and check leverage
        position_value = abs(trade_size * price)
        current_exposure = sum(abs(pos * price) for pos in positions.values())
        total_exposure = current_exposure + position_value
        
        if total_exposure / portfolio_value > self.config.max_leverage:
            adjusted_size = max(0.0, (self.config.max_leverage * portfolio_value - current_exposure) / price)
            if adjusted_size < self.config.min_trade_size * portfolio_value / price:
                return {
                    "allowed": False,
                    "adjusted_size": 0.0,
                    "reason": "Leverage limit reached"
                }
            return {
                "allowed": True,
                "adjusted_size": adjusted_size,
                "reason": "Adjusted for leverage limits"
            }

        # Check position size limits
        max_position_value = portfolio_value * self.config.max_position_size
        if position_value > max_position_value:
            adjusted_size = max_position_value / price
            if adjusted_size < self.config.min_trade_size * portfolio_value / price:
                return {
                    "allowed": False,
                    "adjusted_size": 0.0,
                    "reason": "Position size too large"
                }
            return {
                "allowed": True,
                "adjusted_size": adjusted_size,
                "reason": "Adjusted for position size limits"
            }

        # Check minimum trade size
        min_trade_value = portfolio_value * self.config.min_trade_size
        if position_value < min_trade_value:
            return {
                "allowed": False,
                "adjusted_size": 0.0,
                "reason": "Trade size below minimum"
            }

        # All checks passed
        return {
            "allowed": True,
            "adjusted_size": trade_size,
            "reason": "OK"
        }

    def update_after_trade(self, timestamp: pd.Timestamp) -> None:
        """Update state after trade execution"""
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1
        self.logger.debug(f"Trade counter for {date_key}: {self.trade_counter[date_key]}")

    def calculate_stop_loss(
        self, entry_price: float, is_long: bool = True
    ) -> float:
        """Calculate stop loss price level
        
        Args:
            entry_price: Entry price
            is_long: Whether position is long
            
        Returns:
            Stop loss price level
        """
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        return entry_price * (1 + self.config.stop_loss_pct)

    def update_drawdown(self, portfolio_value: float) -> float:
        """Update and return current drawdown
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Current drawdown as percentage
        """
        if self.peak_value is None:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        elif portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if self.current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(
                    f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
                )
            
        return self.current_drawdown 