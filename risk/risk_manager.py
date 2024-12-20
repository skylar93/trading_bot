"""Risk management system for trading"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

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
    
    def calculate_position_size(self,
                              portfolio_value: float,
                              price: float,
                              volatility: Optional[float] = None) -> float:
        """Calculate position size based on risk parameters
        
        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Optional volatility estimate
            
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
    
    def calculate_stop_loss(self,
                          entry_price: float,
                          position_size: float,
                          is_long: bool = True) -> float:
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
    
    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
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
    
    def check_leverage_limits(self, portfolio_value: float, position_value: float) -> bool:
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
    
    def process_trade_signal(self, signal: dict) -> bool:
        """Process and validate trade signal"""
        # Check for required fields
        required_fields = ['timestamp', 'type', 'price', 'size']
        if not all(field in signal for field in required_fields):
            self.logger.warning("Invalid trade signal: missing required fields")
            return False
            
        # Validate signal values
        if signal['price'] <= 0 or signal['size'] <= 0:
            self.logger.warning("Invalid trade signal: price or size must be positive")
            return False
            
        # Check trade limits
        if not self.check_trade_limits(pd.Timestamp(signal['timestamp'])):
            return False
            
        return True