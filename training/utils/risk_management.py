"""
Risk management system for trading bot.
Includes position sizing, stop-loss, and trade limits.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskConfig:
    """Configuration for risk management"""
    max_position_size: float = 0.2  # Maximum position size as fraction of portfolio
    stop_loss_pct: float = 0.02    # Stop loss percentage
    max_drawdown_pct: float = 0.15  # Maximum allowable drawdown
    daily_trade_limit: int = 10     # Maximum trades per day
    min_trade_size: float = 0.01    # Minimum trade size as fraction of portfolio
    max_leverage: float = 1.0       # Maximum leverage (1.0 = no leverage)
    position_scaling: bool = True    # Whether to scale position size based on volatility

class RiskManager:
    """Risk management system"""
    
    def __init__(self, config: RiskConfig = None):
        """Initialize risk manager with configuration"""
        self.config = config or RiskConfig()
        self.reset()
    
    def reset(self):
        """Reset risk manager state"""
        self.daily_trades = 0
        self.last_trade_day = None
        self.current_drawdown = 0.0
        self.peak_value = None
        self.stop_loss_prices = {}  # trade_id -> stop price
    
    def calculate_position_size(self, 
                              portfolio_value: float,
                              price: float,
                              volatility: float = None) -> float:
        """Calculate allowed position size considering risk limits
        
        Args:
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Optional volatility measure for scaling
            
        Returns:
            Maximum allowed position size in base currency
        """
        # Basic position size
        base_size = portfolio_value * self.config.max_position_size
        
        # Scale by volatility if enabled and volatility is provided
        if self.config.position_scaling and volatility is not None:
            # Reduce position size when volatility is high
            vol_scalar = 1.0 / (1.0 + volatility)
            base_size *= vol_scalar
        
        # Ensure minimum size
        min_size = portfolio_value * self.config.min_trade_size
        if base_size < min_size:
            return 0.0  # Don't trade if can't meet minimum
        
        return base_size
    
    def check_trade_limits(self, timestamp: pd.Timestamp) -> bool:
        """Check if trade is allowed based on daily limits
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Boolean indicating if trade is allowed
        """
        current_day = timestamp.date()
        
        # Reset counter on new day
        if self.last_trade_day != current_day:
            self.daily_trades = 0
            self.last_trade_day = current_day
        
        # Check if under daily limit
        return self.daily_trades < self.config.daily_trade_limit
    
    def update_trade_counter(self, timestamp: pd.Timestamp):
        """Update daily trade counter"""
        self.daily_trades += 1
        self.last_trade_day = timestamp.date()
    
    def set_stop_loss(self, 
                     trade_id: str,
                     entry_price: float,
                     position_type: str) -> float:
        """Set stop loss price for a trade
        
        Args:
            trade_id: Unique trade identifier
            entry_price: Entry price of the trade
            position_type: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        stop_pct = self.config.stop_loss_pct
        
        if position_type == 'long':
            stop_price = entry_price * (1 - stop_pct)
        else:  # short
            stop_price = entry_price * (1 + stop_pct)
            
        self.stop_loss_prices[trade_id] = stop_price
        return stop_price
    
    def check_stop_loss(self, 
                       trade_id: str,
                       current_price: float) -> bool:
        """Check if stop loss has been hit
        
        Args:
            trade_id: Trade identifier
            current_price: Current price
            
        Returns:
            Boolean indicating if stop loss was hit
        """
        if trade_id not in self.stop_loss_prices:
            return False
            
        stop_price = self.stop_loss_prices[trade_id]
        return current_price <= stop_price
    
    def update_drawdown(self, 
                       portfolio_value: float) -> Tuple[float, bool]:
        """Update and check maximum drawdown
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (current_drawdown, drawdown_limit_exceeded)
        """
        if self.peak_value is None:
            self.peak_value = portfolio_value
        else:
            self.peak_value = max(self.peak_value, portfolio_value)
        
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        return self.current_drawdown, self.current_drawdown > self.config.max_drawdown_pct
    
    def adjust_for_leverage(self, 
                          position_size: float,
                          current_leverage: float) -> float:
        """Adjust position size based on leverage limits
        
        Args:
            position_size: Proposed position size
            current_leverage: Current portfolio leverage
            
        Returns:
            Adjusted position size
        """
        if current_leverage >= self.config.max_leverage:
            return 0.0
        
        max_additional = (self.config.max_leverage - current_leverage) * position_size
        return min(position_size, max_additional)
    
    def process_trade_signal(self,
                           timestamp: pd.Timestamp,
                           portfolio_value: float,
                           price: float,
                           volatility: Optional[float] = None,
                           current_leverage: float = 0.0) -> Dict:
        """Process a trade signal through all risk checks
        
        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
            price: Current asset price
            volatility: Optional volatility measure
            current_leverage: Current portfolio leverage
            
        Returns:
            Dict with risk assessment and limits
        """
        # Check trade limits
        if not self.check_trade_limits(timestamp):
            return {'allowed': False, 'reason': 'daily_trade_limit_exceeded'}
        
        # Calculate position size
        position_size = self.calculate_position_size(
            portfolio_value, price, volatility
        )
        
        # Adjust for leverage
        position_size = self.adjust_for_leverage(position_size, current_leverage)
        
        # Check drawdown
        drawdown, exceeded = self.update_drawdown(portfolio_value)
        if exceeded:
            return {'allowed': False, 'reason': 'max_drawdown_exceeded'}
        
        return {
            'allowed': True,
            'position_size': position_size,
            'current_drawdown': drawdown
        }
    
    def update_after_trade(self,
                          trade_id: str,
                          timestamp: pd.Timestamp,
                          entry_price: float,
                          position_type: str):
        """Update risk manager state after trade execution
        
        Args:
            trade_id: Unique trade identifier
            timestamp: Trade timestamp
            entry_price: Entry price
            position_type: Type of position ('long' or 'short')
        """
        self.update_trade_counter(timestamp)
        self.set_stop_loss(trade_id, entry_price, position_type)