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
        self.trade_counter = {}  # {date: count}
        self.logger = logging.getLogger(__name__)
    
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
        """Check if trade is allowed under daily limits
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Whether trade is allowed
        """
        date = timestamp.date()
        
        # Clean up old dates
        today = pd.Timestamp.now().date()
        old_dates = [d for d in self.trade_counter.keys() if d < today]
        for d in old_dates:
            del self.trade_counter[d]
            
        # Get current count
        count = self.trade_counter.get(date, 0)
        
        # Check if limit would be exceeded
        if count > self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit ({self.config.daily_trade_limit}) reached for {date}")
            return False
            
        return True
    
    def update_trade_counter(self, timestamp: pd.Timestamp) -> None:
        """Update daily trade counter
        
        Args:
            timestamp: Trade timestamp
        """
        date = timestamp.date()
        
        # Clean up old dates
        today = pd.Timestamp.now().date()
        old_dates = [d for d in self.trade_counter.keys() if d < today]
        for d in old_dates:
            del self.trade_counter[d]
        
        # Update counter
        count = self.trade_counter.get(date, 0)
        self.trade_counter[date] = count + 1
        self.logger.info(f"Trade counter for {date} updated to {self.trade_counter[date]}")
    
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
    
    def process_trade_signal(self, signal: Dict[str, Any], portfolio_value: float) -> Dict[str, Any]:
        """Process trade signal with risk checks
        
        Args:
            signal: Trade signal dictionary
            portfolio_value: Current portfolio value
            
        Returns:
            Processed signal with risk parameters
        """
        # Validate signal
        required_fields = ['timestamp', 'price', 'direction', 'type']
        if not all(field in signal for field in required_fields):
            return {
                'valid': False,
                'size': 0.0,
                'reason': 'invalid_signal_format'
            }
            
        if signal['direction'] not in ['long', 'short']:
            return {
                'valid': False,
                'size': 0.0,
                'reason': 'invalid_direction'
            }
            
        # Get current count
        date = signal['timestamp'].date()
        count = self.trade_counter.get(date, 0)
        
        # Check if limit would be exceeded
        if count >= self.config.daily_trade_limit:
            return {
                'valid': False,
                'size': 0.0,
                'reason': 'trade_limit_exceeded'
            }
        
        # Calculate position size
        size = self.calculate_position_size(
            portfolio_value=portfolio_value,
            price=signal['price'],
            volatility=signal.get('volatility')
        )
        
        if size == 0:
            return {
                'valid': False,
                'size': 0.0,
                'reason': 'size_too_small'
            }
        
        # Calculate stop loss
        is_long = signal['direction'] == 'long'
        stop_loss = self.calculate_stop_loss(
            entry_price=signal['price'],
            position_size=size,
            is_long=is_long
        )
        
        # All checks passed, update trade counter
        self.update_trade_counter(signal['timestamp'])
        
        return {
            'valid': True,
            'size': size,
            'stop_loss': stop_loss,
            'direction': signal['direction']
        }