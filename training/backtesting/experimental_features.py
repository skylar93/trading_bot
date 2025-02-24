"""
Experimental Features for Backtesting
===================================

This module provides experimental features for backtesting through a mixin class.
Features include weighted entry price calculation and improved PnL calculation.
"""

from typing import Dict, Any
import pandas as pd
from collections import defaultdict

class ExperimentalMixin:
    """
    Mixin class that provides experimental features for backtesting.
    
    Features
    --------
    - Weighted entry price calculation
    - Improved PnL calculation with partial position exits
    - Position tracking with entry timestamps
    
    Implementation Notes
    ------------------
    This mixin overrides execute_trade() to maintain entry prices and
    calculate PnL more accurately when closing positions partially.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize experimental features"""
        super().__init__(*args, **kwargs)
        self._entry_prices: Dict[str, float] = {}
        self._position_entries: Dict[str, Dict[pd.Timestamp, float]] = defaultdict(dict)
        
    def _update_entry_price(self, asset: str, price: float, amount: float):
        """
        Update weighted entry price for an asset
        
        Parameters
        ----------
        asset : str
            Asset identifier
        price : float
            Entry price for this trade
        amount : float
            Position size being added (should be positive)
        """
        current_position = abs(self.positions.get(asset, 0.0))
        if current_position == 0:
            self._entry_prices[asset] = price
            self._position_entries[asset].clear()
        else:
            # Calculate weighted average entry price
            total_position = current_position + amount
            old_entry = self._entry_prices.get(asset, price)
            self._entry_prices[asset] = (
                (old_entry * current_position + price * amount) / total_position
            )
            
        # Record entry with timestamp
        self._position_entries[asset][self.current_time] = amount
        
    def _calculate_pnl(self, asset: str, exit_price: float, exit_amount: float) -> float:
        """
        Calculate PnL for partial position exit using FIFO method
        
        Parameters
        ----------
        asset : str
            Asset identifier
        exit_price : float
            Price at which position is being exited
        exit_amount : float
            Amount being closed (should be positive)
            
        Returns
        -------
        float
            Realized PnL for this exit
        """
        remaining_exit = abs(exit_amount)
        total_pnl = 0.0
        entries_to_remove = []
        
        # Process entries in FIFO order
        for timestamp, entry_amount in sorted(self._position_entries[asset].items()):
            if remaining_exit <= 0:
                break
                
            amount_to_close = min(entry_amount, remaining_exit)
            entry_price = self._entry_prices[asset]
            
            # Calculate PnL for this portion
            if self.positions[asset] > 0:  # Long position
                trade_pnl = (exit_price - entry_price) * amount_to_close
            else:  # Short position
                trade_pnl = (entry_price - exit_price) * amount_to_close
                
            total_pnl += trade_pnl
            remaining_exit -= amount_to_close
            
            # Update or remove entry
            if amount_to_close == entry_amount:
                entries_to_remove.append(timestamp)
            else:
                self._position_entries[asset][timestamp] -= amount_to_close
                
        # Clean up fully closed entries
        for timestamp in entries_to_remove:
            del self._position_entries[asset][timestamp]
            
        return total_pnl
        
    def execute_trade(self, timestamp: pd.Timestamp, action: float,
                     price_data: Dict[str, float], asset: str = 'default',
                     **kwargs) -> Dict[str, Any]:
        """
        Override execute_trade to include experimental features
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Current timestamp
        action : float
            Trading action (-1 to 1, where positive is buy)
        price_data : Dict[str, float]
            Price data for the timestamp
        asset : str, optional
            Asset identifier, defaults to 'default'
            
        Returns
        -------
        Dict[str, Any]
            Trade execution results
        """
        # Store current time for entry tracking
        self.current_time = timestamp
        
        # Execute trade using parent implementation
        result = super().execute_trade(timestamp, action, price_data, asset, **kwargs)
        
        if result['success']:
            trade_price = result['price']
            trade_amount = abs(result['amount'])
            
            if action > 0:  # Buying
                self._update_entry_price(asset, trade_price, trade_amount)
            else:  # Selling
                pnl = self._calculate_pnl(asset, trade_price, trade_amount)
                result['realized_pnl'] = pnl
                
        return result 