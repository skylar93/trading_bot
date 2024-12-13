"""
Backtesting Engine for Trading Bot
Handles portfolio management, trade execution, and performance tracking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as percentage
            max_position: Maximum allowed position size as percentage of portfolio
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reset()
        
    def reset(self):
        """Reset backtesting state"""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions: Dict[str, float] = {}  # symbol: amount
        self.trades: List[Dict] = []
        self.portfolio_history: List[float] = [self.initial_capital]
        self.cash_history: List[float] = [self.initial_capital]
        self.current_timestamp = None
        
    def update(self, 
              timestamp: pd.Timestamp,
              prices: Dict[str, float],
              actions: Dict[str, float]) -> Dict:
        """
        Update portfolio state based on actions
        
        Args:
            timestamp: Current timestamp
            prices: Dictionary of asset prices
            actions: Dictionary of trading actions (-1 to 1)
            
        Returns:
            Dict of current state metrics
        """
        self.current_timestamp = timestamp
        
        # Execute trades
        for symbol, action in actions.items():
            if symbol not in prices:
                continue
                
            # Calculate target position
            current_price = prices[symbol]
            portfolio_value = self.get_portfolio_value(prices)
            
            # For action == 0, fully close the position
            if abs(action) < 1e-6:
                target_position = 0
            else:
                max_position_value = portfolio_value * self.max_position
                target_position_value = action * max_position_value
                target_position = target_position_value / current_price
            
            # Current position
            current_position = self.positions.get(symbol, 0.0)
            
            # Calculate trade amount
            trade_amount = target_position - current_position
            trade_value = abs(trade_amount * current_price)
            trade_cost = trade_value * self.transaction_cost
            
            if abs(trade_amount) > 1e-6:  # Minimum trade threshold
                # Check if we have enough cash for buying
                if trade_amount > 0 and trade_value + trade_cost > self.cash:
                    trade_amount = (self.cash / (current_price * (1 + self.transaction_cost)))
                    trade_value = abs(trade_amount * current_price)
                    trade_cost = trade_value * self.transaction_cost
                
                if abs(trade_amount) > 1e-6:
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'price': current_price,
                        'amount': trade_amount,
                        'value': trade_value,
                        # Check if we're increasing or decreasing position
                        'type': 'buy' if (
                            (trade_amount > 0 and current_position >= 0) or
                            (trade_amount < 0 and current_position < 0)
                        ) else 'sell',
                        'cost': trade_cost
                    }
                    self.trades.append(trade)
                    
                    # Update position and cash
                    self.positions[symbol] = current_position + trade_amount
                    self.cash -= (trade_value + trade_cost) if trade_amount > 0 else -(trade_value - trade_cost)
                    
                    # Remove position if close to zero
                    if abs(self.positions[symbol]) < 1e-6:
                        del self.positions[symbol]
        
        # Update history
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_history.append(portfolio_value)
        self.cash_history.append(self.cash)
        
        return {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'trades': self.trades[-1] if self.trades else None
        }
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash"""
        value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                value += position * prices[symbol]
                
        return value
    
    def get_returns(self) -> pd.Series:
        """Calculate returns series"""
        returns = pd.Series(self.portfolio_history)
        returns = returns.pct_change()
        returns.index = pd.date_range(
            start=self.current_timestamp - pd.Timedelta(days=len(returns)-1),
            end=self.current_timestamp,
            periods=len(returns)
        )
        return returns
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_position_history(self) -> pd.DataFrame:
        """Get position value history for each asset"""
        position_values = []
        timestamps = pd.date_range(
            start=self.current_timestamp - pd.Timedelta(days=len(self.portfolio_history)-1),
            end=self.current_timestamp,
            periods=len(self.portfolio_history)
        )
        
        for timestamp, portfolio_value in zip(timestamps, self.portfolio_history):
            values = {'timestamp': timestamp, 'total': portfolio_value}
            values.update(self.positions)
            position_values.append(values)
            
        return pd.DataFrame(position_values).set_index('timestamp')
