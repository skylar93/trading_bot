import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting system for trading strategies"""
    
    REQUIRED_COLUMNS = {'$open', '$high', '$low', '$close', '$volume'}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, trading_fee: float = 0.001):
        """Initialize backtester
        
        Args:
            data: DataFrame with OHLCV data (must have $ prefixed columns)
            initial_balance: Initial portfolio balance
            trading_fee: Trading fee as decimal
        """
        # Validate required columns
        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {list(missing_columns)}")
            
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.reset()
        
    def reset(self):
        """Reset backtester state"""
        self.portfolio_values = [self.initial_balance]  # Initialize with starting balance
        self.trades = []
        self.position = 0
        self.balance = self.initial_balance
        self.peak_value = self.initial_balance
        
    def run(self, 
            strategy: Union[Any, Any], 
            window_size: int = 20, 
            verbose: bool = False) -> Dict[str, Any]:
        """Run backtest with given strategy
        
        Args:
            strategy: Trading strategy object with get_action method
            window_size: Size of observation window
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        if len(self.data) < window_size:
            raise ValueError(f"Data length ({len(self.data)}) must be >= window_size ({window_size})")
        
        try:
            # Run strategy
            for i in range(window_size, len(self.data)):
                # Get current window of data
                window_data = self.data.iloc[i-window_size:i].copy()
                current_data = self.data.iloc[i].copy()
                timestamp = current_data.name
                
                # Get strategy action
                try:
                    action = strategy.get_action(window_data)
                    if not isinstance(action, (int, float, np.ndarray)):
                        self.logger.warning(f"Invalid action type: {type(action)}, expected float")
                        continue
                    action = float(action)  # Ensure action is float
                except Exception as e:
                    self.logger.error(f"Error getting action from strategy: {str(e)}")
                    continue
                
                # Execute trade
                price_data = {
                    '$open': current_data['$open'],
                    '$high': current_data['$high'],
                    '$low': current_data['$low'],
                    '$close': current_data['$close'],
                    '$volume': current_data['$volume']
                }
                trade_result = self.execute_trade(timestamp, action, price_data)
                
                # Update portfolio value even if trade was skipped
                if 'portfolio_value' not in trade_result:
                    portfolio_value = self._calculate_portfolio_value(price_data['$close'])
                    self.portfolio_values.append(portfolio_value)
                    self.peak_value = max(self.peak_value, portfolio_value)
                
                if verbose and i % 100 == 0:
                    self.logger.info(f"Progress: {i}/{len(self.data)}")
                    
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Ensure portfolio values match the data length
        expected_length = len(self.data) - window_size + 1
        if len(self.portfolio_values) < expected_length:
            last_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
            self.portfolio_values.extend([last_value] * (expected_length - len(self.portfolio_values)))
        elif len(self.portfolio_values) > expected_length:
            self.portfolio_values = self.portfolio_values[:expected_length]
        
        return {
            'metrics': metrics,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'timestamps': self.data.index[window_size-1:].tolist()
        }
    
    def execute_trade(self, timestamp: pd.Timestamp, action: float, price_data: Dict[str, float]) -> Dict[str, Any]:
        """Execute trade based on action
        
        Args:
            timestamp: Current timestamp
            action: Action from strategy (-1 to 1)
            price_data: Dictionary with current price data
            
        Returns:
            Dictionary with trade results
        """
        try:
            # Validate price data
            required_columns = {'$open', '$high', '$low', '$close', '$volume'}
            missing_cols = required_columns - set(price_data.keys())
            if missing_cols:
                self.position = 0
                return {
                    'timestamp': timestamp,
                    'position': self.position,
                    'balance': self.balance,
                    'action': 'error',
                    'reason': f"Missing required columns: {missing_cols}"
                }
            
            # Bound action between -1 and 1
            action = max(min(action, 1.0), -1.0)
            
            # Skip very small actions (increased threshold)
            if abs(action) < 1e-4:  # Increased threshold for small actions
                self.position = 0  # Reset position for very small actions
                return {
                    'timestamp': timestamp,
                    'position': self.position,
                    'balance': self.balance,
                    'action': 'skip',
                    'reason': 'action too small'
                }
            
            current_price = price_data['$close']
            
            # Calculate trade size
            if action > 0:  # Buy
                # Check if balance is too low for any meaningful trade
                if self.balance < 10:  # Minimum balance requirement
                    self.position = 0  # Reset position for insufficient balance
                    return {
                        'timestamp': timestamp,
                        'position': self.position,
                        'balance': self.balance,
                        'action': 'skip',
                        'reason': 'insufficient balance for minimum trade'
                    }
                
                max_shares = self.balance / (current_price * (1 + self.trading_fee))
                trade_shares = max_shares * abs(action)
                
                # Skip if trade size is too small
                if trade_shares < 1e-4:
                    self.position = 0  # Reset position for very small trades
                    return {
                        'timestamp': timestamp,
                        'position': self.position,
                        'balance': self.balance,
                        'action': 'skip',
                        'reason': 'trade size too small'
                    }
                
                cost = trade_shares * current_price * (1 + self.trading_fee)
                
                if cost > self.balance:  # Added balance check
                    self.logger.warning("Insufficient balance for trade")
                    self.position = 0  # Reset position for insufficient balance
                    return {
                        'timestamp': timestamp,
                        'position': self.position,
                        'balance': self.balance,
                        'action': 'skip',
                        'reason': 'insufficient balance'
                    }
                
                self.balance -= cost
                self.position += trade_shares
                
                # Clean up dust after buy
                if self.position < 1e-4:
                    self.position = 0
                
                trade = {
                    'timestamp': timestamp,
                    'entry_time': timestamp,
                    'type': 'buy',
                    'size': trade_shares,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.balance,
                    'position': self.position,
                    'action': 'buy',
                    'reason': 'trade executed'
                }
                
            else:  # Sell
                # Skip if no position to sell
                if self.position < 1e-4:
                    self.position = 0  # Clean up any dust
                    return {
                        'timestamp': timestamp,
                        'position': self.position,
                        'balance': self.balance,
                        'action': 'skip',
                        'reason': 'no position to sell'
                    }
                
                trade_shares = self.position * abs(action)
                
                # Skip if trade size is too small
                if trade_shares < 1e-4:
                    self.position = 0  # Reset position for very small trades
                    return {
                        'timestamp': timestamp,
                        'position': self.position,
                        'balance': self.balance,
                        'action': 'skip',
                        'reason': 'trade size too small'
                    }
                
                proceeds = trade_shares * current_price * (1 - self.trading_fee)
                
                self.balance += proceeds
                self.position -= trade_shares
                
                # Clean up any dust (very small remaining position)
                if self.position < 1e-4:
                    self.position = 0
                
                trade = {
                    'timestamp': timestamp,
                    'entry_time': timestamp,
                    'type': 'sell',
                    'size': trade_shares,
                    'price': current_price,
                    'revenue': proceeds,
                    'balance': self.balance,
                    'position': self.position,
                    'action': 'sell',
                    'reason': 'trade executed'
                }
            
            self.trades.append(trade)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            self.peak_value = max(self.peak_value, portfolio_value)
            
            trade['portfolio_value'] = portfolio_value
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            self.position = 0  # Reset position on error
            return {
                'timestamp': timestamp,
                'error': str(e),
                'position': self.position,
                'balance': self.balance,
                'action': 'error',
                'reason': str(e)
            }
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value
        
        Args:
            current_price: Current asset price
            
        Returns:
            Total portfolio value
        """
        return self.balance + (self.position * current_price)
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate trading metrics
        
        Returns:
            Dictionary with trading metrics
        """
        try:
            # Calculate returns
            values = np.array(self.portfolio_values)
            returns = np.diff(values) / values[:-1]
            
            # Total return
            total_return = (values[-1] / values[0]) - 1 if len(values) > 1 else 0
            
            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            sortino_ratio = (np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
                           if len(negative_returns) > 0 else 0)
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = min(max_drawdown, -drawdown)
            
            # Win rate
            profitable_trades = sum(1 for trade in self.trades 
                                 if ('revenue' in trade and trade['revenue'] > trade.get('cost', 0)) or
                                    ('cost' in trade and trade['cost'] < trade.get('revenue', float('inf'))))
            total_trades = len(self.trades)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'final_balance': self.balance,
                'final_portfolio_value': values[-1] if len(values) > 0 else self.balance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'total_trades': len(self.trades),
                'win_rate': 0,
                'final_balance': self.balance,
                'final_portfolio_value': self.balance
            }