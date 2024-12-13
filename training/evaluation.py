import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingMetrics:
    """Calculate trading strategy performance metrics"""
    
    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """Calculate returns from portfolio values"""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sharpe Ratio
        """
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
        
        std = np.std(excess_returns, ddof=1)
        if std == 0:
            return 0.0
            
        return np.mean(excess_returns) / std * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino Ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sortino Ratio
        """
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
            
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
            
        downside_std = np.std(downside_returns, ddof=1)
        if downside_std == 0:
            return 0.0
            
        return np.mean(excess_returns) / downside_std * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_maximum_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown and its duration
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0, 0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = portfolio_values / running_max - 1
        
        # Find the maximum drawdown
        max_drawdown = np.min(drawdowns)
        end_idx = np.argmin(drawdowns)
        
        # Find the start of the drawdown period
        start_idx = np.argmax(portfolio_values[:end_idx])
        
        return max_drawdown, start_idx, end_idx
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """
        Calculate Win Rate
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Win Rate as a percentage
        """
        if not trades:
            return 0.0
            
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades) * 100
    
    @staticmethod
    def calculate_profit_loss_ratio(trades: List[Dict]) -> float:
        """
        Calculate Profit/Loss Ratio
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Profit/Loss Ratio
        """
        if not trades:
            return 0.0
            
        profits = [trade['pnl'] for trade in trades if trade['pnl'] > 0]
        losses = [abs(trade['pnl']) for trade in trades if trade['pnl'] < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            return np.inf if avg_profit > 0 else 0.0
            
        return avg_profit / avg_loss
    
    @staticmethod
    def calculate_trade_statistics(trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive trade statistics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_trade_duration': 0.0
            }
            
        # Calculate basic statistics
        total_trades = len(trades)
        win_rate = TradingMetrics.calculate_win_rate(trades)
        profit_loss_ratio = TradingMetrics.calculate_profit_loss_ratio(trades)
        
        # Calculate average profit per trade
        total_pnl = sum(trade['pnl'] for trade in trades)
        avg_profit_per_trade = total_pnl / total_trades
        
        # Calculate average trade duration
        durations = [(trade['exit_time'] - trade['entry_time']).total_seconds() / 3600 
                    for trade in trades]  # Duration in hours
        avg_trade_duration = np.mean(durations)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_trade_duration': avg_trade_duration,
            'total_pnl': total_pnl
        }
    
    @staticmethod
    def evaluate_strategy(portfolio_values: np.ndarray, trades: List[Dict], 
                         risk_free_rate: float = 0.0) -> Dict:
        """
        Evaluate trading strategy performance
        
        Args:
            portfolio_values: Array of portfolio values
            trades: List of trade dictionaries
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        returns = TradingMetrics.calculate_returns(portfolio_values)
        max_drawdown, dd_start, dd_end = TradingMetrics.calculate_maximum_drawdown(portfolio_values)
        trade_stats = TradingMetrics.calculate_trade_statistics(trades)
        
        metrics = {
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            'sharpe_ratio': TradingMetrics.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': TradingMetrics.calculate_sortino_ratio(returns, risk_free_rate),
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_duration': dd_end - dd_start,
            **trade_stats
        }
        
        return metrics