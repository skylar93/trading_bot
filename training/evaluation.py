import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingMetrics:
    """Calculate trading strategy performance metrics"""
    
    @staticmethod
    def calculate_returns(portfolio_values: np.ndarray) -> np.ndarray:
        """Calculate returns from portfolio values"""
        portfolio_values = np.array(portfolio_values).reshape(-1)  # Ensure 1D array
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

def calculate_metrics(portfolio_values: List[float], returns: List[float]) -> Dict[str, float]:
    """Calculate trading metrics
    
    Args:
        portfolio_values: List of portfolio values over time
        returns: List of returns over time
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    
    # Replace inf/-inf with nan
    portfolio_values = np.nan_to_num(portfolio_values, nan=0.0, posinf=0.0, neginf=0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1 if len(portfolio_values) > 0 else 0.0
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    # Sharpe ratio (annualized)
    if len(returns) > 1:
        returns_mean = np.mean(returns)
        returns_std = np.std(returns)
        sharpe_ratio = np.sqrt(252) * returns_mean / (returns_std + 1e-6)
    else:
        sharpe_ratio = 0.0
    
    # Clip values to prevent NaN/Inf
    max_drawdown = np.clip(max_drawdown, -1.0, 0.0)
    sharpe_ratio = np.clip(sharpe_ratio, -10.0, 10.0)
    total_return = np.clip(total_return, -1.0, 10.0)
    
    return {
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe_ratio)
    }

def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate trade-specific metrics
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of metrics
    """
    if not trades:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'num_trades': 0
        }
    
    # Calculate trade metrics
    profits = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
    
    win_rate = len(profits) / len(trades) if trades else 0.0
    profit_factor = sum(profits) / (sum(losses) + 1e-6) if losses else float('inf')
    avg_trade = np.mean([t['pnl'] for t in trades])
    
    # Clip values
    profit_factor = np.clip(profit_factor, 0.0, 100.0)
    avg_trade = np.clip(avg_trade, -1000.0, 1000.0)
    
    return {
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'avg_trade': float(avg_trade),
        'num_trades': len(trades)
    }

def calculate_risk_metrics(returns: List[float]) -> Dict[str, float]:
    """Calculate risk metrics
    
    Args:
        returns: List of returns
        
    Returns:
        Dictionary of risk metrics
    """
    returns = np.array(returns)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(returns) < 2:
        return {
            'volatility': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'sortino_ratio': 0.0
        }
    
    # Annualized volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5)
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = np.mean(returns[returns <= var_95])
    
    # Sortino ratio (using 0 as minimum acceptable return)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        sortino_ratio = np.mean(returns) / (downside_std + 1e-6) * np.sqrt(252)
    else:
        sortino_ratio = float('inf')
    
    # Clip values
    volatility = np.clip(volatility, 0.0, 10.0)
    var_95 = np.clip(var_95, -1.0, 0.0)
    cvar_95 = np.clip(cvar_95, -1.0, 0.0)
    sortino_ratio = np.clip(sortino_ratio, -10.0, 10.0)
    
    return {
        'volatility': float(volatility),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'sortino_ratio': float(sortino_ratio)
    }

def calculate_all_metrics(
    portfolio_values: List[float],
    returns: List[float],
    trades: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate all metrics
    
    Args:
        portfolio_values: List of portfolio values
        returns: List of returns
        trades: List of trades
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    metrics.update(calculate_metrics(portfolio_values, returns))
    metrics.update(calculate_trade_metrics(trades))
    metrics.update(calculate_risk_metrics(returns))
    return metrics