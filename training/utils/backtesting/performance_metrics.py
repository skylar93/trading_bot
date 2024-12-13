"""
Performance metrics calculation for backtesting results
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate return-based performance metrics
    
    Args:
        returns: Series of period returns
        
    Returns:
        Dictionary of metrics
    """
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252/len(returns)) - 1
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)
    tracking_error = returns.sub(returns.mean()).std() * np.sqrt(252)
    
    # Risk-adjusted returns
    rf_rate = 0.02  # Assumed risk-free rate
    excess_returns = returns - rf_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    max_drawdown = calculate_max_drawdown(returns)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'tracking_error': tracking_error,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
    }

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-based performance metrics
    
    Args:
        trades_df: DataFrame of trades
        
    Returns:
        Dictionary of metrics
    """
    if trades_df.empty:
        return {}
        
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['value'] > 0])
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = trades_df[trades_df['value'] > 0]['value'].sum()
    total_loss = abs(trades_df[trades_df['value'] < 0]['value'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_trade_return = trades_df['value'].mean()
    avg_win = trades_df[trades_df['value'] > 0]['value'].mean()
    avg_loss = trades_df[trades_df['value'] < 0]['value'].mean()
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_return': avg_trade_return,
        'avg_win': avg_win or 0,
        'avg_loss': avg_loss or 0,
    }

def calculate_position_metrics(positions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate position-based metrics
    
    Args:
        positions_df: DataFrame of position values over time
        
    Returns:
        Dictionary of metrics
    """
    # Calculate average position sizes
    avg_position_size = positions_df.mean()
    max_position_size = positions_df.max()
    position_turnover = positions_df.diff().abs().mean()
    
    # Calculate concentration metrics
    concentration = (positions_df ** 2).sum(axis=1).mean()
    max_concentration = (positions_df ** 2).sum(axis=1).max()
    
    return {
        'avg_position_size': avg_position_size,
        'max_position_size': max_position_size,
        'position_turnover': position_turnover,
        'concentration': concentration,
        'max_concentration': max_concentration,
    }

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as positive percentage
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    return abs(drawdowns.min())

def calculate_all_metrics(
    returns: pd.Series,
    trades_df: pd.DataFrame,
    positions_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate all performance metrics
    
    Args:
        returns: Series of period returns
        trades_df: DataFrame of trades
        positions_df: DataFrame of position values
        
    Returns:
        Nested dictionary of all metrics
    """
    return {
        'returns': calculate_returns_metrics(returns),
        'trades': calculate_trade_metrics(trades_df),
        'positions': calculate_position_metrics(positions_df),
    }
