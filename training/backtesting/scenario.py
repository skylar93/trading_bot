"""
Scenario Data Generation Module
=============================

This module provides functions to generate various market scenario data
for backtesting purposes, such as flash crashes and low liquidity conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_flash_crash_data(
    length: int = 1000,
    crash_at: Optional[int] = None,
    crash_duration: Optional[int] = None,
    crash_size: float = 0.3,  # Increased from 0.15 to 0.3 (30% drop)
    recovery_duration: Optional[int] = None,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Generate flash crash scenario data
    
    Args:
        length (int): Total number of periods
        crash_at (int, optional): Index at which crash occurs. If None, will be set to length//2
        crash_duration (int, optional): Duration of crash in periods. If None, defaults to 5
        crash_size (float): Size of crash as fraction of price (0.3 means 30% drop)
        recovery_duration (int, optional): Duration of recovery in periods. If None, defaults to 2x crash_duration
        base_price (float): Starting price level
        
    Returns:
        pd.DataFrame: OHLCV data with flash crash
    """
    # Set defaults
    if crash_at is None:
        crash_at = length // 2
    if crash_duration is None:
        crash_duration = 5  # Increased from 3 to 5
    if recovery_duration is None:
        recovery_duration = crash_duration * 2

    # Ensure crash_at has enough room for crash and recovery
    crash_at = min(crash_at, length - (crash_duration + recovery_duration))
    
    # Generate base price series with some volatility
    timestamps = pd.date_range(
        start="2024-01-01", periods=length, freq="5min"
    )
    
    # Start with constant base price but add some initial volatility
    prices = np.full(length, base_price)
    pre_crash_returns = np.random.normal(0.0002, 0.002, crash_at)  # Slight upward trend with volatility
    prices[:crash_at] *= np.exp(np.cumsum(pre_crash_returns))
    
    # Store pre-crash price
    pre_crash_price = prices[crash_at - 1]
    
    # Add immediate flash crash with severe drop
    crash_price = pre_crash_price * (1 - crash_size)  # Ensure exact crash size at crash point
    prices[crash_at] = crash_price  # Immediate drop
    
    # Add further decline with high volatility during crash period
    crash_volatility = np.random.normal(-0.03, 0.04, crash_duration - 1)  # Increased volatility with stronger downward bias
    crash_prices = crash_price * np.exp(np.cumsum(crash_volatility))
    crash_prices = np.minimum(crash_prices, crash_price * 0.9)  # Allow further drops up to 90% of crash price
    prices[crash_at + 1 : crash_at + crash_duration] = crash_prices
    
    # Add recovery with high volatility
    recovery_start = crash_at + crash_duration
    recovery_end = recovery_start + recovery_duration
    
    # Initial bounce of 40% of the drop (increased from 30%)
    initial_bounce = crash_price * (1 + 0.4 * crash_size)
    prices[recovery_start] = initial_bounce
    
    # Volatile recovery period with higher volatility
    recovery_volatility = np.random.normal(0.002, 0.02, recovery_duration - 1)  # Increased volatility
    recovery_prices = initial_bounce * np.exp(np.cumsum(recovery_volatility))
    prices[recovery_start + 1:recovery_end] = recovery_prices
    
    # Add post-recovery volatility
    post_recovery_returns = np.random.normal(-0.0001, 0.003, length - recovery_end)
    if len(post_recovery_returns) > 0:
        prices[recovery_end:] = prices[recovery_end - 1] * np.exp(np.cumsum(post_recovery_returns))
    
    # Generate OHLCV data with extreme spreads during crash
    spreads = np.random.uniform(0, 0.002, length)  # Normal spreads
    spreads[crash_at : crash_at + crash_duration] = np.random.uniform(0.02, 0.08, crash_duration)  # More extreme spreads during crash
    
    # Generate volume with larger spikes during crash
    volumes = np.random.uniform(100000, 200000, length)
    volumes[crash_at : crash_at + crash_duration] *= np.random.uniform(5, 8, crash_duration)  # Larger volume spikes during crash
    
    # Ensure OHLCV data maintains the crash characteristics
    data = pd.DataFrame(
        {
            "$open": prices,  # Use exact prices
            "$high": prices * (1 + spreads),
            "$low": prices * (1 - spreads),
            "$close": prices,  # Use exact prices
            "$volume": volumes,
        },
        index=timestamps,
    )

    return data

def generate_flash_crash_data_deterministic(
    length: int = 1000,
    crash_at: Optional[int] = None,
    crash_size: float = 0.3,  # 30% drop
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Generate deterministic flash crash scenario data
    
    Args:
        length (int): Total number of periods
        crash_at (int, optional): Index at which crash occurs. If None, will be set to length//2
        crash_size (float): Size of crash as fraction of price (0.3 means 30% drop)
        base_price (float): Starting price level
        
    Returns:
        pd.DataFrame: OHLCV data with deterministic flash crash
    """
    if crash_at is None:
        crash_at = length // 2
        
    # Ensure crash_at has enough room
    crash_at = min(crash_at, length - 20)  # At least 20 periods for post-crash
    
    # Generate timestamps
    timestamps = pd.date_range(
        start="2024-01-01", periods=length, freq="5min"
    )
    
    # Initialize prices array
    prices = np.full(length, base_price)
    
    # 1) Pre-crash period: Gradual 10% rise
    for i in range(1, crash_at):
        prices[i] = prices[i-1] * 1.0005  # Small deterministic increase
        
    # 2) Crash point: Immediate 30% drop
    pre_crash_price = prices[crash_at - 1]
    crash_price = pre_crash_price * (1 - crash_size)
    prices[crash_at] = crash_price
    
    # 3) Post-crash: Additional 5% drop over 10 periods
    post_crash_periods = 10
    for i in range(crash_at + 1, crash_at + post_crash_periods):
        prices[i] = crash_price * (0.95 + 0.05 * (i - crash_at - 1) / post_crash_periods)
        
    # 4) Recovery period: Gradual rise
    for i in range(crash_at + post_crash_periods, length):
        prices[i] = prices[i-1] * 1.001  # Slow deterministic recovery
        
    # Generate OHLCV data with fixed spreads
    spreads = np.full(length, 0.001)  # 0.1% spread normally
    spreads[crash_at:crash_at + post_crash_periods] = 0.05  # 5% spread during crash
    
    # Generate volume with spikes during crash
    volumes = np.full(length, 100000.0)  # Normal volume
    volumes[crash_at:crash_at + post_crash_periods] *= 5  # 5x volume during crash
    
    data = pd.DataFrame(
        {
            "$open": prices,
            "$high": prices * (1 + spreads),
            "$low": prices * (1 - spreads),
            "$close": prices,
            "$volume": volumes,
        },
        index=timestamps,
    )
    
    return data

def generate_low_liquidity_data(
    length: int = 1000,
    low_liq_start: Optional[int] = None,
    low_liq_length: Optional[int] = None,
    low_liq_duration: Optional[int] = None,  # Alias for low_liq_length
    base_price: float = 100.0,
    base_volume: float = 100000.0,
    volume_reduction: float = 0.9,  # 90% reduction in volume
) -> pd.DataFrame:
    """Generate low liquidity scenario data
    
    Args:
        length (int): Total number of periods
        low_liq_start (int, optional): Start index of low liquidity period. If None, will be set to length//3
        low_liq_length (int, optional): Length of low liquidity period. If None, uses low_liq_duration or defaults to length//10
        low_liq_duration (int, optional): Alias for low_liq_length
        base_price (float): Starting price level
        base_volume (float): Base volume level
        volume_reduction (float): Fraction to reduce volume by during low liquidity (0.9 = 90% reduction)
        
    Returns:
        pd.DataFrame: OHLCV data with low liquidity period
    """
    # Set defaults and handle parameter aliases
    if low_liq_start is None:
        low_liq_start = length // 3
    
    # Use low_liq_duration as fallback for low_liq_length
    if low_liq_length is None:
        low_liq_length = low_liq_duration if low_liq_duration is not None else length // 10

    # Ensure low_liq_start has enough room
    low_liq_start = min(low_liq_start, length - low_liq_length)
    
    timestamps = pd.date_range(
        start="2024-01-01", periods=length, freq="5min"
    )
    returns = np.random.normal(0, 0.001, size=length)

    # Increase volatility during low liquidity period
    returns[low_liq_start : low_liq_start + low_liq_length] *= 3

    prices = base_price * np.exp(np.cumsum(returns))

    # Generate volumes with low liquidity period
    # Use fixed base volume for normal periods
    volumes = np.full(length, base_volume)
    # Add small random variation to normal periods
    volumes *= np.random.uniform(0.98, 1.02, length)
    
    # Apply volume reduction with minimal variation
    reduced_volume = base_volume * (1 - volume_reduction)
    volumes[low_liq_start : low_liq_start + low_liq_length] = reduced_volume
    # Add minimal variation to low liquidity period
    volumes[low_liq_start : low_liq_start + low_liq_length] *= np.random.uniform(0.98, 1.0, low_liq_length)

    data = pd.DataFrame(
        {
            "$open": prices,
            "$high": prices * (1 + np.random.uniform(0, 0.002, length)),
            "$low": prices * (1 - np.random.uniform(0, 0.002, length)),
            "$close": prices,
            "$volume": volumes,
        },
        index=timestamps,
    )

    return data

def calculate_flash_crash_metrics(results: Dict) -> Dict:
    """Calculate metrics specific to flash crash scenario
    
    Args:
        results (Dict): Results from backtester run
        
    Returns:
        Dict: Scenario-specific metrics
    """
    portfolio_values = results["portfolio_values"]
    max_drawdown_idx = np.argmax(
        np.maximum.accumulate(portfolio_values) - portfolio_values
    )
    recovery_time = len(portfolio_values) - max_drawdown_idx

    return {
        "max_drawdown_idx": max_drawdown_idx,
        "recovery_time_periods": recovery_time,
        "survived_crash": portfolio_values[-1] > portfolio_values[0] * 0.5,
        "crash_size": (
            max(portfolio_values[:max_drawdown_idx])
            - min(portfolio_values[max_drawdown_idx:])
        )
        / max(portfolio_values[:max_drawdown_idx]),
    }

def calculate_low_liquidity_metrics(results: Dict) -> Dict:
    """Calculate metrics specific to low liquidity scenario
    
    Args:
        results (Dict): Results from backtester run
        
    Returns:
        Dict: Scenario-specific metrics
    """
    trade_costs = [trade.get("fee", 0) for trade in results["trades"]]
    avg_cost = np.mean(trade_costs) if trade_costs else 0
    timestamps = results["timestamps"]

    return {
        "avg_trade_cost": avg_cost,
        "trade_count_low_liq": len(
            [
                t
                for t in results["trades"]
                if 300 <= timestamps.index(t["timestamp"]) < 400
            ]
        ),
        "avg_slippage": np.mean(
            [t.get("slippage", 0) for t in results["trades"]]
        ),
        "max_slippage": max(
            [t.get("slippage", 0) for t in results["trades"]], default=0
        ),
    }

def plot_scenario_results(
    results: Dict,
    scenario_type: str,
    save_path: str = None,
):
    """Plot scenario-specific results with annotations
    
    Args:
        results (Dict): Results from backtester run
        scenario_type (str): Type of scenario ('flash_crash' or 'low_liquidity')
        save_path (str, optional): Path to save plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    plt.figure(figsize=(12, 6))
    plt.plot(results["timestamps"], results["portfolio_values"], label="Portfolio Value")
    
    if scenario_type == "flash_crash":
        crash_idx = results["scenario_metrics"]["max_drawdown_idx"]
        plt.axvline(
            x=results["timestamps"][crash_idx],
            color="r",
            linestyle="--",
            label="Flash Crash",
        )
        plt.text(
            results["timestamps"][crash_idx],
            plt.ylim()[1],
            "Flash Crash",
            rotation=90,
        )

    elif scenario_type == "low_liquidity":
        plt.axvspan(
            results["timestamps"][300],
            results["timestamps"][400],
            alpha=0.2,
            color="yellow",
            label="Low Liquidity",
        )

    plt.title(f"{scenario_type.replace('_', ' ').title()} Scenario Results")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def calculate_scenario_metrics(
    results: Dict[str, Any],
    scenario_type: str,
    scenario_params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate scenario-specific metrics from backtest results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from backtester run
    scenario_type : str
        Type of scenario ('flash_crash' or 'low_liquidity')
    scenario_params : Dict[str, Any]
        Parameters used to generate the scenario
        
    Returns
    -------
    Dict[str, float]
        Scenario-specific metrics
    """
    metrics = {}
    
    # Convert portfolio values to DataFrame if it's a list
    portfolio_values = pd.Series(results['portfolio_values'], index=results['timestamps'])
    
    if scenario_type == 'flash_crash':
        crash_day = scenario_params['crash_day']
        crash_period = portfolio_values.index[crash_day:crash_day + 5]
        
        # Maximum drawdown during crash
        crash_values = portfolio_values[crash_period]
        max_drawdown = (crash_values.max() - crash_values.min()) / crash_values.max()
        metrics['crash_max_drawdown'] = max_drawdown
        
        # Recovery ratio (final value / pre-crash value)
        pre_crash_value = portfolio_values.iloc[crash_day - 1]
        post_crash_value = portfolio_values.iloc[crash_day + 5]
        metrics['recovery_ratio'] = post_crash_value / pre_crash_value
        
    elif scenario_type == 'low_liquidity':
        start_day = scenario_params['low_liq_start']
        duration = scenario_params['low_liq_duration']
        liq_period = portfolio_values.index[start_day:start_day + duration]
        
        # Average trade size during low liquidity
        period_trades = [t for t in results['trades'] 
                        if t['timestamp'] in liq_period]
        if period_trades:
            avg_trade_size = np.mean([abs(t['amount']) for t in period_trades])
            metrics['avg_trade_size_low_liq'] = avg_trade_size
        
        # Volatility of returns during low liquidity
        period_returns = portfolio_values[liq_period].pct_change().dropna()
        metrics['return_vol_low_liq'] = period_returns.std()
        
    return metrics 