"""
Scenario Data Generation Module
=============================

This module provides functions to generate various market scenario data
for backtesting purposes, such as flash crashes and low liquidity conditions.

There are two main ways to create scenario data:

1. Generate Synthetic Data (Complete OHLCV Generation):
   - Creates entirely artificial data with specified characteristics
   - Useful for controlled testing environments
   Example:
   ```python
   # Generate complete synthetic flash crash data
   data = generate_flash_crash_data(
       length=1000,
       crash_size=0.3,  # 30% drop
       crash_at=500     # Crash occurs at index 500
   )
   
   # Generate synthetic low liquidity data
   data = generate_low_liquidity_data(
       length=1000,
       volume_reduction=0.8,  # 80% volume reduction
       low_liq_start=300     # Low liquidity starts at index 300
   )
   ```

2. Modify Existing Market Data (Scenario Application):
   - Takes real market data and applies scenario characteristics
   - Maintains original market patterns outside scenario period
   - More realistic for production testing
   Example:
   ```python
   # Apply flash crash to real market data
   modified_data = apply_flash_crash_to_real_data(
       base_data=real_market_data,
       crash_size=0.3,
       crash_at=500
   )
   
   # Apply low liquidity to real market data
   modified_data = apply_low_liquidity_to_real_data(
       base_data=real_market_data,
       volume_reduction=0.8,
       low_liq_start=300
   )
   ```

Choose the appropriate method based on your needs:
- Use synthetic data generation for controlled testing and edge cases
- Use scenario application for more realistic backtesting with actual market patterns
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
    logger.info(f"Generating flash crash data with length={length}, crash_size={crash_size*100}%")
    
    # Set defaults
    if crash_at is None:
        crash_at = length // 2
        logger.debug(f"Using default crash_at={crash_at}")
    if crash_duration is None:
        crash_duration = 5  # Increased from 3 to 5
        logger.debug(f"Using default crash_duration={crash_duration}")
    if recovery_duration is None:
        recovery_duration = crash_duration * 2
        logger.debug(f"Using default recovery_duration={recovery_duration}")

    # Ensure crash_at has enough room for crash and recovery
    crash_at = min(crash_at, length - (crash_duration + recovery_duration))
    logger.debug(f"Adjusted crash_at to {crash_at} to ensure room for crash and recovery")
    
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
    logger.debug(f"Pre-crash price: {pre_crash_price:.2f}")
    
    # Add immediate flash crash with severe drop
    crash_price = pre_crash_price * (1 - crash_size)  # Ensure exact crash size at crash point
    prices[crash_at] = crash_price  # Immediate drop
    logger.debug(f"Crash price: {crash_price:.2f} (drop of {crash_size*100:.1f}%)")
    
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
    logger.debug(f"Initial bounce price: {initial_bounce:.2f}")
    
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
    
    logger.info(f"Generated flash crash data: shape={data.shape}, crash_at={crash_at}, final_price={prices[-1]:.2f}")
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
    
    # 1) Pre-crash period: Gradual 5% rise
    for i in range(1, crash_at):
        prices[i] = prices[i-1] * 1.0002  # Smaller deterministic increase
        
    # 2) Crash point: Immediate drop by crash_size
    pre_crash_price = prices[crash_at - 1]
    crash_price = pre_crash_price * (1 - crash_size)  # Exact crash size
    prices[crash_at] = crash_price
    
    # 3) Post-crash: Additional 10% drop over 5 periods
    post_crash_periods = 5
    additional_drop = 0.10  # 10% additional drop
    for i in range(crash_at + 1, crash_at + post_crash_periods):
        drop_progress = (i - crash_at) / post_crash_periods
        current_drop = additional_drop * drop_progress
        prices[i] = crash_price * (1 - current_drop)
        
    # 4) Recovery period: Gradual rise back to 80% of pre-crash
    recovery_target = pre_crash_price * 0.8
    recovery_start = crash_at + post_crash_periods
    recovery_periods = 10
    lowest_price = prices[recovery_start - 1]
    recovery_range = recovery_target - lowest_price
    
    for i in range(recovery_start, min(recovery_start + recovery_periods, length)):
        recovery_progress = (i - recovery_start) / recovery_periods
        prices[i] = lowest_price + (recovery_range * recovery_progress)
    
    # 5) Post-recovery: Slight upward trend
    if recovery_start + recovery_periods < length:
        for i in range(recovery_start + recovery_periods, length):
            prices[i] = prices[i-1] * 1.0001  # Very slow recovery
    
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
    base_price: float = 100.0,
    base_volume: float = 100000.0,
    volume_reduction: float = 0.9,
) -> pd.DataFrame:
    """Generate low liquidity scenario data
    
    Args:
        length (int): Total number of periods
        low_liq_start (int, optional): Start index of low liquidity period
        low_liq_length (int, optional): Length of low liquidity period
        base_price (float): Starting price level
        base_volume (float): Base volume level
        volume_reduction (float): Fraction to reduce volume by during low liquidity
        
    Returns:
        pd.DataFrame: OHLCV data with low liquidity period
    """
    logger.info(f"Generating low liquidity data with length={length}, volume_reduction={volume_reduction*100}%")
    
    if low_liq_start is None:
        low_liq_start = length // 3
        logger.debug(f"Using default low_liq_start={low_liq_start}")
    
    if low_liq_length is None:
        low_liq_length = length // 10
        logger.debug(f"Using default low_liq_length={low_liq_length}")

    # Ensure low_liq_start has enough room
    low_liq_start = min(low_liq_start, length - low_liq_length)
    logger.debug(f"Adjusted low_liq_start to {low_liq_start}")
    
    timestamps = pd.date_range(
        start="2024-01-01", periods=length, freq="5min"
    )
    
    # Generate price series with increased volatility during low liquidity
    returns = np.random.normal(0, 0.001, size=length)
    returns[low_liq_start : low_liq_start + low_liq_length] *= 3  # Triple volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate normal volumes with small random variation
    volumes = np.full(length, base_volume)
    volumes *= np.random.uniform(0.95, 1.05, length)  # ±5% variation
    
    # Apply strict volume reduction during low liquidity period
    # Ensure the maximum volume during low liquidity is below the target reduction
    reduced_volume = base_volume * (1 - volume_reduction)  # Target reduced volume
    low_liq_volumes = np.random.uniform(
        reduced_volume * 0.9,  # Allow 10% below target
        reduced_volume,        # But never exceed target
        low_liq_length
    )
    volumes[low_liq_start : low_liq_start + low_liq_length] = low_liq_volumes
    
    logger.debug(f"Normal volume: {base_volume:.0f}, Reduced volume: {reduced_volume:.0f}")
    
    # Generate spreads with wider spreads during low liquidity
    normal_spreads = np.random.uniform(0.001, 0.002, length)  # 0.1-0.2% spread normally
    spreads = normal_spreads.copy()
    # Increase spreads by 3-5x during low liquidity
    spreads[low_liq_start : low_liq_start + low_liq_length] = np.random.uniform(0.003, 0.01, low_liq_length)
    
    logger.debug(f"Normal spread: {normal_spreads.mean():.4f}, Low liquidity spread: {spreads[low_liq_start:low_liq_start+low_liq_length].mean():.4f}")

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
    
    # Verify volume reduction
    normal_vol = volumes[:low_liq_start].mean()
    low_liq_vol = volumes[low_liq_start:low_liq_start+low_liq_length].mean()
    actual_reduction = (normal_vol - low_liq_vol) / normal_vol
    logger.info(f"Generated low liquidity data: shape={data.shape}, target_reduction={volume_reduction*100:.1f}%, actual_reduction={actual_reduction*100:.1f}%")
    
    return data

def calculate_flash_crash_metrics(results: Dict) -> Dict:
    """Calculate metrics specific to flash crash scenario
    
    Args:
        results (Dict): Results from backtester run including:
            - portfolio_values: List of portfolio values
            - trades: List of trade dictionaries
            - timestamps: List of timestamps
            - prices: Dict with OHLCV data
        
    Returns:
        Dict: Scenario-specific metrics including:
            - max_drawdown_idx: Index of maximum drawdown
            - recovery_time_periods: Number of periods to recover
            - survived_crash: Whether portfolio survived (>50% of initial)
            - crash_size: Size of the crash as percentage
            - recovery_speed: Number of bars to recover to pre-crash value
            - recovery_percentage: Final portfolio value as % of pre-crash value
            - drawdown_depth: Maximum portfolio drawdown during crash
            - crash_trade_efficacy: Score of trade decisions during crash
    """
    portfolio_values = np.array(results["portfolio_values"])
    trades = results.get("trades", [])
    prices = results.get("prices", {})
    
    # Find crash period
    max_drawdown_idx = np.argmax(
        np.maximum.accumulate(portfolio_values) - portfolio_values
    )
    
    # Pre-crash metrics
    pre_crash_value = portfolio_values[max_drawdown_idx - 1]
    crash_bottom = min(portfolio_values[max_drawdown_idx:])
    
    # Recovery metrics
    recovery_mask = portfolio_values[max_drawdown_idx:] >= pre_crash_value
    recovery_periods = np.argmax(recovery_mask) if any(recovery_mask) else len(portfolio_values) - max_drawdown_idx
    recovery_percentage = (portfolio_values[-1] / pre_crash_value) * 100
    
    # Trade efficacy during crash (±20 periods around crash)
    crash_start = max(0, max_drawdown_idx - 20)
    crash_end = min(len(portfolio_values), max_drawdown_idx + 20)
    crash_trades = [
        t for t in trades
        if crash_start <= results["timestamps"].index(t["timestamp"]) < crash_end
    ]
    
    # Score trades: +1 for selling during crash, +1 for buying during recovery
    trade_score = 0
    if crash_trades:
        for trade in crash_trades:
            trade_idx = results["timestamps"].index(trade["timestamp"])
            if trade_idx < max_drawdown_idx:  # Pre-crash
                trade_score += -1 if trade["amount"] > 0 else 1  # Reward selling
            else:  # Post-crash
                trade_score += 1 if trade["amount"] > 0 else -1  # Reward buying
        trade_score /= len(crash_trades)  # Normalize to [-1, 1]

    return {
        "max_drawdown_idx": max_drawdown_idx,
        "recovery_time_periods": len(portfolio_values) - max_drawdown_idx,
        "survived_crash": portfolio_values[-1] > portfolio_values[0] * 0.5,
        "crash_size": (pre_crash_value - crash_bottom) / pre_crash_value,
        "recovery_speed": recovery_periods,
        "recovery_percentage": recovery_percentage,
        "drawdown_depth": (pre_crash_value - crash_bottom) / pre_crash_value * 100,
        "crash_trade_efficacy": trade_score
    }

def calculate_low_liquidity_metrics(results: Dict) -> Dict:
    """Calculate metrics specific to low liquidity scenario
    
    Args:
        results (Dict): Results from backtester run including trades and timestamps
        
    Returns:
        Dict: Scenario-specific metrics including:
            - avg_trade_cost: Average trading cost
            - trade_count_low_liq: Number of trades during low liquidity
            - avg_slippage: Average slippage
            - max_slippage: Maximum slippage
            - fill_rate: Percentage of orders that were filled
            - avg_spread: Average bid-ask spread during low liquidity
            - execution_delay: Average execution delay in periods
    """
    trades = results.get("trades", [])
    prices = results.get("prices", {})
    timestamps = results["timestamps"]
    
    # Basic metrics
    trade_costs = [trade.get("fee", 0) for trade in trades]
    avg_cost = np.mean(trade_costs) if trade_costs else 0
    
    # Low liquidity period (assume 300-400 as in original)
    low_liq_trades = [
        t for t in trades
        if 300 <= timestamps.index(t["timestamp"]) < 400
    ]
    
    # Fill rate calculation (assuming failed trades are marked with 'filled': False)
    total_orders = len([t for t in low_liq_trades if "filled" in t])
    filled_orders = len([t for t in low_liq_trades if t.get("filled", True)])
    fill_rate = (filled_orders / total_orders * 100) if total_orders > 0 else 100.0
    
    # Spread calculation (using high-low as proxy when bid-ask not available)
    if prices and "$high" in prices and "$low" in prices:
        spreads = [
            (h - l) / ((h + l) / 2) * 100  # Spread as percentage
            for h, l in zip(
                prices["$high"][300:400],
                prices["$low"][300:400]
            )
        ]
        avg_spread = np.mean(spreads) if spreads else 0
    else:
        avg_spread = 0
    
    # Execution delay (if available in trade data)
    delays = [
        t.get("execution_delay", 0)
        for t in low_liq_trades
        if "execution_delay" in t
    ]
    avg_delay = np.mean(delays) if delays else 0

    return {
        "avg_trade_cost": avg_cost,
        "trade_count_low_liq": len(low_liq_trades),
        "avg_slippage": np.mean([t.get("slippage", 0) for t in trades]),
        "max_slippage": max([t.get("slippage", 0) for t in trades], default=0),
        "fill_rate": fill_rate,
        "avg_spread": avg_spread,
        "execution_delay": avg_delay
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

def apply_flash_crash_to_real_data(
    base_data: pd.DataFrame,
    crash_at: Optional[int] = None,
    crash_duration: Optional[int] = None,
    crash_size: float = 0.3,
    recovery_duration: Optional[int] = None,
) -> pd.DataFrame:
    """Apply flash crash scenario to existing market data
    
    Args:
        base_data (pd.DataFrame): Base market data with OHLCV columns
        crash_at (int, optional): Index at which crash occurs. If None, will be set to len(data)//2
        crash_duration (int, optional): Duration of crash in periods. If None, defaults to 5
        crash_size (float): Size of crash as fraction of price (0.3 means 30% drop)
        recovery_duration (int, optional): Duration of recovery in periods. If None, defaults to 2x crash_duration
        
    Returns:
        pd.DataFrame: Modified market data with flash crash applied
    """
    logger.info(f"Applying flash crash to real data with crash_size={crash_size*100}%")
    data = base_data.copy()
    length = len(data)
    
    # Set defaults
    if crash_at is None:
        crash_at = length // 2
        logger.debug(f"Using default crash_at={crash_at}")
    if crash_duration is None:
        crash_duration = 5
        logger.debug(f"Using default crash_duration={crash_duration}")
    if recovery_duration is None:
        recovery_duration = crash_duration * 2
        logger.debug(f"Using default recovery_duration={recovery_duration}")
    
    # Ensure crash_at has enough room
    crash_at = min(crash_at, length - (crash_duration + recovery_duration))
    logger.debug(f"Adjusted crash_at to {crash_at}")
    
    # Store original values
    pre_crash_close = data["$close"].iloc[crash_at - 1]
    pre_crash_volume = data["$volume"].iloc[:crash_at].mean()
    logger.debug(f"Pre-crash price: {pre_crash_close:.2f}, volume: {pre_crash_volume:.0f}")
    
    # Apply crash
    crash_price = pre_crash_close * (1 - crash_size)
    data.loc[data.index[crash_at], "$close"] = crash_price
    data.loc[data.index[crash_at], "$open"] = pre_crash_close
    data.loc[data.index[crash_at], "$low"] = crash_price * 0.95  # Additional 5% wick
    data.loc[data.index[crash_at], "$high"] = pre_crash_close
    
    # Increase volume during crash
    data.loc[data.index[crash_at], "$volume"] *= np.random.uniform(5, 8)
    
    # Apply continued decline with high volatility
    for i in range(crash_at + 1, crash_at + crash_duration):
        prev_close = data.loc[data.index[i-1], "$close"]
        decline = np.random.uniform(-0.05, 0.02)  # Continued decline with some bounces
        new_close = prev_close * (1 + decline)
        new_close = max(new_close, crash_price * 0.9)  # Limit additional decline
        
        data.loc[data.index[i], "$close"] = new_close
        data.loc[data.index[i], "$open"] = prev_close
        data.loc[data.index[i], "$low"] = new_close * 0.98
        data.loc[data.index[i], "$high"] = prev_close * 1.02
        data.loc[data.index[i], "$volume"] *= np.random.uniform(3, 5)  # Elevated volume
    
    # Recovery phase
    recovery_start = crash_at + crash_duration
    recovery_end = recovery_start + recovery_duration
    
    # Initial bounce - Increased from 10% to 15%
    bounce_price = data.loc[data.index[recovery_start-1], "$close"] * 1.15
    data.loc[data.index[recovery_start], "$close"] = bounce_price
    data.loc[data.index[recovery_start], "$open"] = data.loc[data.index[recovery_start-1], "$close"]
    data.loc[data.index[recovery_start], "$low"] = data.loc[data.index[recovery_start-1], "$close"]
    data.loc[data.index[recovery_start], "$high"] = bounce_price * 1.05
    data.loc[data.index[recovery_start], "$volume"] *= np.random.uniform(3, 5)  # Increased volume spike
    
    # Gradual recovery with stronger upward bias
    for i in range(recovery_start + 1, recovery_end):
        prev_close = data.loc[data.index[i-1], "$close"]
        recovery = np.random.uniform(0.001, 0.05)  # Removed negative bias, increased upper bound
        new_close = prev_close * (1 + recovery)
        
        data.loc[data.index[i], "$close"] = new_close
        data.loc[data.index[i], "$open"] = prev_close
        data.loc[data.index[i], "$low"] = min(prev_close, new_close) * 0.99
        data.loc[data.index[i], "$high"] = max(prev_close, new_close) * 1.01
        data.loc[data.index[i], "$volume"] *= np.random.uniform(2, 3)  # Increased volume during recovery
    
    logger.info(f"Applied flash crash: pre_crash={pre_crash_close:.2f}, bottom={crash_price:.2f}, final={data['$close'].iloc[-1]:.2f}")
    return data

def apply_low_liquidity_to_real_data(
    base_data: pd.DataFrame,
    low_liq_start: Optional[int] = None,
    low_liq_length: Optional[int] = None,
    volume_reduction: float = 0.9,
    spread_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Apply low liquidity scenario to existing market data
    
    Args:
        base_data (pd.DataFrame): Base market data with OHLCV columns
        low_liq_start (int, optional): Start index of low liquidity period
        low_liq_length (int, optional): Length of low liquidity period
        volume_reduction (float): Fraction to reduce volume by (0.9 = 90% reduction)
        spread_multiplier (float): Factor to multiply spreads by during low liquidity
        
    Returns:
        pd.DataFrame: Modified market data with low liquidity applied
    """
    logger.info(f"Applying low liquidity to real data with volume_reduction={volume_reduction*100}%")
    data = base_data.copy()
    length = len(data)
    
    # Set defaults
    if low_liq_start is None:
        low_liq_start = length // 3
        logger.debug(f"Using default low_liq_start={low_liq_start}")
    if low_liq_length is None:
        low_liq_length = length // 10
        logger.debug(f"Using default low_liq_length={low_liq_length}")
    
    # Ensure low_liq_start has enough room
    low_liq_start = min(low_liq_start, length - low_liq_length)
    low_liq_end = low_liq_start + low_liq_length
    logger.debug(f"Adjusted low_liq_start to {low_liq_start}, end at {low_liq_end}")
    
    # Store original characteristics
    normal_volume = data["$volume"].iloc[:low_liq_start].mean()
    normal_spread = ((data["$high"] - data["$low"]) / data["$close"]).iloc[:low_liq_start].mean()
    logger.debug(f"Normal volume: {normal_volume:.0f}, spread: {normal_spread*100:.2f}%")
    
    # Apply volume reduction
    reduced_volume = normal_volume * (1 - volume_reduction)
    data.loc[data.index[low_liq_start:low_liq_end], "$volume"] *= (1 - volume_reduction)
    data.loc[data.index[low_liq_start:low_liq_end], "$volume"] *= np.random.uniform(0.98, 1.02, low_liq_length)
    
    # Increase spreads
    for i in range(low_liq_start, low_liq_end):
        mid_price = data.loc[data.index[i], "$close"]
        original_spread = (data.loc[data.index[i], "$high"] - data.loc[data.index[i], "$low"]) / mid_price
        new_spread = original_spread * spread_multiplier
        
        data.loc[data.index[i], "$high"] = mid_price * (1 + new_spread/2)
        data.loc[data.index[i], "$low"] = mid_price * (1 - new_spread/2)
    
    # Increase price volatility
    returns = data["$close"].pct_change()
    volatility = returns.std()
    
    for i in range(low_liq_start + 1, low_liq_end):
        extra_vol = np.random.normal(0, volatility * 2)  # Double volatility
        current_price = data.loc[data.index[i], "$close"]
        data.loc[data.index[i], "$close"] = current_price * (1 + extra_vol)
        data.loc[data.index[i], "$open"] = current_price
    
    logger.info(f"Applied low liquidity: avg_volume_reduction={(1 - data['$volume'].iloc[low_liq_start:low_liq_end].mean()/normal_volume)*100:.1f}%")
    return data 