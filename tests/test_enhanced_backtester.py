"""
Test script for the EnhancedBacktester class.

This script tests the functionality of the EnhancedBacktester and compares it 
with the BaseBacktester to ensure compatibility.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.enhanced_backtester import EnhancedBacktester
from training.backtesting.risk_manager import RiskConfig
from data.utils.enhanced_data_loader import EnhancedDataLoader
from training.backtesting.market_simulator import MarketSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_enhanced_backtester")

# Create test output directory
TEST_OUTPUT_DIR = Path("test_results/backtester")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleStrategy:
    """
    A simple strategy for testing purposes:
    - Buy when price is increasing (momentum)
    - Sell when price is decreasing
    """
    
    def __init__(self, window_size: int = 5, threshold: float = 0.01):
        self.window_size = window_size
        self.threshold = threshold
        
    def get_action(self, window_data):
        """
        Return a simple action based on price movement.
        This matches the signature expected by BaseBacktester.
        
        Args:
            window_data: DataFrame with historical price data
            
        Returns:
            float: Action value between 0 and 1 (fractional holding)
        """
        if '$close' not in window_data.columns:
            return 0.5  # Hold current position
            
        if len(window_data) < self.window_size:
            return 0.5  # Not enough data, hold
            
        # Simple momentum: compare current price to average of previous prices
        prices = window_data['$close'].values
        current_price = prices[-1]
        avg_price = np.mean(prices[-self.window_size:])
        
        if current_price > avg_price * (1 + self.threshold):
            return 0.8  # Price increasing, buy
        elif current_price < avg_price * (1 - self.threshold):
            return 0.2  # Price decreasing, sell
        else:
            return 0.5  # No significant movement, hold
    
    def get_actions(self, window_data, symbols):
        """
        Return actions for multiple symbols.
        This is used by the EnhancedBacktester for multi-asset trading.
        
        Args:
            window_data: DataFrame with historical price data for multiple assets
            symbols: List of symbols to generate actions for
            
        Returns:
            dict: Mapping of symbols to action values (0-1)
        """
        actions = {}
        
        for symbol in symbols:
            # For the default case
            if symbol == "default":
                col = "$close"
            else:
                # For multi-asset case, the column naming follows symbol_$close format
                col = f"{symbol}_$close"
                
            if col in window_data.columns:
                prices = window_data[col].values
                if len(prices) >= self.window_size:
                    current_price = prices[-1]
                    avg_price = np.mean(prices[-self.window_size:])
                    
                    if current_price > avg_price * (1 + self.threshold):
                        actions[symbol] = 0.8  # Buy signal
                    elif current_price < avg_price * (1 - self.threshold):
                        actions[symbol] = 0.2  # Sell signal
                    else:
                        actions[symbol] = 0.5  # Hold
                else:
                    actions[symbol] = 0.5  # Not enough data, hold
            else:
                actions[symbol] = 0.5  # No data for this symbol, hold
                
        return actions


def generate_test_data(days: int = 100, symbols: List[str] = None, with_correlation: bool = True) -> pd.DataFrame:
    """
    Generate synthetic price data for testing with optional correlation between assets.
    
    Args:
        days: Number of days of data to generate
        symbols: List of symbols to generate data for
        with_correlation: Whether to correlate asset price movements
        
    Returns:
        DataFrame with OHLCV data
    """
    if symbols is None:
        symbols = ["BTC/USDT"]
        
    # Generate date range
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), 
        end=datetime.now(), 
        freq='1h'
    )
    
    # Initialize DataFrame
    data = pd.DataFrame(index=dates)
    
    # Base parameters
    price_volatilities = {
        "BTC/USDT": 0.02,
        "ETH/USDT": 0.025,
        "XRP/USDT": 0.03,
        "SOL/USDT": 0.035,
        "ADA/USDT": 0.028,
    }
    
    base_prices = {
        "BTC/USDT": 30000,
        "ETH/USDT": 2000,
        "XRP/USDT": 0.5,
        "SOL/USDT": 80,
        "ADA/USDT": 0.4,
    }
    
    # Correlation matrix
    if with_correlation:
        # Positive correlation between BTC and ETH
        # Negative correlation between BTC and XRP
        # Random correlation for others
        correlation_matrix = {
            ("BTC/USDT", "ETH/USDT"): 0.8,
            ("ETH/USDT", "BTC/USDT"): 0.8,
            ("BTC/USDT", "XRP/USDT"): -0.3,
            ("XRP/USDT", "BTC/USDT"): -0.3,
            ("SOL/USDT", "ETH/USDT"): 0.6,
            ("ETH/USDT", "SOL/USDT"): 0.6,
        }
    else:
        correlation_matrix = {}
    
    # Generate random returns for each symbol
    returns = {}
    for symbol in symbols:
        if symbol in price_volatilities:
            volatility = price_volatilities[symbol]
        else:
            volatility = 0.02  # Default
            
        returns[symbol] = np.random.normal(0, volatility, len(dates))
    
    # Apply correlations
    if with_correlation:
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i < j:  # Only process each pair once
                    corr_key = (symbol1, symbol2)
                    if corr_key in correlation_matrix:
                        corr = correlation_matrix[corr_key]
                        
                        # Apply correlation
                        common_factor = np.random.normal(0, 0.01, len(dates))
                        returns[symbol1] = (1 - abs(corr)) * returns[symbol1] + abs(corr) * common_factor
                        
                        if corr >= 0:
                            returns[symbol2] = (1 - abs(corr)) * returns[symbol2] + abs(corr) * common_factor
                        else:
                            returns[symbol2] = (1 - abs(corr)) * returns[symbol2] - abs(corr) * common_factor
    
    # Generate price paths
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100)
        
        # Generate prices using returns
        prices = [base_price]
        for ret in returns[symbol]:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove initial price
        
        # Generate OHLCV data
        if len(symbols) == 1:
            # Single-asset format
            data["$open"] = prices
            data["$high"] = [p * (1 + np.random.uniform(0, 0.01)) for p in prices]
            data["$low"] = [p * (1 - np.random.uniform(0, 0.01)) for p in prices]
            data["$close"] = prices
            data["$volume"] = [p * np.random.uniform(100, 1000) for p in prices]
        else:
            # Multi-asset format
            data[f"{symbol}_$open"] = prices
            data[f"{symbol}_$high"] = [p * (1 + np.random.uniform(0, 0.01)) for p in prices]
            data[f"{symbol}_$low"] = [p * (1 - np.random.uniform(0, 0.01)) for p in prices]
            data[f"{symbol}_$close"] = prices
            data[f"{symbol}_$volume"] = [p * np.random.uniform(100, 1000) for p in prices]
    
    return data


def compare_backtester_results(base_results, enhanced_results, tolerance=0.05):
    """
    Compare results from BaseBacktester and EnhancedBacktester to ensure compatibility.
    
    Args:
        base_results: Results from BaseBacktester
        enhanced_results: Results from EnhancedBacktester
        tolerance: Acceptable difference threshold (as a percentage)
    
    Returns:
        Dict with comparison results
    """
    comparison = {}
    
    # Compare metrics
    base_metrics = base_results.get('metrics', {})
    enhanced_metrics = enhanced_results.get('metrics', {})
    
    metrics_comparison = {}
    for key in base_metrics:
        if key in enhanced_metrics:
            base_value = base_metrics[key]
            enhanced_value = enhanced_metrics[key]
            
            # Skip comparison if both values are zero
            if abs(base_value) < 1e-6 and abs(enhanced_value) < 1e-6:
                metrics_comparison[key] = {
                    'status': 'equal',
                    'difference': 0,
                    'percent_diff': 0
                }
                continue
                
            # Calculate difference
            abs_diff = abs(base_value - enhanced_value)
            if abs(base_value) > 1e-6:
                percent_diff = abs_diff / abs(base_value)
            else:
                percent_diff = float('inf') if abs_diff > 0 else 0
                
            # Check if difference is within tolerance
            if percent_diff <= tolerance:
                status = 'similar'
            else:
                status = 'different'
                
            metrics_comparison[key] = {
                'base_value': base_value,
                'enhanced_value': enhanced_value,
                'difference': abs_diff,
                'percent_diff': percent_diff,
                'status': status
            }
    
    # Compare portfolio value histories
    base_portfolio = base_results.get('portfolio_values', [])
    enhanced_portfolio = enhanced_results.get('portfolio_values', [])
    
    portfolio_length_match = len(base_portfolio) == len(enhanced_portfolio)
    
    # Calculate final portfolio difference
    if base_portfolio and enhanced_portfolio:
        final_base = base_portfolio[-1]
        final_enhanced = enhanced_portfolio[-1]
        
        final_diff = abs(final_base - final_enhanced)
        if abs(final_base) > 1e-6:
            final_percent_diff = final_diff / abs(final_base)
        else:
            final_percent_diff = float('inf') if final_diff > 0 else 0
            
        portfolio_comparison = {
            'length_match': portfolio_length_match,
            'final_base_value': final_base,
            'final_enhanced_value': final_enhanced,
            'final_difference': final_diff,
            'final_percent_diff': final_percent_diff,
            'within_tolerance': final_percent_diff <= tolerance
        }
    else:
        portfolio_comparison = {
            'length_match': portfolio_length_match,
            'error': 'Portfolio history missing'
        }
    
    # Compare trade counts
    base_trades = base_results.get('trades', [])
    enhanced_trades = enhanced_results.get('trades', [])
    
    trade_comparison = {
        'base_trade_count': len(base_trades),
        'enhanced_trade_count': len(enhanced_trades),
        'count_match': len(base_trades) == len(enhanced_trades)
    }
    
    comparison['metrics'] = metrics_comparison
    comparison['portfolio'] = portfolio_comparison
    comparison['trades'] = trade_comparison
    
    # Overall compatibility assessment
    metrics_compatible = all(item['status'] in ['equal', 'similar'] for item in metrics_comparison.values())
    portfolio_compatible = portfolio_comparison.get('within_tolerance', False)
    
    comparison['overall_compatible'] = metrics_compatible and portfolio_compatible
    
    return comparison


def test_basic_functionality():
    """Test basic functionality of both backtester implementations."""
    logger.info("Testing basic functionality...")
    
    # Generate test data
    data = generate_test_data(days=100)
    
    # Initialize strategy
    strategy = SimpleStrategy(window_size=5, threshold=0.01)
    
    # Initialize Base Backtester
    base_backtester = BaseBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy()
    )
    
    # Initialize Enhanced Backtester with identical settings (no advanced features yet)
    enhanced_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy(),
        slippage_model="fixed",  # No slippage
        partial_fill=False       # No partial fills
    )
    
    # Run backtest with both implementations
    logger.info("Running BaseBacktester...")
    base_results = base_backtester.run(strategy, window_size=10, verbose=True)
    
    logger.info("Running EnhancedBacktester...")
    enhanced_results = enhanced_backtester.run(strategy, window_size=10, verbose=True)
    
    # Compare results
    comparison = compare_backtester_results(base_results, enhanced_results)
    
    # Log comparison results
    if comparison['overall_compatible']:
        logger.info("✅ Basic functionality test PASSED - Results are compatible")
    else:
        logger.warning("❌ Basic functionality test FAILED - Results are not compatible")
        
    for key, metric in comparison['metrics'].items():
        if metric['status'] == 'different':
            logger.warning(f"Metric '{key}' differs significantly: {metric['base_value']} vs {metric['enhanced_value']} ({metric['percent_diff']*100:.2f}% difference)")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(base_backtester.portfolio_history, label='BaseBacktester', alpha=0.7)
    ax.plot(enhanced_backtester.portfolio_history, label='EnhancedBacktester', alpha=0.7)
    ax.set_title('Portfolio Value Comparison')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = TEST_OUTPUT_DIR / "basic_comparison.png"
    plt.savefig(plot_path)
    logger.info(f"Saved comparison plot to {plot_path}")
    
    return comparison


def test_slippage_functionality():
    """Test slippage functionality of EnhancedBacktester."""
    logger.info("Testing slippage functionality...")
    
    # Generate test data
    data = generate_test_data(days=100)
    
    # Initialize strategy
    strategy = SimpleStrategy(window_size=5, threshold=0.01)
    
    # Initialize Enhanced Backtester with no slippage
    no_slippage_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy(),
        slippage_model="fixed",
        partial_fill=False
    )
    
    # Initialize Enhanced Backtester with volume-based slippage
    volume_slippage_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy(),
        slippage_model="volume",
        partial_fill=False,
        market_impact_factor=0.2  # Higher market impact for clearer effect
    )
    
    # Run backtest with both configurations
    logger.info("Running without slippage...")
    no_slippage_results = no_slippage_backtester.run(strategy, window_size=10, verbose=True)
    
    logger.info("Running with volume-based slippage...")
    volume_slippage_results = volume_slippage_backtester.run(strategy, window_size=10, verbose=True)
    
    # Compare final portfolio values
    no_slippage_final = no_slippage_backtester.portfolio_history[-1]
    volume_slippage_final = volume_slippage_backtester.portfolio_history[-1]
    
    # Calculate difference
    diff = no_slippage_final - volume_slippage_final
    percent_diff = diff / no_slippage_final if abs(no_slippage_final) > 1e-6 else 0
    
    logger.info(f"No slippage final portfolio: ${no_slippage_final:.2f}")
    logger.info(f"Volume slippage final portfolio: ${volume_slippage_final:.2f}")
    logger.info(f"Difference: ${diff:.2f} ({percent_diff*100:.2f}%)")
    
    # Verify that slippage had an impact
    slippage_impact = percent_diff > 0.01  # Expect at least 1% difference
    
    if slippage_impact:
        logger.info("✅ Slippage functionality test PASSED - Slippage had expected impact")
    else:
        logger.warning("❌ Slippage functionality test FAILED - Slippage had minimal impact")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(no_slippage_backtester.portfolio_history, label='No Slippage', alpha=0.7)
    ax.plot(volume_slippage_backtester.portfolio_history, label='Volume Slippage', alpha=0.7)
    ax.set_title('Slippage Impact on Portfolio Value')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = TEST_OUTPUT_DIR / "slippage_comparison.png"
    plt.savefig(plot_path)
    logger.info(f"Saved slippage comparison plot to {plot_path}")
    
    # Plot slippage history if available
    if volume_slippage_backtester.slippage_history:
        slippage_values = [entry['slippage_percentage'] for entry in volume_slippage_backtester.slippage_history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(slippage_values)
        ax.set_title('Slippage Percentage Over Time')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Slippage (%)')
        ax.grid(True)
        
        # Save plot
        plot_path = TEST_OUTPUT_DIR / "slippage_history.png"
        plt.savefig(plot_path)
        logger.info(f"Saved slippage history plot to {plot_path}")
    
    return {
        'no_slippage_final': no_slippage_final,
        'volume_slippage_final': volume_slippage_final,
        'difference': diff,
        'percent_diff': percent_diff,
        'slippage_impact': slippage_impact
    }


def test_partial_fill_functionality():
    """Test partial fill functionality of EnhancedBacktester."""
    logger.info("Testing partial fill functionality...")
    
    # Generate test data
    data = generate_test_data(days=100)
    
    # Initialize strategy
    strategy = SimpleStrategy(window_size=5, threshold=0.01)
    
    # Initialize Enhanced Backtester with full fills
    full_fill_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy(),
        slippage_model="fixed",
        partial_fill=False
    )
    
    # Initialize Enhanced Backtester with partial fills
    partial_fill_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy(),
        slippage_model="fixed",
        partial_fill=True,
        market_impact_factor=0.1
    )
    
    # Run backtest with both configurations
    logger.info("Running with full fills...")
    full_fill_results = full_fill_backtester.run(strategy, window_size=10, verbose=True)
    
    logger.info("Running with partial fills...")
    partial_fill_results = partial_fill_backtester.run(strategy, window_size=10, verbose=True)
    
    # Compare trade counts and execution
    full_fill_trades = full_fill_results.get('trades', [])
    partial_fill_trades = partial_fill_results.get('trades', [])
    
    # Calculate average fill rate if available
    fill_rates = []
    if partial_fill_backtester.fill_rate_history:
        fill_rates = [entry['fill_percentage'] for entry in partial_fill_backtester.fill_rate_history]
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 1.0
        logger.info(f"Average fill rate: {avg_fill_rate:.2%}")
    
    # Compare final portfolio values
    full_fill_final = full_fill_backtester.portfolio_history[-1]
    partial_fill_final = partial_fill_backtester.portfolio_history[-1]
    
    # Calculate difference
    diff = full_fill_final - partial_fill_final
    percent_diff = diff / full_fill_final if abs(full_fill_final) > 1e-6 else 0
    
    logger.info(f"Full fill final portfolio: ${full_fill_final:.2f}")
    logger.info(f"Partial fill final portfolio: ${partial_fill_final:.2f}")
    logger.info(f"Difference: ${diff:.2f} ({percent_diff*100:.2f}%)")
    
    # Verify that partial fills had an impact
    partial_fill_impact = len(fill_rates) > 0 and any(rate < 0.99 for rate in fill_rates)
    
    if partial_fill_impact:
        logger.info("✅ Partial fill functionality test PASSED - Partial fills occurred")
    else:
        logger.warning("❌ Partial fill functionality test FAILED - No partial fills detected")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(full_fill_backtester.portfolio_history, label='Full Fills', alpha=0.7)
    ax.plot(partial_fill_backtester.portfolio_history, label='Partial Fills', alpha=0.7)
    ax.set_title('Partial Fill Impact on Portfolio Value')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = TEST_OUTPUT_DIR / "partial_fill_comparison.png"
    plt.savefig(plot_path)
    logger.info(f"Saved partial fill comparison plot to {plot_path}")
    
    # Plot fill rate history if available
    if fill_rates:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(fill_rates)
        ax.set_title('Fill Rate Over Time')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Fill Rate')
        ax.grid(True)
        
        # Save plot
        plot_path = TEST_OUTPUT_DIR / "fill_rate_history.png"
        plt.savefig(plot_path)
        logger.info(f"Saved fill rate history plot to {plot_path}")
    
    return {
        'full_fill_final': full_fill_final,
        'partial_fill_final': partial_fill_final,
        'difference': diff,
        'percent_diff': percent_diff,
        'partial_fill_impact': partial_fill_impact,
        'avg_fill_rate': np.mean(fill_rates) if fill_rates else 1.0
    }


def test_multi_asset_functionality():
    """Test multi-asset functionality of EnhancedBacktester."""
    logger.info("Testing multi-asset functionality...")
    
    # Generate multi-asset test data
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    data = generate_test_data(days=100, symbols=symbols, with_correlation=True)
    
    # Initialize strategy
    strategy = SimpleStrategy(window_size=5, threshold=0.01)
    
    # Initialize Enhanced Backtester with multi-asset data
    multi_asset_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy()
    )
    
    # Run multi-asset backtest
    logger.info("Running multi-asset backtest...")
    multi_asset_results = multi_asset_backtester.run_multi_asset(strategy, window_size=10, verbose=True)
    
    # Verify results
    trades = multi_asset_results.get('trades', [])
    
    # Check if trades were executed for different symbols
    traded_symbols = set(trade['symbol'] for trade in trades if trade['success'])
    
    logger.info(f"Detected trades for symbols: {traded_symbols}")
    logger.info(f"Total trades executed: {len(trades)}")
    logger.info(f"Successful trades: {sum(1 for trade in trades if trade['success'])}")
    
    # Verify that trades were executed for multiple assets
    multi_asset_trading = len(traded_symbols) > 1
    
    if multi_asset_trading:
        logger.info("✅ Multi-asset functionality test PASSED - Traded multiple assets")
    else:
        logger.warning("❌ Multi-asset functionality test FAILED - Did not trade multiple assets")
    
    # Generate portfolio composition data
    final_positions = {}
    for symbol in traded_symbols:
        if symbol in multi_asset_backtester.positions:
            position = multi_asset_backtester.positions[symbol]
            final_positions[symbol] = position
    
    logger.info(f"Final positions: {final_positions}")
    
    # Plot portfolio value
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(multi_asset_backtester.portfolio_history)
    ax.set_title('Multi-Asset Portfolio Value')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True)
    
    # Save plot
    plot_path = TEST_OUTPUT_DIR / "multi_asset_portfolio.png"
    plt.savefig(plot_path)
    logger.info(f"Saved multi-asset portfolio plot to {plot_path}")
    
    return {
        'traded_symbols': list(traded_symbols),
        'total_trades': len(trades),
        'successful_trades': sum(1 for trade in trades if trade['success']),
        'multi_asset_trading': multi_asset_trading,
        'final_positions': final_positions,
        'final_portfolio_value': multi_asset_backtester.portfolio_history[-1] if multi_asset_backtester.portfolio_history else None
    }


def test_scenario_functionality():
    """Test scenario functionality of EnhancedBacktester."""
    logger.info("Testing scenario functionality...")
    
    # Generate test data
    data = generate_test_data(days=100)
    
    # Initialize strategy
    strategy = SimpleStrategy(window_size=5, threshold=0.01)
    
    # Initialize Enhanced Backtester
    scenario_backtester = EnhancedBacktester(
        initial_capital=10000.0,
        trading_fee=0.001,
        max_position=1.0,
        data=data.copy()
    )
    
    # Run different scenarios
    scenarios = ["normal", "high_volatility", "low_liquidity", "flash_crash", "perfect_execution"]
    scenario_results = {}
    portfolio_histories = {}
    trade_histories = {}
    
    for scenario_type in scenarios:
        logger.info(f"Running {scenario_type} scenario...")
        results = scenario_backtester.run_scenario_with_slippage(
            strategy=strategy,
            scenario_type=scenario_type,
            window_size=10,
            verbose=True
        )
        
        # Store complete results for analysis
        scenario_results[scenario_type] = {
            'final_value': scenario_backtester.portfolio_history[-1] if scenario_backtester.portfolio_history else None,
            'metrics': results.get('metrics', {}),
            'trades': len(scenario_backtester.trades),
            'successful_trades': sum(1 for trade in scenario_backtester.trades if trade['success']),
            'partial_fills': sum(1 for trade in scenario_backtester.trades if 'executed_amount' in trade and trade['executed_amount'] < trade['amount'] and trade['success']),
            'fill_rate_avg': np.mean([trade.get('executed_amount', trade['amount']) / trade['amount'] for trade in scenario_backtester.trades if trade['amount'] > 0]) if scenario_backtester.trades else 1.0,
            'slippage_avg': np.mean([trade.get('slippage_percentage', 0.0) for trade in scenario_backtester.trades]) if scenario_backtester.trades else 0.0,
            'settings': results.get('scenario_settings', {})
        }
        
        # Store portfolio history for plotting
        portfolio_histories[scenario_type] = scenario_backtester.portfolio_history.copy() if scenario_backtester.portfolio_history else []
        
        # Store trade history
        trade_histories[scenario_type] = scenario_backtester.trades.copy() if scenario_backtester.trades else []
    
    # Compare scenarios with detailed metrics
    logger.info("\n===== Scenario Comparison: =====")
    
    # Create a table for comparison
    metrics_table = []
    for scenario_type, results in scenario_results.items():
        metrics_table.append({
            'Scenario': scenario_type,
            'Final Value': f"${results['final_value']:.2f}",
            'Return %': f"{((results['final_value'] / 10000.0) - 1) * 100:.2f}%",
            'Trades': results['trades'],
            'Success %': f"{(results['successful_trades'] / results['trades'] * 100) if results['trades'] > 0 else 0:.1f}%",
            'Fill Rate': f"{results['fill_rate_avg']:.1%}",
            'Avg Slippage': f"{results['slippage_avg']:.3%}"
        })
    
    # Print table
    if metrics_table:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics_table)
        logger.info(f"\n{metrics_df.to_string(index=False)}")
    
    # Check if scenarios produced different results
    final_values = [results['final_value'] for results in scenario_results.values() if results['final_value'] is not None]
    scenario_variance = np.var(final_values) / np.mean(final_values) if np.mean(final_values) > 0 else 0
    
    logger.info(f"\nScenario variance (portfolio values): {scenario_variance:.6f}")
    
    # Calculate fill rate variance
    fill_rates = [results['fill_rate_avg'] for results in scenario_results.values()]
    fill_rate_variance = np.var(fill_rates) if fill_rates else 0
    logger.info(f"Fill rate variance: {fill_rate_variance:.6f}")
    
    # Calculate slippage variance
    slippages = [results['slippage_avg'] for results in scenario_results.values()]
    slippage_variance = np.var(slippages) if slippages else 0
    logger.info(f"Slippage variance: {slippage_variance:.6f}")
    
    # Verify that scenarios produced different results (at least one of the metrics should show variance)
    scenarios_differentiated = (
        scenario_variance > 0.001 or   # Portfolio value variance threshold increased
        fill_rate_variance > 0.001 or  # Fill rate variance threshold
        slippage_variance > 0.0001     # Slippage variance threshold
    )
    
    if scenarios_differentiated:
        logger.info("✅ Scenario functionality test PASSED - Scenarios produced different results")
    else:
        logger.warning("❌ Scenario functionality test FAILED - Scenarios produced similar results")
        
        # Log detailed information to help debug why scenarios are similar
        for scenario_type, results in scenario_results.items():
            logger.info(f"\nScenario {scenario_type} settings:")
            for key, value in results.get('settings', {}).items():
                logger.info(f"  {key}: {value}")
    
    # Plot scenario comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Portfolio value comparison
    for scenario_type, history in portfolio_histories.items():
        if history:
            axes[0].plot(history, label=scenario_type, alpha=0.7)
    
    axes[0].set_title('Portfolio Value by Scenario')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Slippage and fill rate analysis
    scenario_labels = []
    fill_rates = []
    slippages = []
    
    for scenario_type, results in scenario_results.items():
        scenario_labels.append(scenario_type)
        fill_rates.append(results['fill_rate_avg'] * 100)  # Convert to percentage
        slippages.append(abs(results['slippage_avg'] * 100))  # Convert to percentage and take absolute value
    
    x = np.arange(len(scenario_labels))
    width = 0.35
    
    axes[1].bar(x - width/2, fill_rates, width, label='Fill Rate %')
    axes[1].bar(x + width/2, slippages, width, label='Abs Slippage %')
    
    axes[1].set_title('Fill Rates and Slippage by Scenario')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenario_labels)
    axes[1].set_ylabel('Percentage')
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = TEST_OUTPUT_DIR / "scenario_comparison.png"
    plt.savefig(plot_path)
    logger.info(f"Saved scenario comparison plot to {plot_path}")
    
    # Save detailed trade data for each scenario
    for scenario_type, trades in trade_histories.items():
        if trades:
            # Convert to DataFrame for easier analysis
            trades_df = pd.DataFrame(trades)
            
            # Save to CSV
            trades_file = TEST_OUTPUT_DIR / f"trades_{scenario_type}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved {scenario_type} trades to {trades_file}")
    
    return {
        'scenario_results': scenario_results,
        'scenario_variance': scenario_variance,
        'fill_rate_variance': fill_rate_variance,
        'slippage_variance': slippage_variance,
        'scenarios_differentiated': scenarios_differentiated
    }


def run_all_tests():
    """Run all tests and report results."""
    start_time = datetime.now()
    logger.info(f"Starting EnhancedBacktester tests at {start_time}...")
    
    # Initialize results dictionary
    test_results = {}
    
    # Run tests
    test_results['basic_functionality'] = test_basic_functionality()
    test_results['slippage_functionality'] = test_slippage_functionality()
    test_results['partial_fill_functionality'] = test_partial_fill_functionality()
    test_results['multi_asset_functionality'] = test_multi_asset_functionality()
    test_results['scenario_functionality'] = test_scenario_functionality()
    
    # Count successful tests
    success_count = sum([
        test_results['basic_functionality']['overall_compatible'],
        test_results['slippage_functionality']['slippage_impact'],
        test_results['partial_fill_functionality']['partial_fill_impact'],
        test_results['multi_asset_functionality']['multi_asset_trading'],
        test_results['scenario_functionality']['scenarios_differentiated']
    ])
    
    # Report summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"All tests completed in {duration:.2f} seconds")
    logger.info(f"Test success rate: {success_count}/5 ({success_count/5*100:.0f}%)")
    
    if success_count == 5:
        logger.info("🎉 All tests PASSED - EnhancedBacktester is working correctly")
    else:
        logger.warning("⚠️ Some tests FAILED - EnhancedBacktester may have issues")
    
    return test_results


if __name__ == "__main__":
    run_all_tests() 