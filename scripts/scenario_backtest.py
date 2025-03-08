#!/usr/bin/env python
"""
Scenario Backtesting System for Multi-Asset Trading Environment

This script provides functionality to run backtests across various market scenarios
and risk management configurations. It allows for systematic evaluation of risk
management strategies under different market conditions.

Features:
- Pre-defined market scenarios (bull, bear, crisis, recovery)
- Custom scenario generation with controlled parameters
- Multiple risk management configurations
- Comprehensive backtest results and analysis
- Visualization of portfolio performance across scenarios
- Risk metrics comparison

Implementation Notes:
- Uses synthetic data generation with controlled correlation structures
- Integrates with RiskManager and EnhancedBacktester
- Runs multiple backtests in parallel for efficiency
- Saves results for further analysis and visualization

Recent Changes:
- Added market crash scenario simulation
- Implemented portfolio diversification analysis
- Enhanced visualization with risk event markers
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
from datetime import datetime
import argparse
import multiprocessing as mp
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from envs.risk_manager import RiskManager, RiskConfig
from training.backtesting.enhanced_backtester import EnhancedBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f"logs/scenario_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('scenario_backtest')

# Default paths
DEFAULT_RESULTS_DIR = Path("backtest_results/scenario_tests")
DEFAULT_CONFIG_PATH = Path("configs/risk_management.yaml")

def generate_scenario_data(scenario_params: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price data for a specific market scenario.
    
    Args:
        scenario_params: Dictionary with scenario parameters
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLCV data for all assets in the scenario
    """
    np.random.seed(seed)
    
    # Extract scenario parameters
    assets = list(scenario_params["trend_factors"].keys())
    days = scenario_params.get("duration_days", 180)
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Base parameters for different asset types
    asset_params = {
        # Format: (starting_price, volume_mean)
        "BTC": (20000, 1000),
        "ETH": (1500, 5000),
        "SPY": (400, 10000),
        "AAPL": (150, 20000),
        "GOLD": (1800, 5000),
        "BONDS": (100, 50000),
    }
    
    # Default parameters for assets not in the dict
    default_params = (100, 1000)
    
    # Create correlation matrix from scenario params
    corr_matrix = np.eye(len(assets))
    
    # If correlation_matrix is provided in scenario, use it
    if "correlation_matrix" in scenario_params:
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j:  # Skip diagonal (self-correlation)
                    corr_matrix[i, j] = scenario_params["correlation_matrix"].get(asset1, {}).get(asset2, 0.0)
    
    # Generate correlated returns
    # First, create uncorrelated normal returns
    uncorrelated_returns = np.random.normal(
        size=(days, len(assets)),
        loc=[scenario_params["trend_factors"].get(asset, 0.0) for asset in assets],
        scale=[scenario_params["volatility_factors"].get(asset, 0.01) for asset in assets]
    )
    
    # Apply Cholesky decomposition to get correlated returns
    try:
        L = np.linalg.cholesky(corr_matrix)
        correlated_returns = uncorrelated_returns @ L.T
    except np.linalg.LinAlgError:
        # If correlation matrix is not positive definite, adjust it
        logger.warning("Correlation matrix is not positive definite. Using nearest PD matrix.")
        corr_matrix = nearest_positive_definite(corr_matrix)
        L = np.linalg.cholesky(corr_matrix)
        correlated_returns = uncorrelated_returns @ L.T
    
    # Generate price data
    data = {'date': dates}
    
    for i, asset in enumerate(assets):
        # Get parameters for this asset
        starting_price, volume_mean = asset_params.get(asset, default_params)
        
        # Apply initial drawdown if specified (usually for recovery scenario)
        if scenario_params.get("start_with_drawdown", False) and "initial_drawdown_pct" in scenario_params:
            starting_price = starting_price * (1 + scenario_params["initial_drawdown_pct"].get(asset, 0.0))
        
        # Cumulative returns to create price series
        cum_returns = np.cumprod(1 + correlated_returns[:, i])
        prices = starting_price * cum_returns
        
        # Apply flash crash if specified
        if scenario_params.get("include_flash_crash", False) and "flash_crash_day" in scenario_params:
            crash_day = scenario_params["flash_crash_day"]
            if crash_day < days:
                crash_pct = scenario_params["flash_crash_pct"].get(asset, 0.0)
                # Adjust the price after the crash day
                prices[crash_day:] = prices[crash_day:] * (1 + crash_pct)
        
        # Create OHLCV data
        data[f'{asset}_$open'] = prices * (1 - np.random.uniform(0, 0.005, days))
        data[f'{asset}_$high'] = prices * (1 + np.random.uniform(0.005, 0.015, days))
        data[f'{asset}_$low'] = prices * (1 - np.random.uniform(0.005, 0.015, days))
        data[f'{asset}_$close'] = prices
        
        # Volume - higher on volatile days
        vol_factor = np.abs(correlated_returns[:, i]) * 5 + 1  # 1 to 6x multiplier based on returns
        data[f'{asset}_$volume'] = np.random.normal(volume_mean, volume_mean * 0.2, days) * vol_factor
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure high/low prices are consistent
    for asset in assets:
        df[f'{asset}_$high'] = np.maximum(df[f'{asset}_$high'], np.maximum(df[f'{asset}_$open'], df[f'{asset}_$close']))
        df[f'{asset}_$low'] = np.minimum(df[f'{asset}_$low'], np.minimum(df[f'{asset}_$open'], df[f'{asset}_$close']))
    
    return df

def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to A."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    
    if is_positive_definite(A3):
        return A3
    
    # If still not positive definite, add small diagonal elements
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    
    return A3

def is_positive_definite(A):
    """Check if matrix A is positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def load_and_modify_risk_config(config_path: Union[str, Path], modifications: Dict[str, Any]) -> RiskConfig:
    """
    Load risk configuration from file and apply modifications.
    
    Args:
        config_path: Path to the risk configuration file
        modifications: Dictionary of modifications to apply
        
    Returns:
        Modified RiskConfig object
    """
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default config.")
        config_dict = {}
    
    # Apply modifications
    for section, params in modifications.items():
        if section not in config_dict:
            config_dict[section] = {}
        
        config_dict[section].update(params)
    
    # Create RiskConfig with the merged settings
    risk_config = RiskConfig(
        # Stop loss
        use_stop_loss=config_dict.get("stop_loss", {}).get("use_stop_loss", True),
        stop_loss_threshold=config_dict.get("stop_loss", {}).get("stop_loss_threshold", 0.1),
        
        # Trailing stop
        use_trailing_stop=config_dict.get("trailing_stop", {}).get("use_trailing_stop", True),
        trailing_stop_buffer=config_dict.get("trailing_stop", {}).get("trailing_stop_buffer", 0.05),
        
        # VaR
        use_var=config_dict.get("var", {}).get("use_var", True),
        var_confidence_level=config_dict.get("var", {}).get("var_confidence_level", 0.95),
        
        # Correlation
        use_correlation=config_dict.get("correlation", {}).get("use_correlation", True),
        correlation_threshold=config_dict.get("correlation", {}).get("correlation_threshold", 0.7),
        correlation_risk_reduction=config_dict.get("correlation", {}).get("correlation_risk_reduction", 0.5),
        
        # Portfolio stop loss
        use_portfolio_stop_loss=config_dict.get("portfolio_stop_loss", {}).get("use_portfolio_stop_loss", True),
        portfolio_stop_loss_threshold=config_dict.get("portfolio_stop_loss", {}).get("portfolio_stop_loss_threshold", 0.15),
        
        # Portfolio trailing stop
        use_portfolio_trailing_stop=config_dict.get("portfolio_trailing_stop", {}).get("use_portfolio_trailing_stop", True),
        portfolio_trailing_stop_buffer=config_dict.get("portfolio_trailing_stop", {}).get("portfolio_trailing_stop_buffer", 0.08),
        
        # Portfolio VaR
        use_portfolio_var=config_dict.get("portfolio_var", {}).get("use_portfolio_var", True),
        portfolio_var_threshold=config_dict.get("portfolio_var", {}).get("portfolio_var_threshold", 0.02),
    )
    
    return risk_config

def backtest_scenario(
    scenario_name: str,
    scenario_params: Dict[str, Any],
    risk_config_name: str, 
    risk_params: Dict[str, Any],
    config_path: Union[str, Path],
    results_dir: Union[str, Path],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run backtest for a specific scenario and risk configuration.
    
    Args:
        scenario_name: Name of the scenario
        scenario_params: Parameters for the scenario
        risk_config_name: Name of the risk configuration
        risk_params: Risk management parameters
        config_path: Path to the base risk configuration file
        results_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Running backtest for {scenario_name} with {risk_config_name} risk config")
    
    # Ensure results directory exists
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate scenario data
    data = generate_scenario_data(scenario_params, seed=seed)
    
    # Get assets
    assets = list(scenario_params["trend_factors"].keys())
    
    # Load and modify risk config
    risk_config = load_and_modify_risk_config(config_path, risk_params)
    risk_manager = RiskManager(risk_config)
    
    # Create backtester
    backtester = EnhancedBacktester(
        data=data,
        assets=assets,
        risk_manager=risk_manager,
        trading_fee=0.001,
        slippage_model="proportional",
        slippage_rate=0.0005,
        initial_balance=10000.0,
        window_size=10,
        portfolio_manager_config={
            "rebalance_threshold": 0.05,
            "max_leverage": 1.0
        }
    )
    
    # Run backtest
    start_time = datetime.now()
    backtest_results = backtester.run()
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Add scenario and risk config info to results
    backtest_results.update({
        "scenario": scenario_name,
        "risk_config": risk_config_name,
        "execution_time": execution_time
    })
    
    # Plot and save results
    fig = backtester.plot_results()
    fig_path = results_dir / f"{scenario_name}_{risk_config_name}_results.png"
    fig.savefig(fig_path)
    plt.close(fig)
    
    logger.info(f"Completed backtest: {scenario_name} with {risk_config_name}")
    logger.info(f"Final portfolio value: ${backtest_results['final_portfolio_value']:.2f}")
    logger.info(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.4f}")
    logger.info(f"Max drawdown: {backtest_results['max_drawdown']:.2%}")
    
    return backtest_results

def run_market_scenario_backtest(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
    results_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    risk_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    parallel: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run backtest across multiple market scenarios and risk configurations.
    
    Args:
        config_path: Path to the base risk configuration file
        results_dir: Directory to save results
        scenarios: Dictionary of scenario configurations (or None for defaults)
        risk_configs: Dictionary of risk configurations (or None for defaults)
        parallel: Whether to run backtests in parallel
        seed: Random seed
        
    Returns:
        DataFrame with all backtest results
    """
    # Define default scenarios if not provided
    if scenarios is None:
        scenarios = {
            "bull_market": {
                "trend_factors": {"BTC": 0.002, "ETH": 0.0025, "SPY": 0.001, "GOLD": 0.0005},
                "volatility_factors": {"BTC": 0.02, "ETH": 0.025, "SPY": 0.01, "GOLD": 0.008},
                "correlation_matrix": {
                    "BTC": {"BTC": 1.0, "ETH": 0.8, "SPY": 0.4, "GOLD": 0.1},
                    "ETH": {"BTC": 0.8, "ETH": 1.0, "SPY": 0.35, "GOLD": 0.05},
                    "SPY": {"BTC": 0.4, "ETH": 0.35, "SPY": 1.0, "GOLD": 0.0},
                    "GOLD": {"BTC": 0.1, "ETH": 0.05, "SPY": 0.0, "GOLD": 1.0}
                },
                "duration_days": 180
            },
            "bear_market": {
                "trend_factors": {"BTC": -0.0015, "ETH": -0.002, "SPY": -0.001, "GOLD": 0.0003},
                "volatility_factors": {"BTC": 0.03, "ETH": 0.035, "SPY": 0.018, "GOLD": 0.01},
                "correlation_matrix": {
                    "BTC": {"BTC": 1.0, "ETH": 0.85, "SPY": 0.6, "GOLD": -0.3},
                    "ETH": {"BTC": 0.85, "ETH": 1.0, "SPY": 0.55, "GOLD": -0.25},
                    "SPY": {"BTC": 0.6, "ETH": 0.55, "SPY": 1.0, "GOLD": -0.4},
                    "GOLD": {"BTC": -0.3, "ETH": -0.25, "SPY": -0.4, "GOLD": 1.0}
                },
                "duration_days": 180
            },
            "crisis_market": {
                "trend_factors": {"BTC": -0.005, "ETH": -0.006, "SPY": -0.004, "GOLD": -0.001},
                "volatility_factors": {"BTC": 0.05, "ETH": 0.055, "SPY": 0.04, "GOLD": 0.02},
                "correlation_matrix": {
                    "BTC": {"BTC": 1.0, "ETH": 0.9, "SPY": 0.8, "GOLD": 0.5},
                    "ETH": {"BTC": 0.9, "ETH": 1.0, "SPY": 0.75, "GOLD": 0.45},
                    "SPY": {"BTC": 0.8, "ETH": 0.75, "SPY": 1.0, "GOLD": 0.4},
                    "GOLD": {"BTC": 0.5, "ETH": 0.45, "SPY": 0.4, "GOLD": 1.0}
                },
                "duration_days": 90,
                "include_flash_crash": True,
                "flash_crash_day": 15,
                "flash_crash_pct": {"BTC": -0.3, "ETH": -0.35, "SPY": -0.15, "GOLD": -0.05}
            },
            "recovery_market": {
                "trend_factors": {"BTC": 0.003, "ETH": 0.0035, "SPY": 0.002, "GOLD": 0.0},
                "volatility_factors": {"BTC": 0.025, "ETH": 0.03, "SPY": 0.015, "GOLD": 0.01},
                "correlation_matrix": {
                    "BTC": {"BTC": 1.0, "ETH": 0.75, "SPY": 0.5, "GOLD": -0.2},
                    "ETH": {"BTC": 0.75, "ETH": 1.0, "SPY": 0.45, "GOLD": -0.15},
                    "SPY": {"BTC": 0.5, "ETH": 0.45, "SPY": 1.0, "GOLD": -0.1},
                    "GOLD": {"BTC": -0.2, "ETH": -0.15, "SPY": -0.1, "GOLD": 1.0}
                },
                "duration_days": 180,
                "start_with_drawdown": True,
                "initial_drawdown_pct": {"BTC": -0.25, "ETH": -0.3, "SPY": -0.12, "GOLD": -0.02}
            }
        }
    
    # Define default risk configurations if not provided
    if risk_configs is None:
        risk_configs = {
            "conservative": {
                "stop_loss": {"stop_loss_threshold": 0.05},
                "trailing_stop": {"trailing_stop_buffer": 0.03},
                "portfolio_var": {"portfolio_var_threshold": 0.015},
                "correlation": {"correlation_threshold": 0.6, "correlation_risk_reduction": 0.4}
            },
            "moderate": {
                "stop_loss": {"stop_loss_threshold": 0.1},
                "trailing_stop": {"trailing_stop_buffer": 0.05},
                "portfolio_var": {"portfolio_var_threshold": 0.02},
                "correlation": {"correlation_threshold": 0.7, "correlation_risk_reduction": 0.5}
            },
            "aggressive": {
                "stop_loss": {"stop_loss_threshold": 0.15},
                "trailing_stop": {"trailing_stop_buffer": 0.08},
                "portfolio_var": {"portfolio_var_threshold": 0.03},
                "correlation": {"correlation_threshold": 0.8, "correlation_risk_reduction": 0.6}
            },
            "no_risk_management": {
                "stop_loss": {"use_stop_loss": False},
                "trailing_stop": {"use_trailing_stop": False},
                "var": {"use_var": False},
                "correlation": {"use_correlation": False},
                "portfolio_stop_loss": {"use_portfolio_stop_loss": False},
                "portfolio_trailing_stop": {"use_portfolio_trailing_stop": False},
                "portfolio_var": {"use_portfolio_var": False}
            }
        }
    
    # Prepare the backtest tasks
    backtest_tasks = []
    for scenario_name, scenario_params in scenarios.items():
        for risk_config_name, risk_params in risk_configs.items():
            backtest_tasks.append((
                scenario_name,
                scenario_params,
                risk_config_name,
                risk_params,
                config_path,
                results_dir,
                seed
            ))
    
    # Run backtests (in parallel or sequentially)
    results = []
    if parallel and len(backtest_tasks) > 1:
        logger.info(f"Running {len(backtest_tasks)} backtests in parallel")
        with mp.Pool(processes=min(mp.cpu_count(), len(backtest_tasks))) as pool:
            results = pool.starmap(backtest_scenario, backtest_tasks)
    else:
        logger.info(f"Running {len(backtest_tasks)} backtests sequentially")
        for task in backtest_tasks:
            results.append(backtest_scenario(*task))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv_path = Path(results_dir) / "scenario_backtest_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved results to {results_csv_path}")
    
    # Generate summary and comparison visualizations
    generate_summary_visualizations(results_df, results_dir)
    
    return results_df

def generate_summary_visualizations(results_df: pd.DataFrame, results_dir: Union[str, Path]):
    """
    Generate summary visualizations from backtest results.
    
    Args:
        results_df: DataFrame with backtest results
        results_dir: Directory to save visualizations
    """
    results_dir = Path(results_dir)
    
    # 1. Final Portfolio Value Comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='scenario', y='final_portfolio_value', hue='risk_config', data=results_df)
    plt.title('Final Portfolio Value by Scenario and Risk Configuration')
    plt.ylabel('Final Portfolio Value ($)')
    plt.xlabel('Market Scenario')
    plt.xticks(rotation=45)
    plt.legend(title='Risk Configuration')
    plt.tight_layout()
    plt.savefig(results_dir / "portfolio_value_comparison.png")
    plt.close()
    
    # 2. Risk-Return Scatterplot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x='max_drawdown', 
        y='sharpe_ratio', 
        hue='risk_config', 
        style='scenario',
        s=100, 
        data=results_df
    )
    plt.title('Risk-Return Profile by Scenario and Risk Configuration')
    plt.xlabel('Maximum Drawdown')
    plt.ylabel('Sharpe Ratio')
    plt.xscale('linear')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis as percentage
    def percentage_formatter(x, pos):
        return f'{100*x:.0f}%'
    
    scatter.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Add annotations for each point
    for idx, row in results_df.iterrows():
        plt.annotate(
            f"{row['scenario']}\n{row['risk_config']}",
            (row['max_drawdown'], row['sharpe_ratio']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(results_dir / "risk_return_comparison.png")
    plt.close()
    
    # 3. Heatmap of Performance Metrics
    # Pivot data for heatmaps
    metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
    
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        pivot_data = results_df.pivot_table(
            index='risk_config', 
            columns='scenario', 
            values=metric
        )
        
        # Adjust format for percentages
        if metric == 'max_drawdown' or metric == 'win_rate':
            sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.1%')
        else:
            sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.4f')
            
        plt.title(f'{metric.replace("_", " ").title()} by Risk Configuration and Scenario')
        plt.tight_layout()
        plt.savefig(results_dir / f"{metric}_heatmap.png")
        plt.close()
    
    # 4. Detailed Metrics Table
    # Create a formatted HTML table with all metrics
    metrics_table = results_df[['scenario', 'risk_config', 'final_portfolio_value', 
                              'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                              'win_rate', 'total_trades']]
    
    metrics_table_html = metrics_table.to_html(
        index=False, 
        float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x
    )
    
    with open(results_dir / "metrics_table.html", "w") as f:
        f.write(metrics_table_html)
    
    # 5. Risk Event Analysis
    if 'risk_events' in results_df.columns:
        # Count risk events by type for each scenario and risk config
        event_counts = {}
        
        for idx, row in results_df.iterrows():
            scenario = row['scenario']
            risk_config = row['risk_config']
            
            if scenario not in event_counts:
                event_counts[scenario] = {}
            
            if risk_config not in event_counts[scenario]:
                event_counts[scenario][risk_config] = {}
            
            # Count event types
            if isinstance(row['risk_events'], list):
                for event in row['risk_events']:
                    event_type = event.split('_')[0] if '_' in event else event
                    event_counts[scenario][risk_config][event_type] = event_counts[scenario][risk_config].get(event_type, 0) + 1
        
        # Convert to DataFrame for visualization
        event_data = []
        for scenario in event_counts:
            for risk_config in event_counts[scenario]:
                for event_type, count in event_counts[scenario][risk_config].items():
                    event_data.append({
                        'scenario': scenario,
                        'risk_config': risk_config,
                        'event_type': event_type,
                        'count': count
                    })
        
        if event_data:
            event_df = pd.DataFrame(event_data)
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='scenario', y='count', hue='event_type', data=event_df)
            plt.title('Risk Events by Scenario and Type')
            plt.ylabel('Event Count')
            plt.xlabel('Market Scenario')
            plt.xticks(rotation=45)
            plt.legend(title='Event Type')
            plt.tight_layout()
            plt.savefig(results_dir / "risk_events_analysis.png")
            plt.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run scenario backtests for risk management evaluation")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to risk configuration file")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save results")
    parser.add_argument("--scenario-config", type=str, help="Path to JSON file with scenario configurations")
    parser.add_argument("--risk-config", type=str, help="Path to JSON file with risk configurations")
    parser.add_argument("--sequential", action="store_true", help="Run backtests sequentially")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load scenario config from file if provided
    scenarios = None
    if args.scenario_config:
        try:
            with open(args.scenario_config, "r") as f:
                scenarios = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading scenario config: {e}")
            return
    
    # Load risk config from file if provided
    risk_configs = None
    if args.risk_config:
        try:
            with open(args.risk_config, "r") as f:
                risk_configs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading risk config: {e}")
            return
    
    # Run backtest
    results_df = run_market_scenario_backtest(
        config_path=args.config,
        results_dir=args.results_dir,
        scenarios=scenarios,
        risk_configs=risk_configs,
        parallel=not args.sequential,
        seed=args.seed
    )
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Total scenarios: {results_df['scenario'].nunique()}")
    print(f"Total risk configurations: {results_df['risk_config'].nunique()}")
    print(f"Total backtests: {len(results_df)}")
    print(f"Results saved to: {args.results_dir}")
    
    # Print best configurations by scenario
    print("\nBest Risk Configuration by Scenario (Sharpe Ratio):")
    best_by_sharpe = results_df.loc[results_df.groupby('scenario')['sharpe_ratio'].idxmax()]
    for _, row in best_by_sharpe.iterrows():
        print(f"{row['scenario']}: {row['risk_config']} (Sharpe: {row['sharpe_ratio']:.4f})")
    
    # Print best configurations by scenario (based on drawdown)
    print("\nBest Risk Configuration by Scenario (Minimum Drawdown):")
    best_by_drawdown = results_df.loc[results_df.groupby('scenario')['max_drawdown'].idxmin()]
    for _, row in best_by_drawdown.iterrows():
        print(f"{row['scenario']}: {row['risk_config']} (Drawdown: {row['max_drawdown']:.2%})")

if __name__ == "__main__":
    main() 