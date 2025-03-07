#!/usr/bin/env python
"""Test script for different action space types in multi-asset trading environment."""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.utils.multi_asset_data_loader import MultiAssetDataLoader
from envs.multi_asset_env import MultiAssetTradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("test_action_space")

def load_test_data():
    """Load test data for multiple assets."""
    assets = [
        {'symbol': 'BTC/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'BTC'},
        {'symbol': 'ETH/USDT', 'exchange': 'binance', 'type': 'crypto', 'alias': 'ETH'},
    ]
    
    loader = MultiAssetDataLoader(assets=assets, timeframe='1d')
    df = loader.fetch_multi_asset_data('2023-01-01', '2023-01-31')
    
    if df.empty:
        logger.error("Failed to load data")
        return None
    
    return df, assets

def test_discrete_amount_action():
    """Test discrete_amount action type."""
    logger.info("=== Testing discrete_amount action type ===")
    
    # Load data
    df, assets_info = load_test_data()
    if df is None:
        return False
    
    assets = [asset['alias'] for asset in assets_info]
    
    # Create environment
    env = MultiAssetTradingEnv(
        df=df,
        assets=assets,
        window_size=10,
        normalize_observations=True,
        action_type='discrete_amount',
        max_position_size=0.5,  # Use up to 50% of balance per asset
        trading_fee=0.001
    )
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    logger.info(f"Initial positions: {info['positions']}")
    
    # Test different actions
    actions = [
        [0.5, 0.3],  # Buy 50% of max for BTC, 30% of max for ETH
        [0.0, 0.0],  # Hold both
        [-0.5, 0.0], # Sell 50% of BTC, hold ETH
        [0.2, -0.3], # Buy more BTC, sell some ETH
        [0.0, 0.0],  # Hold both
    ]
    
    results = []
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Record results
        results.append({
            'step': i,
            'action': action,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'positions': info['positions'].copy(),
            'weights': info['weights'].copy(),
            'reward': reward
        })
        
        logger.info(f"Step {i+1}, Action: {action}")
        logger.info(f"Portfolio value: ${info['portfolio_value']:.2f}, Reward: {reward:.6f}")
        logger.info(f"Positions: {info['positions']}")
        logger.info(f"Weights: {info['weights']}")
        logger.info("-" * 50)
    
    # Visualize results
    plot_results(results, 'discrete_amount_results.png', 'Discrete Amount Action Results')
    
    return True

def test_portfolio_weights_action():
    """Test portfolio_weights action type."""
    logger.info("=== Testing portfolio_weights action type ===")
    
    # Load data
    df, assets_info = load_test_data()
    if df is None:
        return False
    
    assets = [asset['alias'] for asset in assets_info]
    
    # Create environment
    env = MultiAssetTradingEnv(
        df=df,
        assets=assets,
        window_size=10,
        normalize_observations=True,
        action_type='portfolio_weights',
        portfolio_constraints={
            'sum_to_one': True,
            'max_weight': 0.8,
            'min_weight': 0.0
        },
        trading_fee=0.001,
        rebalance_freq=1  # Rebalance every step
    )
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    logger.info(f"Initial positions: {info['positions']}")
    
    # Test different target weights
    actions = [
        [0.6, 0.4],  # 60% BTC, 40% ETH
        [0.5, 0.5],  # 50% BTC, 50% ETH
        [0.3, 0.7],  # 30% BTC, 70% ETH
        [0.0, 0.0],  # 0% BTC, 0% ETH (all cash)
        [0.2, 0.8],  # 20% BTC, 80% ETH
    ]
    
    results = []
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Record results
        results.append({
            'step': i,
            'action': action,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'positions': info['positions'].copy(),
            'weights': info['weights'].copy(),
            'reward': reward
        })
        
        logger.info(f"Step {i+1}, Target weights: {action}")
        logger.info(f"Portfolio value: ${info['portfolio_value']:.2f}, Reward: {reward:.6f}")
        logger.info(f"Positions: {info['positions']}")
        logger.info(f"Actual weights: {info['weights']}")
        logger.info("-" * 50)
    
    # Visualize results
    plot_results(results, 'portfolio_weights_results.png', 'Portfolio Weights Action Results')
    
    return True

def test_discrete_signal_action():
    """Test discrete_signal action type."""
    logger.info("=== Testing discrete_signal action type ===")
    
    # Load data
    df, assets_info = load_test_data()
    if df is None:
        return False
    
    assets = [asset['alias'] for asset in assets_info]
    
    # Create environment
    env = MultiAssetTradingEnv(
        df=df,
        assets=assets,
        window_size=10,
        normalize_observations=True,
        action_type='discrete_signal',
        max_position_size=0.5,  # Use up to 50% of balance per asset
        trading_fee=0.001
    )
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    logger.info(f"Initial positions: {info['positions']}")
    
    # Test different signals (0: Sell, 1: Hold, 2: Buy)
    actions = [
        [2, 2],  # Buy BTC, Buy ETH
        [1, 1],  # Hold BTC, Hold ETH
        [0, 1],  # Sell BTC, Hold ETH
        [2, 0],  # Buy BTC, Sell ETH
        [1, 1],  # Hold BTC, Hold ETH
    ]
    
    results = []
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Record results
        results.append({
            'step': i,
            'action': action,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'positions': info['positions'].copy(),
            'weights': info['weights'].copy(),
            'reward': reward
        })
        
        # Convert signals to text
        signal_text = []
        for signal in action:
            if signal == 0:
                signal_text.append("Sell")
            elif signal == 1:
                signal_text.append("Hold")
            else:
                signal_text.append("Buy")
        
        logger.info(f"Step {i+1}, Signals: {signal_text}")
        logger.info(f"Portfolio value: ${info['portfolio_value']:.2f}, Reward: {reward:.6f}")
        logger.info(f"Positions: {info['positions']}")
        logger.info(f"Weights: {info['weights']}")
        logger.info("-" * 50)
    
    # Visualize results
    plot_results(results, 'discrete_signal_results.png', 'Discrete Signal Action Results')
    
    return True

def plot_results(results, filename, title):
    """Plot portfolio value and weights over time.
    
    Args:
        results: List of result dictionaries
        filename: Output filename
        title: Plot title
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot portfolio value
    steps = [r['step'] for r in results]
    portfolio_values = [r['portfolio_value'] for r in results]
    ax1.plot(steps, portfolio_values, 'b-o', linewidth=2)
    ax1.set_title('Portfolio Value')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True)
    
    # Plot weights
    cash_weights = [r['weights'].get('cash', 0) for r in results]
    
    # Get all asset names from positions
    assets = set()
    for r in results:
        assets.update(r['positions'].keys())
    
    # Plot weight for each asset
    for asset in sorted(assets):
        weights = [r['weights'].get(asset, 0) for r in results]
        ax2.plot(steps, weights, 'o-', linewidth=2, label=asset)
    
    # Plot cash weight
    ax2.plot(steps, cash_weights, 'o-', linewidth=2, label='Cash')
    
    ax2.set_title('Portfolio Weights')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Weight')
    ax2.grid(True)
    ax2.legend()
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    fig.savefig(filename)
    logger.info(f"Saved results plot to {filename}")

def main():
    """Run all action space tests."""
    try:
        # Run tests
        discrete_amount_success = test_discrete_amount_action()
        portfolio_weights_success = test_portfolio_weights_action()
        discrete_signal_success = test_discrete_signal_action()
        
        # Report results
        logger.info("\n=== Action Space Test Results ===")
        logger.info(f"Discrete Amount Action: {'✅ Passed' if discrete_amount_success else '❌ Failed'}")
        logger.info(f"Portfolio Weights Action: {'✅ Passed' if portfolio_weights_success else '❌ Failed'}")
        logger.info(f"Discrete Signal Action: {'✅ Passed' if discrete_signal_success else '❌ Failed'}")
        
        if all([discrete_amount_success, portfolio_weights_success, discrete_signal_success]):
            logger.info("\nAll action space tests passed!")
            return 0
        else:
            logger.error("\nSome action space tests failed!")
            return 1
    except Exception as e:
        logger.exception(f"Error in action space tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 