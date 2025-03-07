#!/usr/bin/env python
"""
Multi-Asset Trading Example

This script demonstrates various ways to implement and use multi-asset trading environments
with different agent and capital management approaches.

1. Single agent managing multiple assets
2. Multiple agents each managing a different asset
3. Shared vs. isolated capital management
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, List, Optional, Tuple
import gym
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_asset_example.log')
    ]
)

logger = logging.getLogger(__name__)

# Import our custom modules
from envs.multi_asset_env import MultiAssetTradingEnv
from envs.multi_agent_env import MultiAgentTradingEnv
from envs.capital_manager import CapitalManager, MultiAssetCapitalManager
from agents.strategies.asset_specific_agents import AssetSpecificAgentFactory
from networks.multi_asset_policy import MultiAssetLSTMPolicy, MultiAssetAttentionPolicy
from data.utils.multi_asset_data_loader import MultiAssetDataLoader

def load_example_data(start_date='2023-01-01', end_date='2023-01-31'):
    """Load example data for multiple assets."""
    logger.info(f"Loading data from {start_date} to {end_date}")
    
    data_loader = MultiAssetDataLoader(
        base_dir='data/raw',
        interval='1d'
    )
    
    # Load data for multiple assets
    asset_data = data_loader.load_multi_asset_data(
        assets=['BTC/USDT', 'ETH/USDT'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Prepare unified DataFrame
    unified_df = data_loader.prepare_unified_dataframe(asset_data)
    
    logger.info(f"Loaded data shape: {unified_df.shape}")
    return unified_df

def example_single_agent_multi_asset():
    """Example of a single agent managing multiple assets."""
    logger.info("Running single agent multi-asset example")
    
    # Load data
    data = load_example_data()
    
    # Create environment
    env = MultiAssetTradingEnv(
        df=data,
        assets=['BTC', 'ETH'],
        initial_balance=10000.0,
        window_size=10,
        action_type='portfolio_weights',  # Use portfolio weights for managing multiple assets
        allow_short=False,
        rebalance_freq=1
    )
    
    # Create a capital manager (shared capital pool)
    capital_manager = MultiAssetCapitalManager(
        env=env,
        mode='shared',
        allocation_weights={'BTC': 0.6, 'ETH': 0.4},
        max_leverage=1.0
    )
    
    # Create the policy network
    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Using LSTM policy for time series data
    policy = MultiAssetLSTMPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_assets=len(env.assets),
        window_size=env.window_size,
        features_per_asset=env.observation_space.shape[1] // env.window_size // len(env.assets)
    ).to(device)
    
    # Run a simple example
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    portfolio_values = []
    asset_weights = {asset: [] for asset in env.assets}
    asset_weights['cash'] = []
    
    while not done:
        # Get the observation tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action = policy.get_action(obs_tensor, deterministic=True).squeeze().cpu().numpy()
        
        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update capital manager
        capital_manager.update_from_env_state()
        
        # Record data for plotting
        portfolio_values.append(env.portfolio_value)
        for asset in env.assets:
            asset_weights[asset].append(env.current_weights.get(asset, 0.0))
        asset_weights['cash'].append(env.current_weights.get('cash', 0.0))
        
        # Log state
        logger.info(f"Step {env.current_step}, Portfolio: ${env.portfolio_value:.2f}, Reward: {reward:.6f}")
        for asset in env.assets:
            pos = env.positions.get(asset, 0.0)
            weight = env.current_weights.get(asset, 0.0)
            logger.info(f"  {asset}: {pos:.6f} units, weight: {weight:.2%}")
        
        total_reward += reward
        obs = next_obs
    
    logger.info(f"Episode complete. Total reward: {total_reward:.6f}")
    logger.info(f"Final portfolio value: ${env.portfolio_value:.2f}")
    
    # Plot results
    plot_results(
        portfolio_values=portfolio_values,
        asset_weights=asset_weights,
        title="Single Agent Managing Multiple Assets"
    )

def example_multi_agent():
    """Example of multiple agents each managing a different asset."""
    logger.info("Running multi-agent example")
    
    # Load data
    data = load_example_data()
    
    # Create agents for different assets
    agents = {}
    
    # Set up observation and action spaces for testing
    dummy_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(50,))
    dummy_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    
    # Create crypto agent for BTC
    crypto_agent = AssetSpecificAgentFactory.create_agent(
        asset_id="BTC",
        asset_type="crypto",
        observation_space=dummy_obs_space,
        action_space=dummy_action_space,
        config={
            "volatility_scaling": True
        }
    )
    
    # Create equity agent for ETH (just for demonstration - ETH is crypto but using different agent type)
    equity_agent = AssetSpecificAgentFactory.create_agent(
        asset_id="ETH",
        asset_type="equity",
        observation_space=dummy_obs_space,
        action_space=dummy_action_space
    )
    
    agents["BTC"] = crypto_agent
    agents["ETH"] = equity_agent
    
    # Create environment (using MultiAssetTradingEnv for simplicity in this example)
    env = MultiAssetTradingEnv(
        df=data,
        assets=['BTC', 'ETH'],
        initial_balance=10000.0,
        window_size=10,
        action_type='discrete_amount'  # Each agent decides position sizing directly
    )
    
    # Create capital manager (isolated mode)
    capital_manager = MultiAssetCapitalManager(
        env=env,
        mode='isolated',  # Each asset has its own capital
        allocation_weights={'BTC': 0.7, 'ETH': 0.3}  # 70% capital to BTC, 30% to ETH
    )
    
    # Run a simple example
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    portfolio_values = []
    asset_positions = {asset: [] for asset in env.assets}
    asset_values = {asset: [] for asset in env.assets}
    
    while not done:
        # Extract observations per asset
        asset_obs = {}
        features_per_asset = obs.shape[1] // len(env.assets)
        
        for i, asset in enumerate(env.assets):
            asset_obs[asset] = obs[:, i * features_per_asset:(i+1) * features_per_asset]
        
        # Get actions from each agent
        actions = []
        for asset in env.assets:
            # Agent makes decision based on its specific logic
            agent_obs = asset_obs[asset]
            agent_action = agents[asset].act(agent_obs)
            
            # Check capital constraints
            max_position = capital_manager.get_max_position_size(asset)
            if max_position > 0:
                # Scale action to respect capital constraints
                agent_action = np.clip(agent_action, -max_position, max_position) / max_position
            else:
                agent_action = np.zeros_like(agent_action)
                
            actions.append(agent_action[0])  # Take first action value
        
        action = np.array(actions)
        
        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update capital manager and agents
        capital_manager.update_from_env_state()
        
        # Update agent states
        for asset in env.assets:
            position = env.positions.get(asset, 0.0)
            price = env.prices.get(asset, 0.0)
            pnl = 0.0
            if position != 0 and agents[asset].state["avg_entry_price"] > 0:
                pnl = (price / agents[asset].state["avg_entry_price"] - 1) * position
            
            agents[asset].update_state({
                "position": position,
                "avg_entry_price": env.avg_entry_prices.get(asset, 0.0),
                "unrealized_pnl": pnl
            })
        
        # Record data for plotting
        portfolio_values.append(env.portfolio_value)
        for asset in env.assets:
            asset_positions[asset].append(env.positions.get(asset, 0.0))
            asset_values[asset].append(env.positions.get(asset, 0.0) * env.prices.get(asset, 0.0))
        
        # Log state
        logger.info(f"Step {env.current_step}, Portfolio: ${env.portfolio_value:.2f}, Reward: {reward:.6f}")
        for asset in env.assets:
            pos = env.positions.get(asset, 0.0)
            capital = capital_manager.allocated_capital.get(asset, 0.0)
            logger.info(f"  {asset}: {pos:.6f} units, allocated capital: ${capital:.2f}")
        
        total_reward += reward
        obs = next_obs
    
    logger.info(f"Episode complete. Total reward: {total_reward:.6f}")
    logger.info(f"Final portfolio value: ${env.portfolio_value:.2f}")
    
    # Plot results
    plot_multi_agent_results(
        portfolio_values=portfolio_values,
        asset_positions=asset_positions,
        asset_values=asset_values,
        title="Multiple Agents - Isolated Capital"
    )

def example_shared_vs_isolated_capital():
    """Compare shared vs isolated capital approaches."""
    logger.info("Running shared vs isolated capital example")
    
    # Load data
    data = load_example_data()
    
    # Create two environments with identical settings, except for capital management
    shared_env = MultiAssetTradingEnv(
        df=data,
        assets=['BTC', 'ETH'],
        initial_balance=10000.0,
        window_size=10,
        action_type='portfolio_weights'
    )
    
    isolated_env = MultiAssetTradingEnv(
        df=data.copy(),
        assets=['BTC', 'ETH'],
        initial_balance=10000.0,
        window_size=10,
        action_type='portfolio_weights'
    )
    
    # Create capital managers
    shared_capital = MultiAssetCapitalManager(
        env=shared_env,
        mode='shared',
        allocation_weights={'BTC': 0.6, 'ETH': 0.4}
    )
    
    isolated_capital = MultiAssetCapitalManager(
        env=isolated_env,
        mode='isolated',
        allocation_weights={'BTC': 0.6, 'ETH': 0.4}
    )
    
    # Create a simple fixed-weight strategy (for demonstration)
    def fixed_weight_strategy(env, weights={'BTC': 0.6, 'ETH': 0.4}):
        return np.array([weights.get(asset, 0.0) for asset in env.assets])
    
    # Run simulations
    shared_portfolio_values = []
    isolated_portfolio_values = []
    shared_weights = {asset: [] for asset in shared_env.assets}
    isolated_weights = {asset: [] for asset in isolated_env.assets}
    
    # Reset environments
    shared_obs, _ = shared_env.reset()
    isolated_obs, _ = isolated_env.reset()
    
    shared_done = False
    isolated_done = False
    
    while not (shared_done and isolated_done):
        # Shared capital environment step
        if not shared_done:
            shared_action = fixed_weight_strategy(shared_env)
            shared_next_obs, shared_reward, shared_terminated, shared_truncated, shared_info = shared_env.step(shared_action)
            shared_done = shared_terminated or shared_truncated
            shared_capital.update_from_env_state()
            shared_portfolio_values.append(shared_env.portfolio_value)
            
            for asset in shared_env.assets:
                shared_weights[asset].append(shared_env.current_weights.get(asset, 0.0))
            
            shared_obs = shared_next_obs
        
        # Isolated capital environment step
        if not isolated_done:
            isolated_action = fixed_weight_strategy(isolated_env)
            isolated_next_obs, isolated_reward, isolated_terminated, isolated_truncated, isolated_info = isolated_env.step(isolated_action)
            isolated_done = isolated_terminated or isolated_truncated
            isolated_capital.update_from_env_state()
            isolated_portfolio_values.append(isolated_env.portfolio_value)
            
            for asset in isolated_env.assets:
                isolated_weights[asset].append(isolated_env.current_weights.get(asset, 0.0))
            
            isolated_obs = isolated_next_obs
    
    logger.info(f"Shared capital final portfolio: ${shared_env.portfolio_value:.2f}")
    logger.info(f"Isolated capital final portfolio: ${isolated_env.portfolio_value:.2f}")
    
    # Plot comparison results
    plot_comparison(
        shared_values=shared_portfolio_values,
        isolated_values=isolated_portfolio_values,
        shared_weights=shared_weights,
        isolated_weights=isolated_weights,
        title="Shared vs Isolated Capital"
    )

def plot_results(portfolio_values, asset_weights, title):
    """Plot portfolio performance and asset weights."""
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
    plt.title(f"{title} - Portfolio Value")
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Asset weights
    plt.subplot(2, 1, 2)
    for asset, weights in asset_weights.items():
        plt.plot(weights, label=f'{asset} Weight')
    
    plt.title('Asset Allocation Weights')
    plt.xlabel('Steps')
    plt.ylabel('Weight (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_multi_agent_results(portfolio_values, asset_positions, asset_values, title):
    """Plot multi-agent trading results."""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
    plt.title(f"{title} - Portfolio Value")
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Asset positions
    plt.subplot(3, 1, 2)
    for asset, positions in asset_positions.items():
        plt.plot(positions, label=f'{asset} Position')
    
    plt.title('Asset Positions')
    plt.xlabel('Steps')
    plt.ylabel('Position Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Asset values
    plt.subplot(3, 1, 3)
    for asset, values in asset_values.items():
        plt.plot(values, label=f'{asset} Value')
    
    plt.title('Asset Values')
    plt.xlabel('Steps')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_comparison(shared_values, isolated_values, shared_weights, isolated_weights, title):
    """Plot comparison between shared and isolated capital."""
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Portfolio values comparison
    plt.subplot(2, 1, 1)
    plt.plot(shared_values, label='Shared Capital', color='blue', linewidth=2)
    plt.plot(isolated_values, label='Isolated Capital', color='red', linewidth=2)
    plt.title(f"{title} - Portfolio Value Comparison")
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Asset weight differences
    plt.subplot(2, 1, 2)
    
    weight_diff = {}
    
    # Calculate weight differences
    min_len = min(len(shared_values), len(isolated_values))
    for asset in shared_weights.keys():
        if asset in isolated_weights:
            weight_diff[asset] = []
            for i in range(min_len):
                if i < len(shared_weights[asset]) and i < len(isolated_weights[asset]):
                    diff = shared_weights[asset][i] - isolated_weights[asset][i]
                    weight_diff[asset].append(diff)
    
    # Plot weight differences
    for asset, diffs in weight_diff.items():
        plt.plot(diffs, label=f'{asset} Weight Difference')
    
    plt.title('Asset Weight Differences (Shared - Isolated)')
    plt.xlabel('Steps')
    plt.ylabel('Weight Difference')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def main():
    """Run the examples."""
    logger.info("Starting multi-asset trading examples")
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Example 1: Single agent managing multiple assets
    example_single_agent_multi_asset()
    
    # Example 2: Multiple agents each managing a different asset
    example_multi_agent()
    
    # Example 3: Comparison of shared vs isolated capital
    example_shared_vs_isolated_capital()
    
    logger.info("All examples completed")

if __name__ == "__main__":
    main() 