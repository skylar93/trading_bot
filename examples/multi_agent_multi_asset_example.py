"""
Multi-Agent Multi-Asset Trading Example.

This example demonstrates how to use the MultiAgentMultiAssetEnv to
create a trading environment where multiple agents can trade multiple assets
simultaneously.

Features:
- Configure multiple agents with different strategies
- Assign specific assets to specific agents
- Use shared or independent capital pools
- Process actions and observations in multi-agent context
- Visualize performance across agents and assets
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import torch
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv
from agents.strategies.agent_factory import create_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_example_data() -> pd.DataFrame:
    """
    Load example data with multiple assets.
    
    Returns:
        DataFrame with OHLCV data for multiple assets
    """
    # Check if test data is available
    data_path = os.path.join(project_root, "multi_asset_data_example.csv")
    
    if not os.path.exists(data_path):
        # Create synthetic data if real data not available
        logger.info("Creating synthetic data for multiple assets")
        
        # Generate synthetic price data for BTC, ETH, and SPY
        np.random.seed(42)
        nrows = 1000
        
        assets = ["BTC", "ETH", "SPY", "GOLD"]
        
        # Create empty DataFrame
        data = []
        
        # Generate data for each asset
        for asset in assets:
            # Generate price series with different characteristics
            if asset == "BTC":
                # More volatile
                price = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.04, nrows)))
                volume = np.random.lognormal(15, 1, nrows)
            elif asset == "ETH":
                # Medium volatility, somewhat correlated with BTC
                price = 400 * (1 + np.cumsum(np.random.normal(0.0008, 0.035, nrows)))
                volume = np.random.lognormal(14, 1, nrows)
            elif asset == "SPY":
                # Less volatile
                price = 300 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, nrows)))
                volume = np.random.lognormal(16, 0.8, nrows)
            else:  # GOLD
                # Least volatile
                price = 1800 * (1 + np.cumsum(np.random.normal(0.0002, 0.008, nrows)))
                volume = np.random.lognormal(13, 0.7, nrows)
                
            # Add some price jumps
            for _ in range(5):
                jump_idx = np.random.randint(0, nrows)
                jump_size = np.random.normal(0, 0.1)
                price[jump_idx:] *= (1 + jump_size)
            
            # Create OHLCV data
            for i in range(nrows):
                # Daily volatility
                daily_vol = price[i] * np.random.uniform(0.01, 0.03)
                high = price[i] + daily_vol * np.random.random()
                low = price[i] - daily_vol * np.random.random()
                close = price[i] + daily_vol * (np.random.random() - 0.5)
                
                # Ensure OHLC relationship
                open_price = price[i]
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Ensure prices are positive
                high = max(0.01, high)
                low = max(0.01, low)
                close = max(0.01, close)
                open_price = max(0.01, open_price)
                
                # Add row
                data.append({
                    'asset': asset,
                    '$open': open_price,
                    '$high': high,
                    '$low': low,
                    '$close': close,
                    '$volume': volume[i]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sort by asset
        df = df.sort_values(['asset']).reset_index(drop=True)
        
        # Save to file for future use
        df.to_csv(data_path, index=False)
        logger.info(f"Saved synthetic data to {data_path}")
    else:
        # Load existing data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    
    return df

def run_multi_agent_multi_asset_example():
    """
    Run an example of multiple agents trading multiple assets.
    """
    logger.info("Running multi-agent multi-asset example")
    
    # Load data
    data = load_example_data()
    
    # Check the data
    assets = data['asset'].unique()
    logger.info(f"Loaded data with {len(assets)} assets: {assets}")
    
    # Define agent configurations
    agent_configs = [
        {
            "id": "momentum_trader",
            "agent_type": "ppo",  # Using PPO algorithm
            "strategy": "momentum",  # Momentum trading strategy
            "initial_balance": 5000.0,
            "initial_capital_percentage": 0.4,  # 40% of total capital
            "priority": 2,  # Higher priority agent executes first
            "assigned_assets": ["BTC", "ETH"],  # This agent only trades crypto
            "hyperparameters": {
                "learning_rate": 0.0003,
                "gamma": 0.99
            }
        },
        {
            "id": "value_trader",
            "agent_type": "ppo",
            "strategy": "mean_reversion",  # Mean reversion strategy
            "initial_balance": 7500.0, 
            "initial_capital_percentage": 0.6,  # 60% of total capital
            "priority": 1,
            "assigned_assets": ["SPY", "GOLD"],  # This agent only trades traditional assets
            "hyperparameters": {
                "learning_rate": 0.0001,
                "gamma": 0.95
            }
        }
    ]
    
    # Create environment with shared capital pool
    env = MultiAgentMultiAssetEnv(
        data=data,
        agent_configs=agent_configs,
        window_size=20,
        trading_fee=0.001,
        action_type="portfolio_weights",  # Using portfolio weights for allocation
        shared_capital=True,  # Agents share a capital pool
        capital_reallocation_freq=10  # Reallocate capital every 10 steps
    )
    
    # Reset environment
    observations, _ = env.reset()
    
    # Create agents
    agents = {}
    for agent_id in env.agents:
        # Get observation and action spaces for this agent
        obs_space = env.observation_spaces[agent_id]
        act_space = env.action_spaces[agent_id]
        
        # Get agent config
        agent_config = next(cfg for cfg in agent_configs if cfg["id"] == agent_id)
        
        # Create agent
        agent = create_agent(
            agent_type=agent_config["agent_type"],
            strategy=agent_config["strategy"],
            config=agent_config.get("hyperparameters", {}),
            observation_space=obs_space,
            action_space=act_space
        )
        
        agents[agent_id] = agent
    
    # Run simulation
    portfolio_values = {agent_id: [] for agent_id in env.agents}
    rewards = {agent_id: [] for agent_id in env.agents}
    asset_positions = {agent_id: {asset: [] for asset in env.agent_assets[agent_id]} for agent_id in env.agents}
    
    # Store initial portfolio values
    for agent_id in env.agents:
        portfolio_values[agent_id].append(env.agent_portfolio_values[agent_id])
    
    # Run for 100 steps or until done
    for step in range(100):
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            # Get observation for this agent
            obs = observations[agent_id]
            
            # Get action from agent
            action = agent.get_action(obs)
            actions[agent_id] = action
        
        # Take step in environment
        next_observations, step_rewards, dones, truncated, infos = env.step(actions)
        
        # Update observations
        observations = next_observations
        
        # Store portfolio values and rewards
        for agent_id in env.agents:
            portfolio_values[agent_id].append(infos[agent_id]["portfolio_value"])
            rewards[agent_id].append(step_rewards[agent_id])
            
            # Store positions
            for asset in env.agent_assets[agent_id]:
                position = infos[agent_id]["positions"].get(asset, 0.0)
                asset_positions[agent_id][asset].append(position)
        
        # Print progress
        if step % 10 == 0:
            logger.info(f"Step {step} - Portfolio values: " + 
                       ", ".join([f"{agent_id}: ${pv[-1]:.2f}" for agent_id, pv in portfolio_values.items()]))
        
        # Check if done
        if all(dones.values()):
            logger.info("Environment signaled done")
            break
    
    # Print final results
    logger.info("\nFinal Results:")
    for agent_id in env.agents:
        initial_value = portfolio_values[agent_id][0]
        final_value = portfolio_values[agent_id][-1]
        returns = (final_value / initial_value - 1) * 100
        logger.info(f"Agent {agent_id}:")
        logger.info(f"  Initial: ${initial_value:.2f}, Final: ${final_value:.2f}")
        logger.info(f"  Returns: {returns:.2f}%")
        logger.info(f"  Assigned Assets: {env.agent_assets[agent_id]}")
        logger.info(f"  Final Positions: {', '.join([f'{asset}: {pos[-1]:.4f}' for asset, pos in asset_positions[agent_id].items()])}")
    
    # Plot results
    # Plot portfolio values over time
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for agent_id, values in portfolio_values.items():
        plt.plot(values, label=f"{agent_id}")
    plt.title("Portfolio Values Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    # Plot rewards over time
    plt.subplot(2, 1, 2)
    for agent_id, values in rewards.items():
        plt.plot(values, label=f"{agent_id}")
    plt.title("Rewards Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(project_root, "test_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "multi_agent_multi_asset_results.png"))
    plt.close()
    
    # Plot asset positions over time
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each agent
    for i, agent_id in enumerate(env.agents):
        plt.subplot(len(env.agents), 1, i+1)
        for asset, positions in asset_positions[agent_id].items():
            plt.plot(positions, label=f"{asset}")
        plt.title(f"{agent_id} Asset Positions")
        plt.xlabel("Step")
        plt.ylabel("Position Size")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "multi_agent_multi_asset_positions.png"))
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")
    
    return {
        "portfolio_values": portfolio_values,
        "rewards": rewards,
        "asset_positions": asset_positions
    }

def run_independent_capital_example():
    """
    Run an example of multiple agents with independent capital pools.
    """
    logger.info("Running multi-agent multi-asset example with independent capital")
    
    # Load data
    data = load_example_data()
    
    # Define agent configurations
    agent_configs = [
        {
            "id": "aggressive_trader",
            "agent_type": "ppo",
            "strategy": "momentum",
            "initial_balance": 5000.0,
            "assigned_assets": ["BTC", "ETH", "SPY", "GOLD"]  # Trades all assets
        },
        {
            "id": "conservative_trader",
            "agent_type": "ppo",
            "strategy": "mean_reversion",
            "initial_balance": 5000.0,
            "assigned_assets": ["BTC", "ETH", "SPY", "GOLD"]  # Trades all assets
        }
    ]
    
    # Create environment with independent capital pools
    env = MultiAgentMultiAssetEnv(
        data=data,
        agent_configs=agent_configs,
        window_size=20,
        trading_fee=0.001,
        action_type="portfolio_weights",
        shared_capital=False  # Each agent has independent capital
    )
    
    # Reset environment
    observations, _ = env.reset()
    
    # Create agents
    agents = {}
    for agent_id in env.agents:
        # Get observation and action spaces for this agent
        obs_space = env.observation_spaces[agent_id]
        act_space = env.action_spaces[agent_id]
        
        # Get agent config
        agent_config = next(cfg for cfg in agent_configs if cfg["id"] == agent_id)
        
        # Create agent with different behavior
        if agent_id == "aggressive_trader":
            # Aggressive agent puts more weight on BTC and ETH
            def get_action(obs):
                # Simple rule-based action: overweight crypto
                num_assets = act_space.shape[0]
                weights = np.array([0.4, 0.3, 0.2, 0.1])  # BTC, ETH, SPY, GOLD
                return weights
            
            agent = lambda obs: get_action(obs)
        else:
            # Conservative agent puts more weight on SPY and GOLD
            def get_action(obs):
                # Simple rule-based action: overweight traditional assets
                num_assets = act_space.shape[0]
                weights = np.array([0.1, 0.2, 0.4, 0.3])  # BTC, ETH, SPY, GOLD
                return weights
            
            agent = lambda obs: get_action(obs)
        
        agents[agent_id] = agent
    
    # Run simulation
    portfolio_values = {agent_id: [] for agent_id in env.agents}
    
    # Run for 100 steps
    for step in range(100):
        # Get actions from agents
        actions = {}
        for agent_id, agent in agents.items():
            # Get observation for this agent
            obs = observations[agent_id]
            
            # Get action from agent
            action = agent(obs)
            actions[agent_id] = action
        
        # Take step in environment
        next_observations, rewards, dones, truncated, infos = env.step(actions)
        
        # Update observations
        observations = next_observations
        
        # Store portfolio values
        for agent_id in env.agents:
            portfolio_values[agent_id].append(infos[agent_id]["portfolio_value"])
        
        # Print progress
        if step % 10 == 0:
            logger.info(f"Step {step} - Portfolio values: " + 
                       ", ".join([f"{agent_id}: ${pv[-1]:.2f}" for agent_id, pv in portfolio_values.items()]))
        
        # Check if done
        if all(dones.values()):
            logger.info("Environment signaled done")
            break
    
    # Print final results
    logger.info("\nFinal Results (Independent Capital):")
    for agent_id in env.agents:
        initial_value = portfolio_values[agent_id][0]
        final_value = portfolio_values[agent_id][-1]
        returns = (final_value / initial_value - 1) * 100
        logger.info(f"Agent {agent_id}:")
        logger.info(f"  Initial: ${initial_value:.2f}, Final: ${final_value:.2f}")
        logger.info(f"  Returns: {returns:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for agent_id, values in portfolio_values.items():
        plt.plot(values, label=f"{agent_id}")
    plt.title("Portfolio Values Over Time (Independent Capital)")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_dir = os.path.join(project_root, "test_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "multi_agent_independent_capital_results.png"))
    
    logger.info(f"Plot saved to {output_dir}")
    
    return portfolio_values

if __name__ == "__main__":
    # Run the examples
    logger.info("Starting multi-agent multi-asset examples")
    
    # Run shared capital example
    shared_results = run_multi_agent_multi_asset_example()
    
    # Run independent capital example
    independent_results = run_independent_capital_example()
    
    logger.info("Examples completed successfully") 