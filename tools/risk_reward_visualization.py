import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import matplotlib as mpl

# Add the project root to path so we can import the environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.single_asset_rl_env import SingleAssetRLTradingEnv


def generate_sample_data(periods=500):
    """
    Create realistic sample market data with trends, volatility clusters, and occasional jumps.
    """
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2023-01-01", periods=periods, freq="h")
    
    # Create more realistic price series
    price = 100.0
    prices = []
    for i in range(periods):
        # Add trend component
        trend = 0.01 * np.sin(i/100)
        
        # Add volatility component (higher during certain periods)
        vol = 0.5 + 0.5 * np.abs(np.sin(i/50))
        
        # Add occasional jumps
        jump = 0.0
        if np.random.random() < 0.02:  # 2% chance of jump
            jump = np.random.choice([-3.0, 3.0]) * vol
        
        # Calculate price change and update price
        change = trend + vol * np.random.normal(0, 0.1) + jump
        price = max(price * (1 + change), 1.0)  # Ensure price stays positive
        prices.append(price)
    
    # Calculate realistic OHLCV data
    base_prices = np.array(prices)
    df = pd.DataFrame(
        {
            "$open": base_prices * (1 + np.random.normal(0, 0.002, periods)),
            "$high": base_prices * (1 + np.abs(np.random.normal(0, 0.008, periods))),
            "$low": base_prices * (1 - np.abs(np.random.normal(0, 0.008, periods))),
            "$close": base_prices,
            "$volume": np.abs(np.random.exponential(1, periods) * base_prices * 10),
        },
        index=dates,
    )
    
    # Ensure high is always highest and low is always lowest
    df["$high"] = np.maximum(df["$high"], np.maximum(df["$open"], df["$close"]))
    df["$low"] = np.minimum(df["$low"], np.minimum(df["$open"], df["$close"]))
    
    return df


def generate_plots(save_dir="./test_results"):
    """
    Generate plots to visualize the impact of risk-adjusted rewards and frictions.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    mpl.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    
    # Generate sample data
    sample_data = generate_sample_data(periods=500)
    
    # Create three environments for comparison
    basic_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=False,
        apply_slippage=False,
        partial_fills=False
    )
    
    risk_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=True,
        sharpe_weight=0.5,
        drawdown_penalty=True,
        apply_slippage=False,
        partial_fills=False
    )
    
    friction_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=True,
        sharpe_weight=0.5,
        drawdown_penalty=True,
        apply_slippage=True,
        slippage_factor=0.001,
        partial_fills=True,
        min_fill_rate=0.7
    )
    
    # Reset all environments with the same seed
    seed = 42
    basic_obs, _ = basic_env.reset(seed=seed)
    risk_obs, _ = risk_env.reset(seed=seed)
    friction_obs, _ = friction_env.reset(seed=seed)
    
    # Lists to store data for plotting
    basic_rewards = []
    risk_rewards = []
    friction_rewards = []
    basic_values = []
    risk_values = []
    friction_values = []
    sharpe_ratios = []
    drawdowns = []
    slippages = []
    fill_rates = []
    
    # Trading strategy: Combine trend-following with some randomness
    np.random.seed(seed)
    actions = []
    
    # Generate a mix of strategies for more interesting behavior
    for i in range(200):
        if i < 50:
            # First 50 steps: Mostly positive positions (trend following upward)
            action = np.array([0.5 + 0.5 * np.random.random()])
        elif i < 100:
            # Next 50 steps: Mostly negative positions (trend following downward)
            action = np.array([-0.5 - 0.5 * np.random.random()])
        elif i < 150:
            # Next 50 steps: Oscillating positions (more volatile)
            action = np.array([np.sin(i/5)])
        else:
            # Final steps: Random positions
            action = np.array([np.random.uniform(-1, 1)])
            
        actions.append(action)
    
    # Run all environments with the same actions
    for action in actions:
        # Basic env
        _, r_basic, done_basic, _, info_basic = basic_env.step(action)
        basic_rewards.append(r_basic)
        basic_values.append(info_basic["portfolio_value"])
        
        # Risk env
        _, r_risk, done_risk, _, info_risk = risk_env.step(action)
        risk_rewards.append(r_risk)
        risk_values.append(info_risk["portfolio_value"])
        sharpe_ratios.append(info_risk["sharpe_ratio"])
        drawdowns.append(info_risk["drawdown"])
        
        # Friction env
        _, r_friction, done_friction, _, info_friction = friction_env.step(action)
        friction_rewards.append(r_friction)
        friction_values.append(info_friction["portfolio_value"])
        slippages.append(info_friction["last_slippage"])
        fill_rates.append(info_friction["last_fill_rate"])
        
        if done_basic or done_risk or done_friction:
            break
    
    # Plot 1: Portfolio Values
    plt.figure(figsize=(12, 6))
    plt.plot(basic_values, label="Basic Environment", linewidth=2)
    plt.plot(risk_values, label="Risk-Adjusted Environment", linewidth=2)
    plt.plot(friction_values, label="Friction Environment", linewidth=2)
    plt.title("Portfolio Values Comparison", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/portfolio_values_comparison.png", dpi=300)
    
    # Plot 2: Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(basic_rewards, label="Basic Rewards", linewidth=2, alpha=0.7)
    plt.plot(risk_rewards, label="Risk-Adjusted Rewards", linewidth=2, alpha=0.7)
    plt.plot(friction_rewards, label="Friction Rewards", linewidth=2, alpha=0.7)
    plt.title("Rewards Comparison", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rewards_comparison.png", dpi=300)
    
    # Plot 3: Sharpe Ratio and Drawdown
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Sharpe Ratio', color=color)
    ax1.plot(sharpe_ratios, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Drawdown', color=color)
    ax2.plot(drawdowns, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Sharpe Ratio and Drawdown", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"{save_dir}/sharpe_drawdown.png", dpi=300)
    
    # Plot 4: Slippage and Fill Rates
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:green'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Slippage', color=color)
    ax1.plot(slippages, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Fill Rate', color=color)
    ax2.plot(fill_rates, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Slippage and Fill Rates", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"{save_dir}/slippage_fill_rates.png", dpi=300)
    
    # Plot 5: Cumulative rewards comparison
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(basic_rewards), label="Basic Cumulative Reward", linewidth=2)
    plt.plot(np.cumsum(risk_rewards), label="Risk-Adjusted Cumulative Reward", linewidth=2)
    plt.plot(np.cumsum(friction_rewards), label="Friction Cumulative Reward", linewidth=2)
    plt.title("Cumulative Rewards Comparison", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cumulative_rewards.png", dpi=300)
    
    # Plot 6: Risk-Adjusted vs Basic Reward Difference
    plt.figure(figsize=(12, 6))
    reward_diff = np.array(risk_rewards) - np.array(basic_rewards)
    plt.plot(reward_diff, color='purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Risk-Adjusted Reward Difference (Risk - Basic)", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Reward Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_difference.png", dpi=300)
    
    print(f"Plots saved to {save_dir}")
    
    # Calculate some statistics to show the differences
    stats = {
        "Basic Env": {
            "Final Portfolio": basic_values[-1],
            "Mean Reward": np.mean(basic_rewards),
            "Reward Std": np.std(basic_rewards),
            "Max Drawdown": "N/A",
        },
        "Risk-Adjusted Env": {
            "Final Portfolio": risk_values[-1],
            "Mean Reward": np.mean(risk_rewards),
            "Reward Std": np.std(risk_rewards),
            "Max Drawdown": max(drawdowns),
        },
        "Friction Env": {
            "Final Portfolio": friction_values[-1],
            "Mean Reward": np.mean(friction_rewards),
            "Reward Std": np.std(friction_rewards),
            "Max Drawdown": "N/A",
            "Mean Slippage": np.mean([s for s in slippages if s > 0]),
            "Mean Fill Rate": np.mean([f for f in fill_rates if f < 1.0]),
        }
    }
    
    print("\nComparison Statistics:")
    for env_name, env_stats in stats.items():
        print(f"\n{env_name}:")
        for stat_name, stat_value in env_stats.items():
            print(f"  {stat_name}: {stat_value}")


if __name__ == "__main__":
    generate_plots() 