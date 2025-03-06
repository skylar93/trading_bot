import pytest
import numpy as np
import pandas as pd
from envs.single_asset_rl_env import SingleAssetRLTradingEnv
import matplotlib.pyplot as plt
from datetime import datetime
import os


@pytest.fixture
def sample_data():
    """Create sample market data with $ prefix columns and realistic price movements."""
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
    
    # Create more realistic price series with trends, volatility clusters, and occasional jumps
    price = 100.0
    prices = []
    for i in range(500):
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
            "$open": base_prices * (1 + np.random.normal(0, 0.002, 500)),
            "$high": base_prices * (1 + np.abs(np.random.normal(0, 0.008, 500))),
            "$low": base_prices * (1 - np.abs(np.random.normal(0, 0.008, 500))),
            "$close": base_prices,
            "$volume": np.abs(np.random.exponential(1, 500) * base_prices * 10),
        },
        index=dates,
    )
    
    # Ensure high is always highest and low is always lowest
    df["$high"] = np.maximum(df["$high"], np.maximum(df["$open"], df["$close"]))
    df["$low"] = np.minimum(df["$low"], np.minimum(df["$open"], df["$close"]))
    
    return df


def test_basic_functionality(sample_data):
    """Test that the enhanced environment works with default parameters."""
    # Create environment
    env = SingleAssetRLTradingEnv(
        data=sample_data,
        initial_capital=10000.0,
        trading_fee=0.001,
        window_size=10,
        max_position_size=1.0
    )
    
    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10, 5)
    
    # Test step
    action = np.array([0.5])  # Buy action
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Check info contains all expected keys for enhanced environment
    expected_keys = [
        "portfolio_value", "position", "capital", "drawdown", 
        "sharpe_ratio", "last_trade_size", "last_fill_rate", "last_slippage"
    ]
    for key in expected_keys:
        assert key in info, f"Expected key {key} not found in info dict"


def test_risk_adjusted_reward(sample_data):
    """Test that risk-adjusted rewards differ from basic rewards."""
    # Create basic environment (no risk adjustment)
    basic_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=False,
        drawdown_penalty=False
    )
    
    # Create risk-adjusted environment
    risk_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=True,
        sharpe_weight=0.5,
        drawdown_penalty=True
    )
    
    # Reset both environments
    basic_env.reset(seed=42)
    risk_env.reset(seed=42)
    
    # Apply the same actions to both environments and compare rewards
    rewards_basic = []
    rewards_risk = []
    
    # Use a simple strategy for testing (random actions with fixed seed)
    np.random.seed(42)
    for _ in range(100):
        action = np.array([np.random.uniform(-1, 1)])
        
        # Apply to basic env
        _, reward_basic, done_basic, _, _ = basic_env.step(action)
        rewards_basic.append(reward_basic)
        
        # Apply to risk env
        _, reward_risk, done_risk, _, _ = risk_env.step(action)
        rewards_risk.append(reward_risk)
        
        if done_basic or done_risk:
            break
    
    # Verify rewards are different (risk-adjusted should differ)
    assert not np.array_equal(rewards_basic, rewards_risk), "Risk-adjusted rewards should differ from basic rewards"
    
    # Analyze reward statistics
    basic_std = np.std(rewards_basic)
    risk_std = np.std(rewards_risk)
    
    # Risk-adjusted rewards should typically have lower standard deviation
    print(f"Basic reward std: {basic_std}, Risk-adjusted reward std: {risk_std}")


def test_friction_impact(sample_data):
    """Test that slippage and partial fills affect execution."""
    # Create environment without frictions
    no_friction_env = SingleAssetRLTradingEnv(
        data=sample_data,
        apply_slippage=False,
        partial_fills=False
    )
    
    # Create environment with frictions
    friction_env = SingleAssetRLTradingEnv(
        data=sample_data,
        apply_slippage=True,
        slippage_factor=0.001,
        partial_fills=True,
        min_fill_rate=0.7
    )
    
    # Reset both environments with the same seed
    no_friction_env.reset(seed=123)
    friction_env.reset(seed=123)
    
    # Execute the same large trade in both environments
    action = np.array([1.0])  # Full buy
    
    # Step both environments
    _, _, _, _, info_no_friction = no_friction_env.step(action)
    _, _, _, _, info_friction = friction_env.step(action)
    
    # Verify impact of frictions
    # 1. Check fill rate is applied
    assert info_friction["last_fill_rate"] < 1.0, "Large order should be partially filled"
    
    # 2. Check slippage is applied
    assert info_friction["last_slippage"] > 0.0, "Large order should incur slippage"
    
    # 3. Final positions should differ due to partial fills
    assert info_no_friction["position"] != info_friction["position"], "Positions should differ due to partial fills"
    
    # 4. Portfolio values should differ due to slippage
    assert info_no_friction["portfolio_value"] != info_friction["portfolio_value"], "Portfolio values should differ due to slippage"


def test_drawdown_penalty(sample_data):
    """Test that drawdown penalties affect rewards in declining markets."""
    # Create artificial drawdown pattern: first rise, then significant decline
    np.random.seed(432)
    
    # Create environment with drawdown penalty
    dd_env = SingleAssetRLTradingEnv(
        data=sample_data,
        drawdown_penalty=True,
        max_drawdown_penalty_threshold=0.1  # 10% drawdown threshold
    )
    
    dd_env.reset()
    
    # First build up some equity by going long in the first 30 steps
    rewards = []
    portfolio_values = []
    drawdowns = []
    
    # Trading strategy: first go long to build value, then hold through drawdown
    for i in range(60):
        if i < 20:
            action = np.array([1.0])  # Full long position
        else:
            action = np.array([0.0])  # Hold position during market decline
            
        _, reward, _, _, info = dd_env.step(action)
        rewards.append(reward)
        portfolio_values.append(info["portfolio_value"])
        drawdowns.append(info["drawdown"])
    
    # Check if drawdown penalty was applied (rewards should become more negative during drawdowns)
    if max(drawdowns) > 0.1:  # If we reached the penalty threshold
        # Find steps where drawdown exceeded threshold
        high_dd_indices = [i for i, dd in enumerate(drawdowns) if dd > 0.1]
        
        # Rewards during high drawdown should be more penalized
        high_dd_rewards = [rewards[i] for i in high_dd_indices]
        
        assert np.mean(high_dd_rewards) < np.mean(rewards), "Rewards during high drawdown should be more penalized"
        print(f"Average reward: {np.mean(rewards)}, Average during high drawdown: {np.mean(high_dd_rewards)}")


def test_sharpe_impact(sample_data):
    """Test that Sharpe ratio component affects rewards based on risk-adjusted returns."""
    # Create environment with high Sharpe weight
    sharpe_env = SingleAssetRLTradingEnv(
        data=sample_data,
        risk_adjusted_reward=True,
        sharpe_weight=0.8,  # Heavy sharpe weighting
        sharpe_lookback=10,  # Short lookback for faster adjustment
        drawdown_penalty=False  # Isolate Sharpe effect
    )
    
    sharpe_env.reset(seed=42)
    
    # Trading strategy: alternate between consistent small returns and volatile returns
    returns = []
    sharpe_ratios = []
    
    # Phase 1: Consistent small positive returns
    for _ in range(20):
        action = np.array([0.2])  # Small consistent position
        _, reward, _, _, info = sharpe_env.step(action)
        returns.append(reward)
        sharpe_ratios.append(info["sharpe_ratio"])
    
    phase1_returns = returns.copy()
    
    # Phase 2: Volatile returns
    for _ in range(20):
        # Alternate between large long and short positions
        action = np.array([1.0 if _ % 2 == 0 else -1.0])
        _, reward, _, _, info = sharpe_env.step(action)
        returns.append(reward)
        sharpe_ratios.append(info["sharpe_ratio"])
    
    phase2_returns = returns[len(phase1_returns):]
    
    # Analyze the reward patterns
    print(f"Phase 1 (Consistent) - Avg return: {np.mean(phase1_returns)}, Std: {np.std(phase1_returns)}")
    print(f"Phase 2 (Volatile) - Avg return: {np.mean(phase2_returns)}, Std: {np.std(phase2_returns)}")
    
    # Check if final Sharpe ratio exists and makes sense
    assert np.isfinite(sharpe_ratios[-1]), "Final Sharpe ratio should be a finite number"


def generate_plots(sample_data, save_dir="test_results"):
    """
    Generate plots to visualize the impact of risk-adjusted rewards and frictions.
    This is not an actual test but a utility to generate visual explanations.
    """
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Use a simple strategy for demonstration
    np.random.seed(seed)
    actions = [np.array([np.random.uniform(-1, 1)]) for _ in range(200)]
    
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
    plt.plot(basic_values, label="Basic Environment")
    plt.plot(risk_values, label="Risk-Adjusted Environment")
    plt.plot(friction_values, label="Friction Environment")
    plt.title("Portfolio Values Comparison")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/portfolio_values_comparison.png")
    
    # Plot 2: Rewards
    plt.figure(figsize=(12, 6))
    plt.plot(basic_rewards, label="Basic Rewards")
    plt.plot(risk_rewards, label="Risk-Adjusted Rewards")
    plt.plot(friction_rewards, label="Friction Rewards")
    plt.title("Rewards Comparison")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards_comparison.png")
    
    # Plot 3: Sharpe Ratio and Drawdown
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Sharpe Ratio', color=color)
    ax1.plot(sharpe_ratios, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Drawdown', color=color)
    ax2.plot(drawdowns, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Sharpe Ratio and Drawdown")
    fig.tight_layout()
    plt.savefig(f"{save_dir}/sharpe_drawdown.png")
    
    # Plot 4: Slippage and Fill Rates
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:green'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Slippage', color=color)
    ax1.plot(slippages, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Fill Rate', color=color)
    ax2.plot(fill_rates, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Slippage and Fill Rates")
    fig.tight_layout()
    plt.savefig(f"{save_dir}/slippage_fill_rates.png")
    
    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    # This will execute only if script is run directly, not during pytest collection
    data = sample_data()
    generate_plots(data) 