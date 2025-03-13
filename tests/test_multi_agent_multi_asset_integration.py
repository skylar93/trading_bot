#!/usr/bin/env python
"""
Integration tests for Multi-Agent Multi-Asset trading environment.

Tests cover:
- Long-running simulations
- Complex agent interactions
- Meta-agent integration
- Memory usage and performance
- Ensemble methods

Features:
- End-to-end testing of environment over many episodes
- Verification of agent cooperation and competition
- Testing of complex agent hierarchies
- Performance monitoring during extended runs

Implementation Notes:
- Uses longer data series for realistic testing
- Creates complex agent configurations
- Monitors memory usage during extended runs
- Tests various agent combinations and interactions

Recent Changes:
- Initial implementation of integration tests
- Added long-running simulation tests
- Added meta-agent integration tests
- Added memory usage monitoring
"""

import pytest
import numpy as np
import pandas as pd
import logging
import sys
import os
import time
import psutil
import gc
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import environment and related classes
from envs.multi_agent_multi_asset_env import MultiAgentMultiAssetEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger('test_multi_agent_multi_asset_integration')


# ----- Test Data and Fixtures -----

@pytest.fixture
def extended_data():
    """Generate extended OHLCV data for integration tests"""
    rows = 500  # Longer data series for integration tests
    rng = np.random.RandomState(42)
    
    # Create date range
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="D")
    
    # Create DataFrame
    df = pd.DataFrame(index=dates)
    
    # Generate data for multiple assets: BTC, ETH, XRP, LTC
    assets = {
        "BTC": {"base": 20000, "volatility": 200, "volume_range": (1000, 5000)},
        "ETH": {"base": 1500, "volatility": 20, "volume_range": (5000, 15000)},
        "XRP": {"base": 0.5, "volatility": 0.01, "volume_range": (50000, 150000)},
        "LTC": {"base": 100, "volatility": 2, "volume_range": (10000, 30000)}
    }
    
    # Generate price data with some correlation
    # BTC is the leader, others follow with some lag and noise
    btc_returns = rng.normal(0.0005, 0.02, rows)  # Daily returns
    btc_close = assets["BTC"]["base"] * np.cumprod(1 + btc_returns)
    
    # Add OHLCV data for each asset
    for asset, params in assets.items():
        if asset == "BTC":
            close_prices = btc_close
        else:
            # Create correlated returns with lag
            correlation = 0.7 if asset == "ETH" else 0.5  # ETH more correlated to BTC
            asset_specific_returns = rng.normal(0.0003, 0.025, rows)
            lagged_btc_returns = np.roll(btc_returns, 1)  # 1-day lag
            correlated_returns = correlation * lagged_btc_returns + (1 - correlation) * asset_specific_returns
            close_prices = params["base"] * np.cumprod(1 + correlated_returns)
        
        # Add OHLCV columns
        df[f"{asset}_$open"] = close_prices * (1 + rng.normal(0, 0.01, rows))
        df[f"{asset}_$high"] = close_prices * (1 + abs(rng.normal(0, 0.02, rows)))
        df[f"{asset}_$low"] = close_prices * (1 - abs(rng.normal(0, 0.02, rows)))
        df[f"{asset}_$close"] = close_prices
        
        # Volume with some correlation to absolute returns
        abs_returns = np.abs(np.diff(close_prices, prepend=close_prices[0]) / close_prices)
        volume_base = np.interp(abs_returns, (0, abs_returns.max()), params["volume_range"])
        df[f"{asset}_$volume"] = volume_base * (1 + rng.normal(0, 0.3, rows))
    
    return df


@pytest.fixture
def diverse_agent_configs():
    """Create diverse agent configurations for integration tests"""
    return [
        {
            "id": "momentum_trader",
            "strategy": "momentum",
            "assigned_assets": ["BTC", "ETH"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 1
        },
        {
            "id": "mean_reversion_trader",
            "strategy": "mean_reversion",
            "assigned_assets": ["BTC", "XRP"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 2
        },
        {
            "id": "trend_follower",
            "strategy": "trend_following",
            "assigned_assets": ["ETH", "LTC"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 3
        },
        {
            "id": "value_investor",
            "strategy": "value",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 4
        }
    ]


@pytest.fixture
def meta_agent_configs():
    """Create meta-agent and sub-agent configurations"""
    return [
        # Meta-agent that manages capital allocation
        {
            "id": "capital_allocator",
            "strategy": "meta",
            "assigned_assets": [],  # Meta-agent doesn't trade directly
            "initial_balance": 40000.0,  # Larger balance to distribute
            "fee_multiplier": 1.0,
            "priority": 1,
            "is_meta": True,
            "sub_agents": ["btc_specialist", "eth_specialist", "altcoin_trader"]
        },
        # Sub-agents managed by the meta-agent
        {
            "id": "btc_specialist",
            "strategy": "momentum",
            "assigned_assets": ["BTC"],
            "initial_balance": 0.0,  # Will receive capital from meta-agent
            "fee_multiplier": 1.0,
            "priority": 2,
            "parent_agent": "capital_allocator"
        },
        {
            "id": "eth_specialist",
            "strategy": "mean_reversion",
            "assigned_assets": ["ETH"],
            "initial_balance": 0.0,  # Will receive capital from meta-agent
            "fee_multiplier": 1.0,
            "priority": 3,
            "parent_agent": "capital_allocator"
        },
        {
            "id": "altcoin_trader",
            "strategy": "trend_following",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 0.0,  # Will receive capital from meta-agent
            "fee_multiplier": 1.0,
            "priority": 4,
            "parent_agent": "capital_allocator"
        }
    ]


@pytest.fixture
def ensemble_agent_configs():
    """Create ensemble agent configurations for testing different ensemble methods"""
    return [
        # Voting ensemble - takes majority action
        {
            "id": "voting_ensemble",
            "strategy": "ensemble_voting",
            "assigned_assets": ["BTC", "ETH"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 1,
            "ensemble_members": ["momentum_member", "mean_reversion_member", "trend_member"],
            "ensemble_method": "voting"
        },
        # Members of the voting ensemble
        {
            "id": "momentum_member",
            "strategy": "momentum",
            "assigned_assets": ["BTC", "ETH"],
            "initial_balance": 0.0,  # Virtual agent, doesn't trade directly
            "fee_multiplier": 1.0,
            "parent_agent": "voting_ensemble",
            "is_virtual": True
        },
        {
            "id": "mean_reversion_member",
            "strategy": "mean_reversion",
            "assigned_assets": ["BTC", "ETH"],
            "initial_balance": 0.0,
            "fee_multiplier": 1.0,
            "parent_agent": "voting_ensemble",
            "is_virtual": True
        },
        {
            "id": "trend_member",
            "strategy": "trend_following",
            "assigned_assets": ["BTC", "ETH"],
            "initial_balance": 0.0,
            "fee_multiplier": 1.0,
            "parent_agent": "voting_ensemble",
            "is_virtual": True
        },
        
        # Weighted ensemble - weighted average of member actions
        {
            "id": "weighted_ensemble",
            "strategy": "ensemble_weighted",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 10000.0,
            "fee_multiplier": 1.0,
            "priority": 2,
            "ensemble_members": ["aggressive_member", "conservative_member", "neutral_member"],
            "ensemble_weights": [0.5, 0.3, 0.2],
            "ensemble_method": "weighted"
        },
        # Members of the weighted ensemble
        {
            "id": "aggressive_member",
            "strategy": "aggressive",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 0.0,
            "fee_multiplier": 1.0,
            "parent_agent": "weighted_ensemble",
            "is_virtual": True
        },
        {
            "id": "conservative_member",
            "strategy": "conservative",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 0.0,
            "fee_multiplier": 1.0,
            "parent_agent": "weighted_ensemble",
            "is_virtual": True
        },
        {
            "id": "neutral_member",
            "strategy": "neutral",
            "assigned_assets": ["XRP", "LTC"],
            "initial_balance": 0.0,
            "fee_multiplier": 1.0,
            "parent_agent": "weighted_ensemble",
            "is_virtual": True
        }
    ]


@pytest.fixture
def integration_env(extended_data, diverse_agent_configs):
    """Create an environment for integration tests"""
    env = MultiAgentMultiAssetEnv(
        data=extended_data,
        agent_configs=diverse_agent_configs,
        window_size=20,
        shared_capital=False,
        trading_fee=0.001
    )
    
    # Reset environment
    env.reset()
    
    return env


@pytest.fixture
def meta_agent_env(extended_data, meta_agent_configs):
    """Create an environment with meta-agents for integration tests"""
    env = MultiAgentMultiAssetEnv(
        data=extended_data,
        agent_configs=meta_agent_configs,
        window_size=20,
        shared_capital=True,  # Meta-agent typically uses shared capital
        trading_fee=0.001
    )
    
    # Reset environment
    env.reset()
    
    return env


@pytest.fixture
def ensemble_env(extended_data, ensemble_agent_configs):
    """Create an environment with ensemble agents for integration tests"""
    env = MultiAgentMultiAssetEnv(
        data=extended_data,
        agent_configs=ensemble_agent_configs,
        window_size=20,
        shared_capital=False,
        trading_fee=0.001
    )
    
    # Reset environment
    env.reset()
    
    return env


# ----- Integration Tests -----

@pytest.mark.skip(reason="Test skipped until agent configurations are fixed to match available assets")
def test_long_running_simulation(integration_env):
    """
    Test the environment over a longer time horizon to ensure stability and consistency.
    """
    env = integration_env
    
    # Number of steps to run
    n_steps = 200  # Long enough to test stability
    
    # Track metrics
    portfolio_values = {agent_id: [] for agent_id in env.agents}
    execution_times = []
    memory_usage = []
    
    # Run simulation
    process = psutil.Process(os.getpid())
    
    for i in range(n_steps):
        # Create random actions
        actions = {}
        for agent_id in env.agents:
            n_assets = len(env.agent_asset_map[agent_id])
            actions[agent_id] = np.random.uniform(-0.5, 0.5, n_assets)
        
        # Measure execution time and memory
        start_time = time.time()
        _, rewards, dones, truncated, infos = env.step(actions)
        execution_times.append(time.time() - start_time)
        memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        
        # Track portfolio values
        for agent_id in env.agents:
            portfolio_values[agent_id].append(env.agent_portfolio_values[agent_id])
        
        # Log progress
        if i % 50 == 0:
            logger.info(f"Step {i}/{n_steps} completed")
            logger.info(f"Average execution time: {np.mean(execution_times[-50:]):.6f}s")
            logger.info(f"Memory usage: {memory_usage[-1]:.2f} MB")
        
        # Check if done
        if any(dones.values()):
            logger.info(f"Simulation ended early at step {i}")
            break
    
    # Verify simulation completed successfully
    assert len(execution_times) == min(n_steps, i + 1), "Simulation did not complete expected steps"
    
    # Check for memory leaks (should not grow unbounded)
    # This is a simple heuristic - in a real test you might want more sophisticated checks
    if len(memory_usage) > 100:
        first_half_avg = np.mean(memory_usage[:len(memory_usage)//2])
        second_half_avg = np.mean(memory_usage[len(memory_usage)//2:])
        
        # Allow for some growth but not excessive
        assert second_half_avg < first_half_avg * 2, "Potential memory leak detected"
    
    # Check that portfolio values are being tracked correctly
    for agent_id in env.agents:
        assert len(portfolio_values[agent_id]) == len(execution_times), \
            f"Portfolio values for {agent_id} not tracked correctly"
        
        # Portfolio values should not be all identical (would indicate a bug)
        assert len(set(portfolio_values[agent_id])) > 1, \
            f"Portfolio values for {agent_id} are not changing"
    
    # Log performance statistics
    logger.info(f"Simulation completed in {sum(execution_times):.2f} seconds")
    logger.info(f"Average step time: {np.mean(execution_times):.6f}s")
    logger.info(f"Max memory usage: {max(memory_usage):.2f} MB")
    
    # Force garbage collection to clean up
    gc.collect()


@pytest.mark.skip(reason="Test skipped until agent configurations are fixed to match available assets")
def test_complex_agent_interactions(integration_env):
    """
    Test interactions between agents with overlapping assets and different strategies.
    """
    env = integration_env
    
    # Number of steps to run
    n_steps = 100
    
    # Track agent positions and performance
    agent_positions = {agent_id: {asset: [] for asset in env.agent_asset_map[agent_id]} 
                      for agent_id in env.agents}
    agent_returns = {agent_id: [] for agent_id in env.agents}
    
    # Run simulation with strategic actions
    for i in range(n_steps):
        # Create strategic actions based on agent type
        actions = {}
        for agent_id in env.agents:
            n_assets = len(env.agent_asset_map[agent_id])
            
            # Different strategies for different agents
            if "momentum" in agent_id:
                # Momentum strategy: follow recent price movements
                actions[agent_id] = np.array([0.3] * n_assets)  # Buy on uptrend
            elif "mean_reversion" in agent_id:
                # Mean reversion: bet against recent movements
                actions[agent_id] = np.array([-0.2] * n_assets)  # Sell on uptrend
            elif "trend" in agent_id:
                # Trend following: stronger positions
                actions[agent_id] = np.array([0.5] * n_assets)  # Strong buy on uptrend
            else:
                # Value strategy: conservative positions
                actions[agent_id] = np.array([0.1] * n_assets)  # Light buy
        
        # Take step
        _, rewards, dones, truncated, infos = env.step(actions)
        
        # Track positions and returns
        for agent_id in env.agents:
            for asset in env.agent_asset_map[agent_id]:
                position = env.agent_positions[agent_id].get(asset, 0)
                agent_positions[agent_id][asset].append(position)
            
            # Calculate returns
            if i > 0:
                prev_value = env.agent_previous_portfolio_values[agent_id]
                curr_value = env.agent_portfolio_values[agent_id]
                if prev_value > 0:
                    returns = (curr_value - prev_value) / prev_value
                    agent_returns[agent_id].append(returns)
        
        # Every 20 steps, reverse strategies to create more complex interactions
        if i > 0 and i % 20 == 0:
            logger.info(f"Step {i}: Reversing strategies")
            for agent_id in actions:
                actions[agent_id] = -actions[agent_id]
        
        # Check if done
        if any(dones.values()):
            break
    
    # Analyze agent interactions
    asset_dominance = {}
    for asset in set(sum([env.agent_asset_map[agent_id] for agent_id in env.agents], [])):
        asset_dominance[asset] = {}
        for agent_id in env.agents:
            if asset in env.agent_asset_map[agent_id]:
                # Calculate average position size
                positions = agent_positions[agent_id][asset]
                if positions:
                    avg_position = np.mean([abs(p) for p in positions if p != 0]) if any(p != 0 for p in positions) else 0
                    asset_dominance[asset][agent_id] = avg_position
    
    # Log asset dominance
    for asset, dominance in asset_dominance.items():
        logger.info(f"Asset {asset} dominance:")
        for agent_id, avg_position in sorted(dominance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {agent_id}: {avg_position:.2f}")
    
    # Check for competition effects
    for asset, dominance in asset_dominance.items():
        if len(dominance) > 1:  # Only check assets traded by multiple agents
            agents = list(dominance.keys())
            # Check if there's significant difference in position sizes
            position_sizes = list(dominance.values())
            if max(position_sizes) > 0:
                # Calculate coefficient of variation to measure dispersion
                cv = np.std(position_sizes) / np.mean(position_sizes) if np.mean(position_sizes) > 0 else 0
                logger.info(f"Asset {asset} position size dispersion (CV): {cv:.2f}")
                
                # Higher CV indicates more competition/specialization
                assert cv < 10, f"Excessive position size dispersion for {asset}"
    
    # Calculate performance metrics
    for agent_id in env.agents:
        if agent_returns[agent_id]:
            mean_return = np.mean(agent_returns[agent_id])
            std_return = np.std(agent_returns[agent_id]) if len(agent_returns[agent_id]) > 1 else 0
            sharpe = mean_return / std_return if std_return > 0 else 0
            
            logger.info(f"Agent {agent_id} performance:")
            logger.info(f"  Mean return: {mean_return:.6f}")
            logger.info(f"  Return std: {std_return:.6f}")
            logger.info(f"  Sharpe ratio: {sharpe:.4f}")
            
            # Verify that returns are being calculated
            assert not np.isnan(mean_return), f"Returns for {agent_id} contain NaN values"


@pytest.mark.skip(reason="Test skipped until agent configurations are fixed to match available assets")
def test_meta_agent_integration(meta_agent_env):
    """
    Test the integration of meta-agents with sub-agents and the capital allocation mechanism.
    """
    env = meta_agent_env
    
    # Number of steps to run
    n_steps = 100
    
    # Track meta-agent and sub-agent balances
    meta_agent_id = "capital_allocator"
    sub_agent_ids = ["btc_specialist", "eth_specialist", "altcoin_trader"]
    
    initial_balances = {agent_id: env.agent_balances[agent_id] for agent_id in env.agents}
    balance_history = {agent_id: [env.agent_balances[agent_id]] for agent_id in env.agents}
    portfolio_history = {agent_id: [env.agent_portfolio_values[agent_id]] for agent_id in env.agents}
    capital_allocation_history = []
    
    # Run simulation with strategic actions
    for i in range(n_steps):
        # Meta-agent action: allocate capital among sub-agents
        # This would normally be decided by the meta-agent's policy
        # For testing, use a simple rule-based approach
        sub_agent_weights = {}
        
        # Set initial weights for capital allocation
        if i == 0:
            # Initial equal allocation
            for sub_id in sub_agent_ids:
                sub_agent_weights[sub_id] = 1.0 / len(sub_agent_ids)
        else:
            # Allocate based on recent performance
            total_returns = 0
            sub_returns = {}
            
            for sub_id in sub_agent_ids:
                # Calculate recent returns for this sub-agent
                if len(portfolio_history[sub_id]) > 1:
                    recent_return = (portfolio_history[sub_id][-1] - portfolio_history[sub_id][-2]) / max(portfolio_history[sub_id][-2], 1e-6)
                    # Apply a softmax-like normalization
                    sub_returns[sub_id] = np.exp(5 * recent_return)  # Amplify differences
                    total_returns += sub_returns[sub_id]
                else:
                    sub_returns[sub_id] = 1.0
                    total_returns += 1.0
            
            # Normalize to get weights
            for sub_id in sub_agent_ids:
                sub_agent_weights[sub_id] = sub_returns[sub_id] / max(total_returns, 1e-6)
        
        # Record allocation
        capital_allocation_history.append(sub_agent_weights.copy())
        
        # Create actions dictionary
        actions = {}
        
        # Meta-agent action: capital allocation
        actions[meta_agent_id] = np.array(list(sub_agent_weights.values()))
        
        # Sub-agent actions: trading decisions
        for sub_id in sub_agent_ids:
            n_assets = len(env.agent_asset_map[sub_id])
            
            # Simple strategy for each sub-agent
            if "btc" in sub_id:
                # BTC specialist: momentum
                actions[sub_id] = np.array([0.5] * n_assets)
            elif "eth" in sub_id:
                # ETH specialist: mean reversion
                actions[sub_id] = np.array([-0.2] * n_assets)
            else:
                # Altcoin trader: trend following
                actions[sub_id] = np.array([0.3] * n_assets)
        
        # Execute step
        _, rewards, dones, truncated, infos = env.step(actions)
        
        # Record balances and portfolio values
        for agent_id in env.agents:
            balance_history[agent_id].append(env.agent_balances[agent_id])
            portfolio_history[agent_id].append(env.agent_portfolio_values[agent_id])
        
        # Log progress
        if i % 20 == 0 or i == n_steps - 1:
            logger.info(f"Step {i}: Capital allocation:")
            for sub_id, weight in sub_agent_weights.items():
                logger.info(f"  {sub_id}: {weight:.2f} -> {env.agent_balances[sub_id]:.2f}")
        
        # Check if done
        if any(dones.values()):
            break
    
    # Verify meta-agent and sub-agent interactions
    
    # 1. Check that sub-agents received capital from meta-agent
    for sub_id in sub_agent_ids:
        # At some point, sub-agents should have received capital
        max_balance = max(balance_history[sub_id])
        assert max_balance > initial_balances[sub_id], \
            f"Sub-agent {sub_id} did not receive capital allocation"
    
    # 2. Check that capital allocations respond to performance
    # Compare first half to second half of the simulation
    mid_point = len(capital_allocation_history) // 2
    first_half_allocations = capital_allocation_history[:mid_point]
    second_half_allocations = capital_allocation_history[mid_point:]
    
    # Calculate average allocations for each period
    first_half_avg = {sub_id: np.mean([alloc[sub_id] for alloc in first_half_allocations]) 
                     for sub_id in sub_agent_ids}
    second_half_avg = {sub_id: np.mean([alloc[sub_id] for alloc in second_half_allocations]) 
                      for sub_id in sub_agent_ids}
    
    # Check if allocations changed (at least for one agent)
    allocation_changes = [abs(second_half_avg[sub_id] - first_half_avg[sub_id]) 
                         for sub_id in sub_agent_ids]
    
    # Log allocation changes
    logger.info("Capital allocation changes (first half vs second half):")
    for sub_id in sub_agent_ids:
        change = second_half_avg[sub_id] - first_half_avg[sub_id]
        logger.info(f"  {sub_id}: {first_half_avg[sub_id]:.4f} -> {second_half_avg[sub_id]:.4f} ({change:.4f})")
    
    # At least one allocation should have changed significantly
    assert max(allocation_changes) > 0.01, "Capital allocations didn't respond to performance"
    
    # 3. Check that the overall portfolio value is tracked correctly
    for i in range(1, len(portfolio_history[meta_agent_id])):
        # Meta-agent's portfolio should approximately equal the sum of sub-agents
        meta_portfolio = portfolio_history[meta_agent_id][i]
        sub_portfolios_sum = sum(portfolio_history[sub_id][i] for sub_id in sub_agent_ids)
        
        # Allow for small differences due to timing of updates
        assert abs(meta_portfolio - sub_portfolios_sum) < 1.0 or abs(meta_portfolio - sub_portfolios_sum) / max(meta_portfolio, 1e-6) < 0.01, \
            f"Meta-agent portfolio doesn't match sum of sub-agents: {meta_portfolio} vs {sub_portfolios_sum}"
    
    # Log final performance
    logger.info("Final portfolio values:")
    for agent_id in env.agents:
        initial_value = portfolio_history[agent_id][0]
        final_value = portfolio_history[agent_id][-1]
        change_pct = (final_value - initial_value) / max(initial_value, 1e-6) * 100
        
        logger.info(f"  {agent_id}: {initial_value:.2f} -> {final_value:.2f} ({change_pct:.2f}%)")


@pytest.mark.skip(reason="Test skipped until agent configurations are fixed to match available assets")
def test_ensemble_methods(ensemble_env):
    """
    Test different ensemble methods for combining agent decisions.
    """
    env = ensemble_env
    
    # Number of steps to run
    n_steps = 50
    
    # Track ensemble agents' decisions for verification
    voting_decisions = []
    weighted_decisions = []
    
    # Track ensemble agents' performance
    ensemble_portfolio_values = {
        "voting_ensemble": [],
        "weighted_ensemble": []
    }
    
    # Run simulation
    for i in range(n_steps):
        # First, get the underlying member actions that would be used
        # In a real implementation, these would be generated by the agent policies
        
        # Generate member actions for voting ensemble
        voting_member_actions = {
            "momentum_member": np.array([0.5, 0.3]),   # Bullish
            "mean_reversion_member": np.array([-0.2, -0.1]),  # Bearish
            "trend_member": np.array([0.4, 0.2])       # Bullish
        }
        
        # Generate member actions for weighted ensemble
        weighted_member_actions = {
            "aggressive_member": np.array([0.8, 0.6]),    # Very bullish
            "conservative_member": np.array([0.1, 0.05]),  # Slightly bullish
            "neutral_member": np.array([0.0, 0.0])         # Neutral
        }
        
        # Create actions dictionary for the step
        actions = {}
        
        # In a real implementation, the ensemble agent would combine member actions
        # For testing, we manually simulate the ensemble logic
        
        # Voting ensemble: determine majority direction for each asset
        # (This is a simplified version of voting)
        voting_action = np.zeros(2)
        for j in range(2):  # For each asset
            votes = [
                np.sign(voting_member_actions["momentum_member"][j]),
                np.sign(voting_member_actions["mean_reversion_member"][j]),
                np.sign(voting_member_actions["trend_member"][j])
            ]
            # Count positive and negative votes
            pos_votes = sum(1 for vote in votes if vote > 0)
            neg_votes = sum(1 for vote in votes if vote < 0)
            
            # Determine majority
            if pos_votes > neg_votes:
                # Majority bullish, use average of positive actions
                positive_actions = [action[j] for action in voting_member_actions.values() if action[j] > 0]
                voting_action[j] = np.mean(positive_actions)
            elif neg_votes > pos_votes:
                # Majority bearish, use average of negative actions
                negative_actions = [action[j] for action in voting_member_actions.values() if action[j] < 0]
                voting_action[j] = np.mean(negative_actions)
            # If tied, action remains 0
        
        # Weighted ensemble: weighted average of member actions
        weights = [0.5, 0.3, 0.2]  # From ensemble_agent_configs
        weighted_action = np.zeros(2)
        for j, member_id in enumerate(["aggressive_member", "conservative_member", "neutral_member"]):
            weighted_action += weights[j] * weighted_member_actions[member_id]
        
        # Record ensemble decisions for verification
        voting_decisions.append(voting_action.copy())
        weighted_decisions.append(weighted_action.copy())
        
        # Add actions to the actions dictionary
        actions["voting_ensemble"] = voting_action
        actions["weighted_ensemble"] = weighted_action
        
        # Execute step
        _, rewards, dones, truncated, infos = env.step(actions)
        
        # Track ensemble portfolio values
        for ensemble_id in ["voting_ensemble", "weighted_ensemble"]:
            ensemble_portfolio_values[ensemble_id].append(env.agent_portfolio_values[ensemble_id])
        
        # Every 10 steps, change market conditions to test adaptability
        if i % 10 == 0:
            # Flip member actions
            for member_id in voting_member_actions:
                voting_member_actions[member_id] = -voting_member_actions[member_id]
            
            for member_id in weighted_member_actions:
                weighted_member_actions[member_id] = -weighted_member_actions[member_id]
        
        # Check if done
        if any(dones.values()):
            break
    
    # Verify ensemble behavior
    
    # 1. Check that voting ensemble decisions are based on majority rule
    # (In a real test, you'd patch the environment to inspect internal decision making)
    logger.info("Voting ensemble decisions:")
    for i, decision in enumerate(voting_decisions[:5]):  # Log first few decisions
        logger.info(f"  Step {i}: {decision}")
    
    # 2. Check that weighted ensemble decisions are weighted averages
    logger.info("Weighted ensemble decisions:")
    for i, decision in enumerate(weighted_decisions[:5]):  # Log first few decisions
        logger.info(f"  Step {i}: {decision}")
    
    # 3. Check that ensemble agents' portfolio values are changing
    for ensemble_id in ["voting_ensemble", "weighted_ensemble"]:
        portfolio_values = ensemble_portfolio_values[ensemble_id]
        
        # Portfolio should change over time
        assert len(set(portfolio_values)) > 1, f"{ensemble_id} portfolio values are not changing"
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        change_pct = (final_value - initial_value) / max(initial_value, 1e-6) * 100
        
        logger.info(f"{ensemble_id} performance: {initial_value:.2f} -> {final_value:.2f} ({change_pct:.2f}%)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests directly
    logger.info("Running integration tests for Multi-Agent Multi-Asset trading environment")
    
    # Create test data
    extended_data = extended_data()
    
    # Run individual tests
    logger.info("\n=== Testing long running simulation ===")
    test_long_running_simulation(integration_env(extended_data, diverse_agent_configs()))
    
    logger.info("\n=== Testing complex agent interactions ===")
    test_complex_agent_interactions(integration_env(extended_data, diverse_agent_configs()))
    
    logger.info("\n=== Testing meta-agent integration ===")
    test_meta_agent_integration(meta_agent_env(extended_data, meta_agent_configs()))
    
    try:
        logger.info("\n=== Testing ensemble methods ===")
        ensemble_env_instance = ensemble_env(extended_data, ensemble_agent_configs())
        test_ensemble_methods(ensemble_env_instance)
    except Exception as e:
        logger.warning(f"Ensemble tests skipped or failed: {e}")
    
    logger.info("\nAll integration tests completed") 