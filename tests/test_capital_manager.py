#!/usr/bin/env python
"""Test script for capital management classes."""

import os
import sys
import unittest
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from envs.capital_manager import (
    CapitalManager,
    MultiAssetCapitalManager,
    MultiAgentCapitalManager
)


class TestCapitalManager(unittest.TestCase):
    """Test class for the base CapitalManager class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.initial_capital = 10000.0
        self.assets = ["BTC", "ETH", "AAPL"]
        self.allocation_weights = {"BTC": 0.5, "ETH": 0.3, "AAPL": 0.2}
        
        # Create shared mode manager
        self.shared_manager = CapitalManager(
            initial_capital=self.initial_capital,
            mode="shared",
            assets=self.assets,
            allocation_weights=self.allocation_weights,
            max_leverage=1.5,
            drawdown_limit=0.2,
            auto_rebalance=False
        )
        
        # Create isolated mode manager
        self.isolated_manager = CapitalManager(
            initial_capital=self.initial_capital,
            mode="isolated",
            assets=self.assets,
            allocation_weights=self.allocation_weights,
            max_leverage=1.5,
            drawdown_limit=0.2,
            auto_rebalance=False
        )
    
    def test_initialization(self):
        """Test proper initialization of capital managers."""
        # Test shared mode
        self.assertEqual(self.shared_manager.initial_capital, self.initial_capital)
        self.assertEqual(self.shared_manager.total_capital, self.initial_capital)
        self.assertEqual(self.shared_manager.mode, "shared")
        self.assertEqual(self.shared_manager.max_leverage, 1.5)
        self.assertEqual(self.shared_manager.drawdown_limit, 0.2)
        self.assertFalse(self.shared_manager.auto_rebalance)
        
        # Check allocation
        self.assertIn("shared", self.shared_manager.allocated_capital)
        self.assertEqual(self.shared_manager.allocated_capital["shared"], self.initial_capital)
        
        # Used capital should be initialized to zero
        for asset in self.assets:
            self.assertEqual(self.shared_manager.used_capital.get(asset, 0.0), 0.0)
        
        # Test isolated mode
        self.assertEqual(self.isolated_manager.mode, "isolated")
        
        # Check allocation
        for asset in self.assets:
            self.assertIn(asset, self.isolated_manager.allocated_capital)
            expected_capital = self.initial_capital * self.allocation_weights[asset]
            self.assertAlmostEqual(
                self.isolated_manager.allocated_capital[asset],
                expected_capital,
                delta=0.01
            )
    
    def test_allocation_weights_normalization(self):
        """Test that allocation weights are properly normalized."""
        # Create manager with non-normalized weights
        non_normalized_weights = {"BTC": 5, "ETH": 3, "AAPL": 2}  # Sum = 10
        
        manager = CapitalManager(
            initial_capital=self.initial_capital,
            assets=self.assets,
            allocation_weights=non_normalized_weights
        )
        
        # Weights should be normalized
        self.assertAlmostEqual(manager.allocation_weights["BTC"], 0.5, delta=0.001)
        self.assertAlmostEqual(manager.allocation_weights["ETH"], 0.3, delta=0.001)
        self.assertAlmostEqual(manager.allocation_weights["AAPL"], 0.2, delta=0.001)
    
    def test_get_available_capital_shared(self):
        """Test getting available capital in shared mode."""
        manager = self.shared_manager
        
        # Initially, all capital is available
        for asset in self.assets:
            self.assertEqual(manager.get_available_capital(asset), self.initial_capital)
        
        # Allocate some capital
        manager.used_capital["BTC"] = 2000.0
        manager.used_capital["ETH"] = 3000.0
        
        # Available capital should be reduced for all assets
        for asset in self.assets:
            self.assertEqual(manager.get_available_capital(asset), 5000.0)
    
    def test_get_available_capital_isolated(self):
        """Test getting available capital in isolated mode."""
        manager = self.isolated_manager
        
        # Initially, capital is allocated according to weights
        for asset in self.assets:
            expected_capital = self.initial_capital * self.allocation_weights[asset]
            self.assertAlmostEqual(
                manager.get_available_capital(asset),
                expected_capital,
                delta=0.01
            )
        
        # Allocate some capital for BTC
        manager.used_capital["BTC"] = 2000.0
        
        # Only BTC's available capital should be reduced
        expected_btc_available = self.initial_capital * self.allocation_weights["BTC"] - 2000.0
        self.assertAlmostEqual(
            manager.get_available_capital("BTC"),
            expected_btc_available,
            delta=0.01
        )
        
        # ETH's available capital should remain the same
        expected_eth_available = self.initial_capital * self.allocation_weights["ETH"]
        self.assertAlmostEqual(
            manager.get_available_capital("ETH"),
            expected_eth_available,
            delta=0.01
        )
    
    def test_allocate_capital(self):
        """Test allocating capital to assets."""
        # Test shared mode
        shared_manager = self.shared_manager
        
        # Allocate capital to BTC
        allocated = shared_manager.allocate_capital("BTC", 3000.0)
        self.assertEqual(allocated, 3000.0)
        self.assertEqual(shared_manager.used_capital["BTC"], 3000.0)
        
        # All assets should now have 7000 available
        for asset in self.assets:
            self.assertEqual(shared_manager.get_available_capital(asset), 7000.0)
        
        # Allocate more than available
        allocated = shared_manager.allocate_capital("ETH", 8000.0)
        self.assertEqual(allocated, 7000.0)  # Should be capped at available capital
        self.assertEqual(shared_manager.used_capital["ETH"], 7000.0)
        
        # No capital should be available now
        for asset in self.assets:
            self.assertEqual(shared_manager.get_available_capital(asset), 0.0)
        
        # Test isolated mode
        isolated_manager = self.isolated_manager
        
        # Allocate capital to BTC
        btc_available = isolated_manager.get_available_capital("BTC")
        allocated = isolated_manager.allocate_capital("BTC", btc_available + 1000.0)
        self.assertEqual(allocated, btc_available)  # Should be capped at available capital
        self.assertEqual(isolated_manager.used_capital["BTC"], btc_available)
        
        # BTC should have no available capital, but ETH should still have its allocation
        self.assertEqual(isolated_manager.get_available_capital("BTC"), 0.0)
        self.assertGreater(isolated_manager.get_available_capital("ETH"), 0.0)
    
    def test_release_capital(self):
        """Test releasing previously allocated capital."""
        # Test shared mode
        shared_manager = self.shared_manager
        
        # First allocate capital
        shared_manager.allocate_capital("BTC", 3000.0)
        shared_manager.allocate_capital("ETH", 2000.0)
        
        # Release some capital
        released = shared_manager.release_capital("BTC", 1000.0)
        self.assertEqual(released, 1000.0)
        self.assertEqual(shared_manager.used_capital["BTC"], 2000.0)
        
        # Available capital should increase
        self.assertEqual(shared_manager.get_available_capital("ETH"), 6000.0)
        
        # Try to release more than used
        released = shared_manager.release_capital("ETH", 3000.0)
        self.assertEqual(released, 2000.0)  # Should be capped at used capital
        self.assertEqual(shared_manager.used_capital["ETH"], 0.0)
    
    def test_update_capital(self):
        """Test updating capital based on trading results."""
        # Test shared mode
        shared_manager = self.shared_manager
        
        # Update capital (profit scenario)
        capital_changes = {"BTC": 500.0, "ETH": 300.0, "AAPL": 200.0}
        net_change = shared_manager.update_capital(capital_changes)
        
        # Check results
        self.assertEqual(net_change, 1000.0)
        self.assertEqual(shared_manager.total_capital, self.initial_capital + 1000.0)
        self.assertEqual(shared_manager.allocated_capital["shared"], self.initial_capital + 1000.0)
        
        # Test isolated mode
        isolated_manager = self.isolated_manager
        
        # Update capital (loss scenario)
        capital_changes = {"BTC": -500.0, "ETH": -300.0, "AAPL": -200.0}
        net_change = isolated_manager.update_capital(capital_changes)
        
        # Check results
        self.assertEqual(net_change, -1000.0)
        self.assertEqual(isolated_manager.total_capital, self.initial_capital - 1000.0)
        
        # Each asset's allocation should be reduced by its loss
        expected_btc_allocation = (self.initial_capital * 0.5) - 500.0
        self.assertAlmostEqual(
            isolated_manager.allocated_capital["BTC"],
            expected_btc_allocation,
            delta=0.01
        )
    
    def test_drawdown_tracking(self):
        """Test tracking of drawdowns and peak capital."""
        manager = self.shared_manager
        
        # Initial state
        self.assertEqual(manager.peak_capital, self.initial_capital)
        self.assertEqual(manager.current_drawdown, 0.0)
        self.assertEqual(manager.max_drawdown, 0.0)
        
        # Update with profit, peak should increase
        manager.update_capital({"BTC": 1000.0})
        self.assertEqual(manager.peak_capital, self.initial_capital + 1000.0)
        self.assertEqual(manager.current_drawdown, 0.0)
        
        # Update with loss, drawdown should be tracked
        manager.update_capital({"BTC": -2000.0})
        expected_drawdown = 1.0 - (9000.0 / 11000.0)  # 1 - (current / peak)
        self.assertAlmostEqual(manager.current_drawdown, expected_drawdown, delta=0.001)
        self.assertAlmostEqual(manager.max_drawdown, expected_drawdown, delta=0.001)
        
        # Further gain, but still in drawdown
        manager.update_capital({"BTC": 500.0})
        expected_drawdown = 1.0 - (9500.0 / 11000.0)
        self.assertAlmostEqual(manager.current_drawdown, expected_drawdown, delta=0.001)
        
        # Return to peak, drawdown should be 0 again
        manager.update_capital({"BTC": 1500.0})
        self.assertEqual(manager.current_drawdown, 0.0)
        self.assertGreater(manager.max_drawdown, 0.0)  # Max drawdown should persist
    
    def test_exposure_reduction_on_drawdown(self):
        """Test that exposure is reduced when drawdown exceeds limit."""
        manager = self.shared_manager
        
        # Create a drawdown exceeding the limit
        initial_leverage = manager.max_leverage
        manager.update_capital({"BTC": 5000.0})  # First increase peak
        manager.update_capital({"BTC": -5000.0, "ETH": -1100.0})  # Then create drawdown
        
        # Calculate expected drawdown
        expected_drawdown = 1.0 - (8900.0 / 15000.0)  # > 0.2
        
        # Drawdown should exceed limit
        self.assertGreater(manager.current_drawdown, manager.drawdown_limit)
        
        # Max leverage should be reduced
        self.assertLess(manager.max_leverage, initial_leverage)
    
    def test_position_values_and_leverage(self):
        """Test tracking of position values and leverage calculation."""
        manager = self.shared_manager
        
        # Initial state
        self.assertEqual(manager.current_leverage, 0.0)
        
        # Update position values
        position_values = {"BTC": 3000.0, "ETH": 2000.0, "AAPL": 0.0}
        manager.update_position_values(position_values)
        
        # Check leverage calculation
        expected_leverage = 5000.0 / 10000.0
        self.assertEqual(manager.current_leverage, expected_leverage)
    
    def test_rebalance(self):
        """Test portfolio rebalancing functionality."""
        # Create manager with auto-rebalance enabled
        auto_rebalance_manager = CapitalManager(
            initial_capital=self.initial_capital,
            mode="isolated",
            assets=self.assets,
            allocation_weights=self.allocation_weights,
            auto_rebalance=True,
            rebalance_threshold=0.1
        )
        
        # Manual rebalance
        isolated_manager = self.isolated_manager
        
        # First modify allocations to deviate from target
        isolated_manager.allocated_capital["BTC"] = 2000.0  # Should be 5000
        isolated_manager.allocated_capital["ETH"] = 7000.0  # Should be 3000
        isolated_manager.allocated_capital["AAPL"] = 1000.0  # Should be 2000
        
        # Trigger rebalance
        adjustments = isolated_manager.rebalance()
        
        # Check adjustments
        # The actual values depend on the implementation; just verify direction and approximate magnitude
        self.assertGreater(adjustments["BTC"], 0)  # BTC should increase
        self.assertLess(adjustments["ETH"], 0)     # ETH should decrease
        self.assertGreater(adjustments["AAPL"], 0) # AAPL should increase
        
        # Check new allocations match target weights
        total_capital = sum(isolated_manager.allocated_capital.values())
        for asset in self.assets:
            expected_allocation = total_capital * self.allocation_weights[asset]
            self.assertAlmostEqual(
                isolated_manager.allocated_capital[asset], 
                expected_allocation, 
                delta=0.01
            )
    
    def test_get_allocation_status(self):
        """Test getting allocation status information."""
        manager = self.shared_manager
        
        # Update position values
        position_values = {"BTC": 3000.0, "ETH": 2000.0, "AAPL": 0.0}
        manager.update_position_values(position_values)
        
        # Get status
        status = manager.get_allocation_status()
        
        # Check key information
        self.assertEqual(status["total_capital"], self.initial_capital)
        self.assertEqual(status["mode"], "shared")
        self.assertEqual(status["current_leverage"], 0.5)
        self.assertEqual(status["shared_capital_pool"], self.initial_capital)
        self.assertEqual(status["position_values"], position_values)
    
    def test_reset(self):
        """Test resetting the capital manager."""
        manager = self.shared_manager
        
        # Modify state
        manager.total_capital = 12000.0
        manager.allocated_capital["shared"] = 12000.0
        manager.used_capital["BTC"] = 3000.0
        manager.update_position_values({"BTC": 3000.0, "ETH": 2000.0})
        manager.update_capital({"BTC": -1000.0})  # Create drawdown
        
        # Reset
        manager.reset()
        
        # Check reset state
        self.assertEqual(manager.total_capital, self.initial_capital)
        self.assertEqual(manager.allocated_capital["shared"], self.initial_capital)
        self.assertEqual(manager.used_capital["BTC"], 0.0)
        self.assertEqual(manager.position_values["BTC"], 0.0)
        self.assertEqual(manager.current_leverage, 0.0)
        self.assertEqual(manager.current_drawdown, 0.0)
        self.assertEqual(len(manager.returns_history), 0)


class MockMultiAssetEnv:
    """Mock environment for testing MultiAssetCapitalManager."""
    
    def __init__(self):
        self.initial_balance = 10000.0
        self.assets = ["BTC", "ETH"]
        self.positions = {"BTC": 0.1, "ETH": 1.0}
        self.prices = {"BTC": 30000.0, "ETH": 2000.0}
        self.portfolio_value = 12000.0
        self.current_weights = {"BTC": 0.25, "ETH": 0.167, "cash": 0.583}


class TestMultiAssetCapitalManager(unittest.TestCase):
    """Test class for MultiAssetCapitalManager."""
    
    def setUp(self):
        """Set up the test environment."""
        self.mock_env = MockMultiAssetEnv()
        
        # Create capital manager
        self.manager = MultiAssetCapitalManager(
            env=self.mock_env,
            mode="shared",
            allocation_weights={"BTC": 0.6, "ETH": 0.4},
            max_leverage=1.5
        )
    
    def test_initialization(self):
        """Test proper initialization of MultiAssetCapitalManager."""
        self.assertEqual(self.manager.env, self.mock_env)
        self.assertEqual(self.manager.initial_capital, self.mock_env.initial_balance)
        self.assertEqual(self.manager.assets, self.mock_env.assets)
        self.assertEqual(self.manager.allocation_weights["BTC"], 0.6)
    
    def test_update_from_env_state(self):
        """Test updating capital manager state from environment state."""
        # Update state
        self.manager.update_from_env_state()
        
        # Check position values
        expected_btc_value = self.mock_env.positions["BTC"] * self.mock_env.prices["BTC"]
        expected_eth_value = self.mock_env.positions["ETH"] * self.mock_env.prices["ETH"]
        
        self.assertEqual(self.manager.position_values["BTC"], expected_btc_value)
        self.assertEqual(self.manager.position_values["ETH"], expected_eth_value)
        
        # Check capital update
        capital_change = self.mock_env.portfolio_value - self.manager.initial_capital
        self.assertEqual(self.manager.total_capital, self.mock_env.portfolio_value)
    
    def test_check_capital_constraints(self):
        """Test checking capital constraints for position changes."""
        # Set up constraints
        self.manager.used_capital["BTC"] = 2000.0
        self.manager.used_capital["ETH"] = 3000.0
        
        # Check constraint for positive position change (buy)
        # Available capital = 10000 - 5000 = 5000
        # Max BTC we can buy = 5000 / 30000 = 0.1667 BTC
        result = self.manager.check_capital_constraints("BTC", 0.2)
        self.assertAlmostEqual(result, 5000.0 / 30000.0, delta=0.001)
        
        # Check constraint for negative position change (sell)
        # We can sell up to what we have
        result = self.manager.check_capital_constraints("BTC", -0.05)
        self.assertEqual(result, -0.05)  # No constraint on selling
    
    def test_allocate_for_position(self):
        """Test allocating capital for a position change."""
        # Allocate for buying BTC
        allocated = self.manager.allocate_for_position("BTC", 0.1)
        expected_allocation = 0.1 * self.mock_env.prices["BTC"]
        self.assertEqual(allocated, expected_allocation)
        
        # Check used capital was updated
        self.assertEqual(self.manager.used_capital["BTC"], expected_allocation)
    
    def test_get_max_position_size(self):
        """Test getting maximum position size based on available capital."""
        # Set up constraints
        self.manager.used_capital["BTC"] = 2000.0
        self.manager.used_capital["ETH"] = 3000.0
        
        # Check max position size
        # Available capital = 10000 - 5000 = 5000
        # Max BTC we can buy = 5000 / 30000 = 0.1667 BTC
        max_size = self.manager.get_max_position_size("BTC")
        self.assertAlmostEqual(max_size, 5000.0 / 30000.0, delta=0.001)
    
    def test_get_max_leverage_position(self):
        """Test getting maximum position size based on leverage constraints."""
        # Update position values
        self.manager.update_position_values({
            "BTC": 3000.0,
            "ETH": 2000.0
        })
        
        # Max leverage = 1.5, total capital = 10000
        # Max position value = 10000 * 1.5 = 15000
        # Current position value = 5000
        # Available position value = 15000 - 5000 = 10000
        # Max BTC we can buy = 10000 / 30000 = 0.333 BTC
        max_size = self.manager.get_max_leverage_position("BTC")
        self.assertAlmostEqual(max_size, 10000.0 / 30000.0, delta=0.001)


class MockMultiAgentEnv:
    """Mock environment for testing MultiAgentCapitalManager."""
    
    def __init__(self):
        self.initial_balance = 10000.0
        self.agents = ["agent1", "agent2"]
        self.positions = {"agent1": 0.1, "agent2": 1.0}
        self.balances = {"agent1": 5000.0, "agent2": 3000.0}
        self.prices = {"BTC": 30000.0, "ETH": 2000.0}
        # Map agents to assets they manage
        self.agent_assets = {"agent1": "BTC", "agent2": "ETH"}
        # Portfolio values per agent
        self.agent_portfolio_values = {
            "agent1": 8000.0,  # 5000 balance + 0.1 BTC worth 3000
            "agent2": 5000.0   # 3000 balance + 1 ETH worth 2000
        }


class TestMultiAgentCapitalManager(unittest.TestCase):
    """Test class for MultiAgentCapitalManager."""
    
    def setUp(self):
        """Set up the test environment."""
        self.mock_env = MockMultiAgentEnv()
        
        # Create capital manager
        self.manager = MultiAgentCapitalManager(
            env=self.mock_env,
            mode="isolated",
            allocation_weights={"agent1": 0.7, "agent2": 0.3},
            max_leverage=1.5
        )
    
    def test_initialization(self):
        """Test proper initialization of MultiAgentCapitalManager."""
        self.assertEqual(self.manager.env, self.mock_env)
        self.assertEqual(self.manager.initial_capital, self.mock_env.initial_balance)
        self.assertEqual(self.manager.assets, self.mock_env.agents)  # Assets = agent IDs
        self.assertEqual(self.manager.allocation_weights["agent1"], 0.7)
        self.assertEqual(self.manager.agent_assets, self.mock_env.agent_assets)
    
    def test_update_from_env_state(self):
        """Test updating capital manager state from environment state."""
        # Update state
        self.manager.update_from_env_state()
        
        # Check capital update
        self.assertEqual(self.manager.total_capital, 
                         self.mock_env.agent_portfolio_values["agent1"] + 
                         self.mock_env.agent_portfolio_values["agent2"])
        
        # In isolated mode, each agent's capital should be updated
        self.assertEqual(self.manager.allocated_capital["agent1"], 
                         self.mock_env.agent_portfolio_values["agent1"])
        self.assertEqual(self.manager.allocated_capital["agent2"], 
                         self.mock_env.agent_portfolio_values["agent2"])
    
    def test_get_agent_allocation(self):
        """Test getting allocation information for a specific agent."""
        # Update state first
        self.manager.update_from_env_state()
        
        # Get allocation info
        allocation = self.manager.get_agent_allocation("agent1")
        
        # Check info
        self.assertEqual(allocation["agent_id"], "agent1")
        self.assertEqual(allocation["mode"], "isolated")
        self.assertEqual(allocation["allocated_capital"], 
                         self.mock_env.agent_portfolio_values["agent1"])
        self.assertEqual(allocation["allocation_weight"], 0.7)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 