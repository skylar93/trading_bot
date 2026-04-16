"""
Test the risk manager implementation for RL environments.

Tests cover:
- Stop loss functionality
- Trailing stop functionality
- VaR calculation and threshold checking
- Max drawdown detection
- Integration with environment step
"""

import unittest
import numpy as np
import pandas as pd
from risk_management import create_risk_manager


class TestRLRiskManager(unittest.TestCase):
    """Test suite for RL environment risk manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "use_stop_loss": True,
            "stop_loss_threshold": 0.1,  # 10% loss
            "use_trailing_stop": True,
            "trailing_stop_buffer": 0.05,  # 5% drop from highest
            "use_var": True,
            "var_confidence_level": 0.95,
            "rolling_var_window": 10,
            "action_on_var_exceed": "reduce_position",
            "max_drawdown_pct": 0.15,
            "use_forced_liquidation": True,
            "check_frequency": 1
        }
        self.risk_manager = create_risk_manager("rl", self.config)
        
    def test_stop_loss_long_position(self):
        """Test stop loss for long positions."""
        agent_id = "agent1"
        position_size = 1.0  # Long position
        entry_price = 100.0
        
        # Price above entry - no stop loss
        current_price = 105.0
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertFalse(stop_loss_triggered)
        
        # Small loss - no stop loss
        current_price = 95.0  # 5% loss
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertFalse(stop_loss_triggered)
        
        # Loss exceeding threshold - stop loss triggered
        current_price = 85.0  # 15% loss
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertTrue(stop_loss_triggered)
        
    def test_stop_loss_short_position(self):
        """Test stop loss for short positions."""
        agent_id = "agent1"
        position_size = -1.0  # Short position
        entry_price = 100.0
        
        # Price below entry - no stop loss
        current_price = 95.0
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertFalse(stop_loss_triggered)
        
        # Small loss - no stop loss
        current_price = 105.0  # 5% loss
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertFalse(stop_loss_triggered)
        
        # Loss exceeding threshold - stop loss triggered
        current_price = 115.0  # 15% loss
        stop_loss_triggered = self.risk_manager.check_stop_loss(
            agent_id, position_size, entry_price, current_price
        )
        self.assertTrue(stop_loss_triggered)
        
    def test_trailing_stop_long_position(self):
        """Test trailing stop for long positions."""
        agent_id = "agent1"
        asset = "BTC"
        position_size = 1.0  # Long position
        
        # Initial position
        current_price = 100.0
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price rises - no trailing stop
        current_price = 110.0
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price drops slightly - no trailing stop
        current_price = 106.0  # 3.6% drop from high
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price drops more than buffer - trailing stop triggered
        current_price = 103.0  # 6.4% drop from high
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertTrue(triggered)
        
    def test_trailing_stop_short_position(self):
        """Test trailing stop for short positions."""
        agent_id = "agent1"
        asset = "BTC"
        position_size = -1.0  # Short position
        
        # Initial position
        current_price = 100.0
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price drops - no trailing stop
        current_price = 90.0
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price rises slightly - no trailing stop
        current_price = 93.0  # 3.3% rise from low
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertFalse(triggered)
        
        # Price rises more than buffer - trailing stop triggered
        current_price = 96.0  # 6.7% rise from low
        triggered = self.risk_manager.check_trailing_stop(
            agent_id, asset, position_size, current_price
        )
        self.assertTrue(triggered)
        
    def test_var_calculation(self):
        """Test VaR calculation and threshold checking."""
        agent_id = "agent1"
        
        # Not enough data
        var = self.risk_manager.calculate_var(agent_id)
        self.assertIsNone(var)
        
        # Add returns data
        returns = [0.01, -0.02, 0.015, -0.01, -0.03, 0.02, -0.015, -0.025, 0.01, -0.01]
        self.risk_manager.returns_history[agent_id] = returns
        
        # Calculate VaR
        var = self.risk_manager.calculate_var(agent_id)
        self.assertIsNotNone(var)
        self.assertGreater(var, 0)
        
        # Test VaR exceedance - no exceedance
        current_return = -0.01  # Small loss
        action = self.risk_manager.check_var_exceed(agent_id, current_return)
        self.assertIsNone(action)
        
        # Test VaR exceedance - exceeds VaR
        current_return = -0.04  # Large loss
        action = self.risk_manager.check_var_exceed(agent_id, current_return)
        self.assertEqual(action, "reduce_position")
        
    def test_max_drawdown(self):
        """Test maximum drawdown detection."""
        agent_id = "agent1"
        
        # Initial values
        self.risk_manager.peak_values[agent_id] = 10000.0
        self.risk_manager.current_values[agent_id] = 10000.0
        
        # No drawdown
        exceeded = self.risk_manager.check_drawdown(agent_id)
        self.assertFalse(exceeded)
        
        # Small drawdown
        self.risk_manager.current_values[agent_id] = 9000.0  # 10% drawdown
        exceeded = self.risk_manager.check_drawdown(agent_id)
        self.assertFalse(exceeded)
        
        # Drawdown exceeding threshold
        self.risk_manager.current_values[agent_id] = 8000.0  # 20% drawdown
        exceeded = self.risk_manager.check_drawdown(agent_id)
        self.assertTrue(exceeded)
        
    def test_update_portfolio_values(self):
        """Test updating portfolio values and tracking peaks."""
        agent_ids = ["agent1", "agent2"]
        
        # Initial update
        portfolio_values = {
            "agent1": 10000.0,
            "agent2": 15000.0
        }
        self.risk_manager.update_portfolio_values(portfolio_values)
        
        for agent_id in agent_ids:
            self.assertEqual(
                self.risk_manager.peak_values[agent_id], 
                portfolio_values[agent_id]
            )
            self.assertEqual(
                self.risk_manager.current_values[agent_id], 
                portfolio_values[agent_id]
            )
        
        # Update with higher values
        portfolio_values = {
            "agent1": 12000.0,
            "agent2": 14000.0
        }
        self.risk_manager.update_portfolio_values(portfolio_values)
        
        self.assertEqual(self.risk_manager.peak_values["agent1"], 12000.0)  # Updated
        self.assertEqual(self.risk_manager.peak_values["agent2"], 15000.0)  # Retained peak
        
        # Update with lower values
        portfolio_values = {
            "agent1": 11000.0,
            "agent2": 13000.0
        }
        self.risk_manager.update_portfolio_values(portfolio_values)
        
        self.assertEqual(self.risk_manager.peak_values["agent1"], 12000.0)  # Retained peak
        self.assertEqual(self.risk_manager.peak_values["agent2"], 15000.0)  # Retained peak
        
    def test_risk_events_info(self):
        """Test tracking of risk events."""
        # Initial state
        info = self.risk_manager._get_risk_events_info()
        self.assertEqual(info["stop_loss_events"], 0)
        self.assertEqual(info["trailing_stop_events"], 0)
        self.assertEqual(info["var_exceed_events"], 0)
        self.assertEqual(info["forced_liquidation_events"], 0)
        
        # Trigger events
        agent_id = "agent1"
        
        # Stop loss
        self.risk_manager.check_stop_loss(
            agent_id, 1.0, 100.0, 85.0  # 15% loss
        )
        
        # Trailing stop
        self.risk_manager.position_highest_values[f"{agent_id}_default"] = 110.0
        self.risk_manager.check_trailing_stop(
            agent_id, "default", 1.0, 103.0  # 6.4% drop from high
        )
        
        # VaR exceed
        self.risk_manager.returns_history[agent_id] = [0.01, -0.02, 0.015, -0.01, -0.03, 0.02, -0.015, -0.025, 0.01, -0.01]
        self.risk_manager.check_var_exceed(agent_id, -0.04)
        
        # Check updated counts
        info = self.risk_manager._get_risk_events_info()
        self.assertEqual(info["stop_loss_events"], 1)
        self.assertEqual(info["trailing_stop_events"], 1)
        self.assertEqual(info["var_exceed_events"], 1)
        
        # Reset risk manager
        self.risk_manager.reset()
        info = self.risk_manager._get_risk_events_info()
        self.assertEqual(info["stop_loss_events"], 0)


if __name__ == "__main__":
    unittest.main() 