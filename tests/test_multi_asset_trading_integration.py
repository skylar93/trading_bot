#!/usr/bin/env python
"""Integration test for multi-asset trading components."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from envs.multi_asset_env import MultiAssetTradingEnv
from envs.capital_manager import MultiAssetCapitalManager, CapitalManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger('test_multi_asset_trading_integration')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_test_data() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    
    # Create price data for BTC
    btc_prices = 20000 + np.cumsum(np.random.normal(0, 500, len(dates)))
    btc_prices = np.maximum(btc_prices, 15000)  # Ensure no negative prices
    
    # Create price data for ETH (correlated with BTC)
    eth_prices = 1500 + 0.8 * np.cumsum(np.random.normal(0, 30, len(dates))) + 0.2 * (btc_prices - 20000) / 10
    eth_prices = np.maximum(eth_prices, 1000)
    
    # Create volumes
    btc_volumes = np.random.uniform(500, 2000, len(dates))
    eth_volumes = np.random.uniform(5000, 20000, len(dates))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'BTC_$open': btc_prices * 0.99,
        'BTC_$high': btc_prices * 1.02,
        'BTC_$low': btc_prices * 0.98,
        'BTC_$close': btc_prices,
        'BTC_$volume': btc_volumes,
        'ETH_$open': eth_prices * 0.99,
        'ETH_$high': eth_prices * 1.02,
        'ETH_$low': eth_prices * 0.98,
        'ETH_$close': eth_prices,
        'ETH_$volume': eth_volumes
    })
    
    return data

class TestEnvironmentBasics(unittest.TestCase):
    """Basic tests for the multi-asset trading environment."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create test data
        self.data = create_test_data()
        
        # Create trading environment
        self.env = MultiAssetTradingEnv(
            df=self.data,
            assets=['BTC', 'ETH'],
            initial_balance=10000.0,
            window_size=7,
            action_type='portfolio_weights',
            add_position_info=True
        )
        
        # Create capital manager
        self.capital_manager = MultiAssetCapitalManager(
            env=self.env,
            mode='shared',
            allocation_weights={'BTC': 0.6, 'ETH': 0.4},
            max_leverage=1.0
        )
    
    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        # Check basic properties
        self.assertEqual(self.env.assets, ['BTC', 'ETH'])
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.window_size, 7)
        self.assertEqual(self.env.action_type, 'portfolio_weights')
        
        # Check observation space
        obs, info = self.env.reset()
        self.assertEqual(obs.shape[0], self.env.window_size)
        
        # Check action space
        self.assertEqual(self.env.action_space.shape[0], len(self.env.assets))
    
    def test_step_with_fixed_action(self):
        """Test taking a step with a fixed action."""
        # Reset environment
        obs, info = self.env.reset()
        
        # Take action (40% BTC, 40% ETH, 20% cash)
        action = np.array([0.4, 0.4])
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check that we get a valid observation and reward
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, (int, float))
        
        # Check portfolio weights
        self.assertAlmostEqual(self.env.current_weights.get('cash', 0.0) + 
                              self.env.current_weights.get('BTC', 0.0) + 
                              self.env.current_weights.get('ETH', 0.0), 
                              1.0, delta=0.01)
    
    def test_capital_manager_integration(self):
        """Test that the capital manager tracks the environment state."""
        # Reset environment
        obs, info = self.env.reset()
        
        # Take a step
        action = np.array([0.3, 0.3])  # 30% BTC, 30% ETH, 40% cash
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update capital manager
        self.capital_manager.update_from_env_state()
        
        # Check that capital manager's state matches environment
        self.assertAlmostEqual(
            self.capital_manager.total_capital,
            self.env.portfolio_value,
            delta=0.01
        )
        
        # Check position values
        for asset in self.env.assets:
            position_value = self.env.positions[asset] * self.env.prices[asset]
            self.assertAlmostEqual(
                self.capital_manager.position_values[asset],
                position_value,
                delta=0.01
            )

class TestCapitalManagementModes(unittest.TestCase):
    """Tests comparing shared vs isolated capital management."""
    
    def setUp(self):
        """Set up the test environments."""
        # Create test data
        self.data = create_test_data()
        
        # Fixed action for both environments
        self.fixed_action = np.array([0.4, 0.3])  # 40% BTC, 30% ETH, 30% cash
    
    def test_shared_vs_isolated_capital(self):
        """Test basic differences between shared and isolated capital modes."""
        # Create managers directly without environments
        shared_manager = CapitalManager(
            initial_capital=10000.0,
            mode='shared',
            assets=['BTC', 'ETH'],
            allocation_weights={'BTC': 0.6, 'ETH': 0.4}
        )
        
        isolated_manager = CapitalManager(
            initial_capital=10000.0,
            mode='isolated',
            assets=['BTC', 'ETH'],
            allocation_weights={'BTC': 0.6, 'ETH': 0.4}
        )
        
        # Check initial allocations
        self.assertEqual(shared_manager.allocated_capital['shared'], 10000.0)
        self.assertEqual(isolated_manager.allocated_capital['BTC'], 6000.0)
        self.assertEqual(isolated_manager.allocated_capital['ETH'], 4000.0)
        
        # Allocate capital for BTC
        shared_allocated = shared_manager.allocate_capital('BTC', 5000.0)
        isolated_allocated = isolated_manager.allocate_capital('BTC', 5000.0)
        
        # In shared mode, full amount should be allocated
        self.assertEqual(shared_allocated, 5000.0)
        
        # In isolated mode, only up to the asset's allocation can be used
        self.assertEqual(isolated_allocated, 5000.0)  # BTC has 6000 allocated
        
        # Check available capital for ETH
        shared_eth_available = shared_manager.get_available_capital('ETH')
        isolated_eth_available = isolated_manager.get_available_capital('ETH')
        
        # In shared mode, ETH's available capital is reduced by BTC's allocation
        self.assertEqual(shared_eth_available, 5000.0)  # 10000 - 5000
        
        # In isolated mode, ETH's allocation is separate from BTC
        self.assertEqual(isolated_eth_available, 4000.0)  # ETH's full allocation
        
        # Simulate some profit/loss
        # BTC gains 1000, ETH loses 500
        shared_manager.update_capital({'BTC': 1000.0, 'ETH': -500.0})
        isolated_manager.update_capital({'BTC': 1000.0, 'ETH': -500.0})
        
        # In shared mode, the total capital is updated
        self.assertEqual(shared_manager.total_capital, 10500.0)  # 10000 + 1000 - 500
        
        # In isolated mode, each asset's capital is updated separately
        self.assertEqual(isolated_manager.allocated_capital['BTC'], 7000.0)  # 6000 + 1000
        self.assertEqual(isolated_manager.allocated_capital['ETH'], 3500.0)  # 4000 - 500

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 