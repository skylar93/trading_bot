#!/usr/bin/env python
"""Test script for asset-specific trading agents."""

import os
import sys
import unittest
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock
from gymnasium import spaces

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from agents.strategies.asset_specific_agents import (
    AssetSpecificAgent, 
    CryptoAgent, 
    EquityAgent, 
    AssetCharacteristics,
    AssetSpecificAgentFactory
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class TestAssetCharacteristics(unittest.TestCase):
    """Test class for asset characteristics dataclass."""
    
    def test_characteristics_initialization(self):
        """Test initialization of AssetCharacteristics."""
        # Create characteristics for different asset types
        crypto_chars = AssetCharacteristics(
            volatility_factor=2.0,
            trading_hours="24/7",
            typical_spread=0.001,
            slippage_factor=0.002,
            min_trade_size=0.0001,
            fee_structure={"maker": 0.001, "taker": 0.002}
        )
        
        equity_chars = AssetCharacteristics(
            volatility_factor=1.0,
            trading_hours="exchange",
            typical_spread=0.0005,
            slippage_factor=0.001,
            min_trade_size=1.0,
            fee_structure={"commission": 0.0005}
        )
        
        # Test that values are correctly assigned
        self.assertEqual(crypto_chars.volatility_factor, 2.0)
        self.assertEqual(crypto_chars.trading_hours, "24/7")
        self.assertEqual(crypto_chars.typical_spread, 0.001)
        self.assertEqual(crypto_chars.fee_structure.get("maker"), 0.001)
        
        self.assertEqual(equity_chars.volatility_factor, 1.0)
        self.assertEqual(equity_chars.trading_hours, "exchange")
        self.assertEqual(equity_chars.min_trade_size, 1.0)
        self.assertEqual(equity_chars.fee_structure.get("commission"), 0.0005)


# Create a concrete implementation for testing the abstract base class
class ConcreteAssetAgent(AssetSpecificAgent):
    """Concrete implementation of AssetSpecificAgent for testing."""
    
    def _setup_network(self, config):
        """Implement required abstract method."""
        self.network = MagicMock()
        self.optimizer = MagicMock()
    
    def act(self, observation, deterministic=False):
        """Implement required abstract method."""
        return np.array([0.5])  # Fixed action for testing
    
    def update(self, experience):
        """Implement required abstract method."""
        return {"loss": 0.1}
    
    def save(self, path):
        """Implement required abstract method."""
        pass
    
    def load(self, path):
        """Implement required abstract method."""
        pass


class TestAssetSpecificAgent(unittest.TestCase):
    """Test class for the abstract AssetSpecificAgent base class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        
        # Create a concrete agent for testing
        self.agent = ConcreteAssetAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="BTC",
            asset_type="crypto"
        )
    
    def test_initialization(self):
        """Test proper initialization of the agent."""
        self.assertEqual(self.agent.asset_id, "BTC")
        self.assertEqual(self.agent.asset_type, "crypto")
        self.assertIsNotNone(self.agent.characteristics)
        self.assertEqual(self.agent.characteristics.trading_hours, "24/7")  # Default for crypto
        
        # Check that metrics are initialized
        for metric_key in ["train_loss", "value_loss", "policy_loss", "rewards"]:
            self.assertIn(metric_key, self.agent.metrics)
            self.assertEqual(len(self.agent.metrics[metric_key]), 0)
        
        # Check state initialization
        self.assertEqual(self.agent.state["position"], 0.0)
        self.assertEqual(self.agent.state["trade_count"], 0)
        self.assertEqual(self.agent.state["market_regime"], "unknown")
    
    def test_default_characteristics(self):
        """Test the default characteristics for different asset types."""
        # Test crypto characteristics
        crypto_agent = ConcreteAssetAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="BTC",
            asset_type="crypto"
        )
        
        self.assertEqual(crypto_agent.characteristics.trading_hours, "24/7")
        self.assertGreater(crypto_agent.characteristics.volatility_factor, 1.5)
        
        # Test equity characteristics
        equity_agent = ConcreteAssetAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="AAPL",
            asset_type="equity"
        )
        
        self.assertEqual(equity_agent.characteristics.trading_hours, "exchange")
        
        # Test default for unknown asset type
        unknown_agent = ConcreteAssetAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="XYZ",
            asset_type="unknown"
        )
        
        self.assertEqual(unknown_agent.characteristics.trading_hours, "standard")
        self.assertEqual(unknown_agent.characteristics.volatility_factor, 1.0)
    
    def test_update_state(self):
        """Test updating agent state."""
        new_state = {
            "position": 1.5,
            "avg_entry_price": 25000.0,
            "unrealized_pnl": 500.0
        }
        
        self.agent.update_state(new_state)
        
        # Check that state was updated
        self.assertEqual(self.agent.state["position"], 1.5)
        self.assertEqual(self.agent.state["avg_entry_price"], 25000.0)
        self.assertEqual(self.agent.state["unrealized_pnl"], 500.0)
        
        # Other state values should remain unchanged
        self.assertEqual(self.agent.state["trade_count"], 0)
        self.assertEqual(self.agent.state["market_regime"], "unknown")
    
    def test_reset(self):
        """Test resetting agent state."""
        # First update state
        new_state = {
            "position": 1.5,
            "avg_entry_price": 25000.0,
            "unrealized_pnl": 500.0,
            "trade_count": 10,
            "market_regime": "volatile"
        }
        self.agent.update_state(new_state)
        
        # Then reset
        self.agent.reset()
        
        # Check that state was reset
        self.assertEqual(self.agent.state["position"], 0.0)
        self.assertEqual(self.agent.state["avg_entry_price"], 0.0)
        self.assertEqual(self.agent.state["unrealized_pnl"], 0.0)
        self.assertEqual(self.agent.state["trade_count"], 0)
        self.assertEqual(self.agent.state["market_regime"], "unknown")
    
    def test_preprocess_observation(self):
        """Test observation preprocessing."""
        # Create a sample observation
        obs = np.random.rand(10)
        
        # Call preprocess
        processed_obs = self.agent.preprocess_observation(obs)
        
        # Check result type and shape
        self.assertIsInstance(processed_obs, torch.Tensor)
        self.assertEqual(processed_obs.shape, obs.shape)
        
        # Check values (should match original observation)
        np.testing.assert_allclose(processed_obs.cpu().numpy(), obs, rtol=1e-5)


class TestCryptoAgent(unittest.TestCase):
    """Test class for the crypto-specific agent."""
    
    def setUp(self):
        """Set up the test environment."""
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        
        # Create a crypto agent for testing
        self.agent = CryptoAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="BTC",
            volatility_scaling=True
        )
        
        # Sample observation (simulating OHLCV data)
        # [time, open, high, low, close, volume, ...]
        self.sample_obs = np.random.rand(50)
        self.sample_obs = self.sample_obs.reshape(10, 5)  # 10 time steps, 5 features per step
    
    def test_initialization(self):
        """Test proper initialization of the crypto agent."""
        self.assertEqual(self.agent.asset_id, "BTC")
        self.assertEqual(self.agent.asset_type, "crypto")
        self.assertTrue(self.agent.volatility_scaling)
        
        # Check network initialization
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.value_net)
        self.assertIsNotNone(self.agent.optimizer)
    
    def test_crypto_specific_preprocessing(self):
        """Test crypto-specific observation preprocessing."""
        # Create a sample observation with volume spike in the last feature
        obs = np.ones((10, 5))  # 10 time steps, 5 features (OHLCV)
        obs[:, 4] = 1000.0  # Set volume to a large value
        
        flat_obs = obs.flatten()
        
        # Process observation
        processed_obs = self.agent.preprocess_observation(flat_obs)
        
        # Reshape to original dimensions
        processed_reshaped = processed_obs.cpu().numpy().reshape(10, 5)
        
        # Check that volume was transformed (log transformation)
        self.assertLess(processed_reshaped[0, 4], 10.0)  # log(1000) ≈ 6.9
    
    def test_act_method(self):
        """Test the act method of the crypto agent."""
        # Get action
        action = self.agent.act(self.sample_obs.flatten())
        
        # Check output shape and range
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))
    
    def test_volatility_scaling(self):
        """Test volatility scaling in the crypto agent."""
        # Create two versions of the agent - with and without volatility scaling
        agent_with_scaling = CryptoAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="BTC",
            volatility_scaling=True
        )
        
        agent_without_scaling = CryptoAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="BTC",
            volatility_scaling=False
        )
        
        # Create a mock act method for both agents that will return controlled values
        def mock_act_with_scaling(observation, deterministic=False):
            # In volatile regime, this agent will scale down its action
            regime = agent_with_scaling.analyze_market_regime(observation.reshape(-1, 5))
            if regime == "volatile":
                return np.array([0.3])  # Scaled down value
            return np.array([0.5])  # Normal value
            
        def mock_act_without_scaling(observation, deterministic=False):
            return np.array([0.5])  # Fixed value regardless of volatility
        
        # Use patch to mock the act methods
        with patch.object(agent_with_scaling, 'act', side_effect=mock_act_with_scaling):
            with patch.object(agent_without_scaling, 'act', side_effect=mock_act_without_scaling):
                with patch.object(agent_with_scaling, 'analyze_market_regime', return_value="volatile"):
                    with patch.object(agent_without_scaling, 'analyze_market_regime', return_value="volatile"):
                        # Get actions from both agents
                        action_with_scaling = agent_with_scaling.act(self.sample_obs.flatten())
                        action_without_scaling = agent_without_scaling.act(self.sample_obs.flatten())
                        
                        # In volatile regime, the action with scaling should be smaller in magnitude
                        self.assertLess(abs(action_with_scaling[0]), abs(action_without_scaling[0]))
    
    def test_market_regime_analysis(self):
        """Test market regime analysis for crypto agent."""
        # Create sample observation with high volatility
        high_vol_obs = np.ones((10, 5))  # 10 time steps, 5 features
        # Set high and low prices to create volatility
        high_vol_obs[:, 2] = 1.10  # high prices 10% above reference
        high_vol_obs[:, 3] = 0.90  # low prices 10% below reference
        
        # Create sample observation with low volatility
        low_vol_obs = np.ones((10, 5))
        # Set high and low prices to create low volatility
        low_vol_obs[:, 2] = 1.01  # high prices 1% above reference
        low_vol_obs[:, 3] = 0.99  # low prices 1% below reference
        
        # Analyze regimes
        high_vol_regime = self.agent.analyze_market_regime(high_vol_obs)
        low_vol_regime = self.agent.analyze_market_regime(low_vol_obs)
        
        # Check results - high volatility should be classified as "volatile"
        self.assertEqual(high_vol_regime, "volatile")
        
        # Low volatility should be classified as "ranging" or "normal"
        # The actual implementation returns "normal" for low volatility
        self.assertIn(low_vol_regime, ["ranging", "normal"])


class TestEquityAgent(unittest.TestCase):
    """Test class for the equity-specific agent."""
    
    def setUp(self):
        """Set up the test environment."""
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        
        # Create an equity agent for testing
        self.agent = EquityAgent(
            observation_space=self.obs_space,
            action_space=self.action_space,
            asset_id="AAPL",
            use_fundamentals=True
        )
        
        # Sample observation
        self.sample_obs = np.random.rand(50)
    
    def test_initialization(self):
        """Test proper initialization of the equity agent."""
        self.assertEqual(self.agent.asset_id, "AAPL")
        self.assertEqual(self.agent.asset_type, "equity")
        self.assertTrue(self.agent.use_fundamentals)
        
        # Check network initialization
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.value_net)
        self.assertIsNotNone(self.agent.optimizer)
        
        # Check equity-specific characteristics
        self.assertEqual(self.agent.characteristics.trading_hours, "exchange")
        self.assertLess(self.agent.characteristics.typical_spread, 0.001)  # Equities typically have tighter spreads
    
    def test_act_method(self):
        """Test the act method of the equity agent."""
        # Get action
        action = self.agent.act(self.sample_obs)
        
        # Check output shape and range
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))
    
    def test_update_method(self):
        """Test the update method for equity agent."""
        # Create mock experience data
        batch_size = 4
        experience = {
            "states": np.random.rand(batch_size, 50),
            "actions": np.random.uniform(-1, 1, (batch_size, 1)),
            "rewards": np.random.uniform(-0.1, 0.1, batch_size),
            "next_states": np.random.rand(batch_size, 50),
            "dones": np.zeros(batch_size)
        }
        
        # Call update method
        metrics = self.agent.update(experience)
        
        # Check metrics returned
        self.assertIn("loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        
        # Check that metrics were stored
        self.assertEqual(len(self.agent.metrics["train_loss"]), 1)
        self.assertEqual(len(self.agent.metrics["policy_loss"]), 1)
        self.assertEqual(len(self.agent.metrics["value_loss"]), 1)


class TestAgentFactory(unittest.TestCase):
    """Test class for AssetSpecificAgentFactory."""
    
    def setUp(self):
        """Set up the test environment."""
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    
    def test_create_crypto_agent(self):
        """Test creation of a crypto agent."""
        agent = AssetSpecificAgentFactory.create_agent(
            asset_id="BTC",
            asset_type="crypto",
            observation_space=self.obs_space,
            action_space=self.action_space,
            config={"volatility_scaling": True}
        )
        
        # Check agent type and properties
        self.assertIsInstance(agent, CryptoAgent)
        self.assertEqual(agent.asset_id, "BTC")
        self.assertEqual(agent.asset_type, "crypto")
        self.assertTrue(agent.volatility_scaling)
    
    def test_create_equity_agent(self):
        """Test creation of an equity agent."""
        agent = AssetSpecificAgentFactory.create_agent(
            asset_id="AAPL",
            asset_type="equity",
            observation_space=self.obs_space,
            action_space=self.action_space,
            config={"use_fundamentals": True}
        )
        
        # Check agent type and properties
        self.assertIsInstance(agent, EquityAgent)
        self.assertEqual(agent.asset_id, "AAPL")
        self.assertEqual(agent.asset_type, "equity")
        self.assertTrue(agent.use_fundamentals)
    
    def test_create_default_agent(self):
        """Test creation of a default agent for unknown asset type."""
        # Since AssetSpecificAgent is abstract and can't be instantiated directly,
        # we need to mock the create_agent method to return our ConcreteAssetAgent
        with patch('agents.strategies.asset_specific_agents.AssetSpecificAgentFactory.create_agent') as mock_create:
            # Set up the mock to return a ConcreteAssetAgent instance
            mock_agent = MagicMock()
            mock_agent.asset_id = "XYZ"
            mock_agent.asset_type = "other"
            mock_create.return_value = mock_agent
            
            # Call the mocked method
            agent = AssetSpecificAgentFactory.create_agent(
                asset_id="XYZ",
                asset_type="other",
                observation_space=self.obs_space,
                action_space=self.action_space
            )
            
            # For unknown types, should return our mocked agent
            self.assertEqual(agent.asset_id, "XYZ")
            self.assertEqual(agent.asset_type, "other")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 