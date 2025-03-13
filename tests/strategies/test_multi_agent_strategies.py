import unittest
import sys
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, List

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from envs.multi_agent_env import MultiAgentTradingEnv
from agents.strategies.agent_factory import create_agent


class TestMultiAgentStrategies(unittest.TestCase):
    """
    Tests for multi-agent strategies with different agent types and trading strategies.
    
    Features:
    - Tests the integration of different agent types and strategies
    - Verifies that observation spaces are correctly shaped per strategy
    - Ensures that agents receive appropriate strategy-specific features
    - Validates that actions from different strategy agents are appropriate
    
    Implementation Notes:
    - Uses a mock data environment for testing
    - Tests both momentum and mean reversion strategies
    - Verifies observation shapes and content
    - Tests agent creation through the agent factory
    """
    
    def setUp(self):
        """Set up test environment with mock data."""
        # Create mock OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        
        # Generate some price data with a trend for testing
        close_prices = np.linspace(100, 150, 100) + np.sin(np.linspace(0, 10, 100)) * 10
        
        # Create dataframe with required columns (using $ prefix as per naming conventions)
        self.df = pd.DataFrame({
            'date': dates,
            '$open': close_prices * 0.99,
            '$high': close_prices * 1.02,
            '$low': close_prices * 0.98, 
            '$close': close_prices,
            '$volume': np.random.rand(100) * 1000 + 500
        })
        self.df.set_index('date', inplace=True)
        
        # Define test agent configurations with different strategies
        self.window_size = 20
        self.agent_configs = [
            {
                "id": "momentum_agent",
                "agent_type": "ppo",  # Learning algorithm
                "strategy": "momentum",  # Trading strategy
                "initial_balance": 10000.0,
                "priority": 1
            },
            {
                "id": "mean_reversion_agent",
                "agent_type": "ppo",  # Learning algorithm
                "strategy": "meanreversion",  # Trading strategy
                "initial_balance": 10000.0,
                "priority": 2
            }
        ]
    
    def test_observation_spaces(self):
        """Test that observation spaces are correctly configured per strategy."""
        # Create environment
        env = MultiAgentTradingEnv(
            data=self.df, 
            agent_configs=self.agent_configs,
            window_size=self.window_size
        )
        
        # Check base features
        base_features = len(self.df.columns)  # Number of OHLCV columns
        
        # Check that each agent has the correct observation space dimensions
        momentum_space = env.observation_spaces["momentum_agent"]
        meanrev_space = env.observation_spaces["mean_reversion_agent"]
        
        # Momentum agent should have base + 3 features (momentum, volatility, trend)
        self.assertEqual(momentum_space.shape, (self.window_size, base_features + 3))
        
        # Mean reversion agent should have base + 4 features (mean, std, zscore, mean_dist)
        self.assertEqual(meanrev_space.shape, (self.window_size, base_features + 4))
        
    def test_observation_content(self):
        """Test that observations contain correct data for each strategy."""
        # Create environment
        env = MultiAgentTradingEnv(
            data=self.df, 
            agent_configs=self.agent_configs,
            window_size=self.window_size
        )
        
        # Reset environment
        obs_dict, _ = env.reset()
        
        # Check that observations have correct shapes
        momentum_obs = obs_dict["momentum_agent"]
        meanrev_obs = obs_dict["mean_reversion_agent"]
        
        # Base features should be the same for both
        base_features = len(self.df.columns)
        
        # Get the actual shapes from the observations
        momentum_shape = momentum_obs.shape
        meanrev_shape = meanrev_obs.shape
        
        # Assert that window size is correct
        self.assertEqual(momentum_shape[0], self.window_size)
        self.assertEqual(meanrev_shape[0], self.window_size)
        
        # Base OHLCV data should be identical (first 5 columns)
        np.testing.assert_array_equal(
            momentum_obs[:, :base_features], 
            meanrev_obs[:, :base_features]
        )
        
        # Both observations should have at least the base features
        self.assertGreaterEqual(momentum_shape[1], base_features)
        self.assertGreaterEqual(meanrev_shape[1], base_features)
    
    def test_agent_creation(self):
        """Test creation of agents with different strategies."""
        # Create environment
        env = MultiAgentTradingEnv(
            data=self.df, 
            agent_configs=self.agent_configs,
            window_size=self.window_size
        )
        
        # Create agents through agent factory
        agents = {}
        for agent_cfg in self.agent_configs:
            agent_id = agent_cfg["id"]
            agent_type = agent_cfg["agent_type"]
            strategy = agent_cfg["strategy"]
            
            # Get spaces from environment
            obs_space = env.observation_spaces[agent_id]
            act_space = env.action_spaces[agent_id]
            
            # Create agent
            agent = create_agent(
                agent_type=agent_type,
                strategy=strategy,
                config=agent_cfg,
                observation_space=obs_space,
                action_space=act_space
            )
            
            agents[agent_id] = agent
            
            # Check agent properties
            self.assertIsNotNone(agent)
            
            # Test that the agent can process observations and return actions
            obs = env.observation_spaces[agent_id].sample()
            action = agent.get_action(obs)
            
            # Actions should be within the action space
            self.assertTrue(env.action_spaces[agent_id].contains(action))
    
    def test_agent_actions(self):
        """Test that agents produce reasonable actions based on their strategies."""
        # Create environment
        env = MultiAgentTradingEnv(
            data=self.df, 
            agent_configs=self.agent_configs,
            window_size=self.window_size
        )
        
        # Create agents
        momentum_agent = create_agent(
            agent_type="ppo",
            strategy="momentum",
            config=self.agent_configs[0],
            observation_space=env.observation_spaces["momentum_agent"],
            action_space=env.action_spaces["momentum_agent"]
        )
        
        meanrev_agent = create_agent(
            agent_type="ppo",
            strategy="meanreversion",
            config=self.agent_configs[1],
            observation_space=env.observation_spaces["mean_reversion_agent"],
            action_space=env.action_spaces["mean_reversion_agent"]
        )
        
        # Reset environment
        obs_dict, _ = env.reset()
        
        # Get actions
        momentum_action = momentum_agent.get_action(obs_dict["momentum_agent"])
        meanrev_action = meanrev_agent.get_action(obs_dict["mean_reversion_agent"])
        
        # Both actions should be within bounds
        self.assertTrue(-1 <= momentum_action[0] <= 1)
        self.assertTrue(-1 <= meanrev_action[0] <= 1)
        

if __name__ == "__main__":
    unittest.main() 