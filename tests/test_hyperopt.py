"""Unit tests for hyperparameter optimization components"""
import unittest
import numpy as np
import pandas as pd
import torch

from training.hyperopt.hyperopt_env import SimplifiedTradingEnv
from training.hyperopt.hyperopt_agent import MinimalPPOAgent

class TestComponents(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'open': [100, 101, 102, 101, 100] * 2,
            'high': [102, 102, 103, 102, 101] * 2,
            'low':  [99, 100, 101, 100, 99] * 2,
            'close':[101, 102, 101, 100, 100] * 2,
            'volume':[1000, 1100, 900, 1000, 1200] * 2
        })
    
    def test_env_init(self):
        """Test environment initialization"""
        env = SimplifiedTradingEnv(
            df=self.test_data,
            window_size=3
        )
        self.assertEqual(env.window_size, 3)
        self.assertEqual(env.initial_balance, 10000)
    
    def test_env_reset(self):
        """Test environment reset"""
        env = SimplifiedTradingEnv(df=self.test_data, window_size=3)
        obs, info = env.reset()
        
        self.assertEqual(obs.shape, (3, 5))
        self.assertIsInstance(info, dict)
    
    def test_env_step(self):
        """Test environment step"""
        env = SimplifiedTradingEnv(df=self.test_data, window_size=3)
        env.reset()
        
        obs, reward, done, truncated, info = env.step(0.5)  # Buy action
        
        self.assertEqual(obs.shape, (3, 5))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('portfolio_value', info)
        self.assertIn('return', info)
    
    def test_agent_init(self):
        """Test agent initialization"""
        env = SimplifiedTradingEnv(df=self.test_data, window_size=3)
        agent = MinimalPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_size=32
        )
        
        self.assertIsInstance(agent.network, torch.nn.Module)
    
    def test_agent_action(self):
        """Test agent action generation"""
        env = SimplifiedTradingEnv(df=self.test_data, window_size=3)
        agent = MinimalPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_size=32
        )
        
        obs = env.reset()[0]
        action = agent.get_action(obs)
        
        self.assertGreaterEqual(action, -1)
        self.assertLessEqual(action, 1)
    
    def test_agent_train_step(self):
        """Test single training step"""
        env = SimplifiedTradingEnv(df=self.test_data, window_size=3)
        agent = MinimalPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            hidden_size=32
        )
        
        # Get initial state
        state = env.reset()[0]
        action = agent.get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Train step
        metrics = agent.train(state, action, reward, next_state, done)
        
        self.assertIn('loss', metrics)
        self.assertIn('policy_loss', metrics)
        self.assertIn('value_loss', metrics)

if __name__ == '__main__':
    unittest.main()