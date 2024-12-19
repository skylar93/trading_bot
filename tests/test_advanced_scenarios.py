"""Test advanced backtesting scenarios"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.backtesting.advanced_scenarios import ScenarioGenerator, ScenarioTester
from envs.trading_env import TradingEnvironment
from agents.strategies.ppo_agent import PPOAgent

class TestAdvancedScenarios(unittest.TestCase):
    def setUp(self):
        """Set up test environment and agent"""
        self.scenario_generator = ScenarioGenerator()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        self.sample_data = pd.DataFrame({
            '$open': np.random.randn(100) * 10 + 100,
            '$high': np.random.randn(100) * 10 + 105,
            '$low': np.random.randn(100) * 10 + 95,
            '$close': np.random.randn(100) * 10 + 100,
            '$volume': np.abs(np.random.randn(100) * 1000),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.normal(0, 1, 100),
            'Signal': np.random.normal(0, 1, 100)
        }, index=dates)
        
        # Create basic environment and agent
        self.env = TradingEnvironment(
            df=self.sample_data,
            initial_balance=10000.0,
            trading_fee=0.001,
            window_size=20
        )
        
        # Initialize agent with proper spaces
        self.env.reset()  # Need to reset to get proper spaces
        self.agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        self.scenario_tester = ScenarioTester(self.env, self.agent)
    
    def test_flash_crash(self):
        """Test flash crash scenario generation and properties"""
        flash_crash_data = self.scenario_generator.generate_flash_crash(
            length=100,
            crash_idx=50,
            crash_size=0.15
        )
        
        # Check data structure
        self.assertTrue(isinstance(flash_crash_data, pd.DataFrame))
        self.assertEqual(len(flash_crash_data), 100)
        
        # Verify crash occurred
        pre_crash = flash_crash_data['$close'][49]
        crash_price = flash_crash_data['$close'][50]
        self.assertTrue(crash_price < pre_crash * 0.9)
    
    def test_low_liquidity(self):
        """Test low liquidity scenario generation and properties"""
        low_liq_data = self.scenario_generator.generate_low_liquidity(
            length=100,
            low_liq_start=30,
            low_liq_length=20
        )
        
        # Check volume reduction during low liquidity period
        normal_volume = low_liq_data['$volume'][:30].mean()
        low_liq_volume = low_liq_data['$volume'][30:50].mean()
        self.assertTrue(low_liq_volume < normal_volume * 0.2)
    
    def test_choppy_market(self):
        """Test choppy market scenario generation and properties"""
        choppy_data = self.scenario_generator.generate_choppy_market(
            length=100,
            chop_intensity=2.0
        )
        
        # Calculate price changes
        price_changes = choppy_data['$close'].pct_change().dropna()
        
        # Verify increased volatility
        self.assertTrue(price_changes.std() > 0.01)
    
    def test_scenario_combination(self):
        """Test combining multiple scenarios"""
        scenarios = [
            self.scenario_generator.generate_flash_crash(length=50),
            self.scenario_generator.generate_low_liquidity(length=50)
        ]
        
        combined = self.scenario_generator.combine_scenarios(scenarios)
        self.assertEqual(len(combined), 100)
    
    def test_scenario_testing(self):
        """Test the scenario testing functionality"""
        # Generate test scenario
        test_data = self.scenario_generator.generate_flash_crash()
        
        # Run test
        results = self.scenario_tester.test_scenario(test_data)
        
        # Check results structure
        self.assertIn('metrics', results)
        self.assertIn('trades', results)
        self.assertIn('portfolio_values', results)
        self.assertIn('timestamps', results)
        
        # Check metrics
        metrics = results['metrics']
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # Check portfolio values
        self.assertTrue(len(results['portfolio_values']) > 0)
        self.assertTrue(len(results['timestamps']) > 0)
        self.assertEqual(len(results['portfolio_values']), len(results['timestamps']))

if __name__ == '__main__':
    unittest.main()