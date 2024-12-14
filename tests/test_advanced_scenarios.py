"""Test advanced backtesting scenarios"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.backtesting.advanced_scenarios import ScenarioGenerator, ScenarioTester
from envs.trading_env import TradingEnvironment
from agents.ppo_agent import PPOAgent

class TestAdvancedScenarios(unittest.TestCase):
    def setUp(self):
        """Set up test environment and agent"""
        self.scenario_generator = ScenarioGenerator()
        
        # Create basic environment and agent
        self.env = TradingEnvironment()
        self.agent = PPOAgent()
        
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
        pre_crash = flash_crash_data['close'][49]
        crash_price = flash_crash_data['close'][50]
        self.assertTrue(crash_price < pre_crash * 0.9)
    
    def test_low_liquidity(self):
        """Test low liquidity scenario generation and properties"""
        low_liq_data = self.scenario_generator.generate_low_liquidity(
            length=100,
            low_liq_start=30,
            low_liq_length=20
        )
        
        # Check volume reduction during low liquidity period
        normal_volume = low_liq_data['volume'][:30].mean()
        low_liq_volume = low_liq_data['volume'][30:50].mean()
        self.assertTrue(low_liq_volume < normal_volume * 0.2)
    
    def test_choppy_market(self):
        """Test choppy market scenario generation and properties"""
        choppy_data = self.scenario_generator.generate_choppy_market(
            length=100,
            chop_intensity=2.0
        )
        
        # Calculate price changes
        price_changes = choppy_data['close'].pct_change().dropna()
        
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
        self.assertIn('total_reward', results)
        self.assertIn('final_balance', results)
        self.assertIn('trades', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('sharpe_ratio', results)

if __name__ == '__main__':
    unittest.main()