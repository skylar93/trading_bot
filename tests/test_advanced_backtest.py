import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.advanced_backtest import ScenarioBacktester
from agents.strategies.ppo_agent import PPOAgent

class MockAgent:
    """Mock agent for testing"""
    def get_action(self, observation):
        return np.random.uniform(-1, 1)

class TestScenarioBacktester(unittest.TestCase):
    def setUp(self):
        self.backtester = ScenarioBacktester()
        self.agent = MockAgent()
    
    def test_flash_crash_data_generation(self):
        """Test flash crash data generation"""
        data = self.backtester.generate_flash_crash_data(
            length=1000,
            crash_at=500,
            crash_size=0.15
        )
        
        self.assertEqual(len(data), 1000)
        self.assertTrue(all(col in data.columns 
                          for col in ['open', 'high', 'low', 'close', 'volume']))
        
        # Verify crash occurs
        pre_crash = data['close'].iloc[499]
        post_crash = data['close'].iloc[500]
        self.assertTrue(post_crash < pre_crash * 0.9)
    
    def test_low_liquidity_data_generation(self):
        """Test low liquidity data generation"""
        data = self.backtester.generate_low_liquidity_data(
            length=1000,
            low_liq_start=300,
            low_liq_length=100
        )
        
        self.assertEqual(len(data), 1000)
        
        # Verify low liquidity period
        normal_volume = data['volume'].iloc[0:300].mean()
        low_liq_volume = data['volume'].iloc[300:400].mean()
        self.assertTrue(low_liq_volume < normal_volume * 0.2)
    
    def test_flash_crash_scenario(self):
        """Test full flash crash scenario backtest"""
        results = self.backtester.run_flash_crash_scenario(self.agent)
        
        self.assertIn('scenario_metrics', results)
        self.assertIn('max_drawdown_idx', results['scenario_metrics'])
        self.assertIn('recovery_time_periods', results['scenario_metrics'])
        self.assertIn('survived_crash', results['scenario_metrics'])
    
    def test_low_liquidity_scenario(self):
        """Test full low liquidity scenario backtest"""
        results = self.backtester.run_low_liquidity_scenario(self.agent)
        
        self.assertIn('scenario_metrics', results)
        self.assertIn('avg_trade_cost', results['scenario_metrics'])
        self.assertIn('trade_count_low_liq', results['scenario_metrics'])

if __name__ == '__main__':
    unittest.main()