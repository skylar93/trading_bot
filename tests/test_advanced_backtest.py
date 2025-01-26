import unittest
import numpy as np
import pandas as pd
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.scenario import (
    generate_flash_crash_data,
    generate_low_liquidity_data,
    calculate_flash_crash_metrics,
    calculate_low_liquidity_metrics
)
from agents.strategies.single.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class MockAgent:
    """Mock agent for testing"""

    def get_action(self, observation):
        return np.random.uniform(-1, 1)


class TestScenarioBacktesting(unittest.TestCase):
    def setUp(self):
        self.backtester = BaseBacktester(initial_capital=10000.0)
        self.backtester.logger = logging.getLogger(
            self.backtester.__class__.__name__
        )
        self.agent = MockAgent()

    def test_flash_crash_data_generation(self):
        """Test flash crash data generation"""
        data = generate_flash_crash_data(
            length=1000, crash_at=500, crash_size=0.15
        )

        self.assertEqual(len(data), 1000)
        self.assertTrue(
            all(
                col in data.columns
                for col in ["$open", "$high", "$low", "$close", "$volume"]
            )
        )

        # Verify crash occurs
        pre_crash = data["$close"].iloc[499]
        post_crash = data["$close"].iloc[500]
        self.assertTrue(post_crash < pre_crash * 0.9)

    def test_low_liquidity_data_generation(self):
        """Test low liquidity data generation"""
        data = generate_low_liquidity_data(
            length=1000, low_liq_start=300, low_liq_length=100
        )

        self.assertEqual(len(data), 1000)

        # Verify low liquidity period
        normal_volume = data["$volume"].iloc[0:300].mean()
        low_liq_volume = data["$volume"].iloc[300:400].mean()
        self.assertTrue(low_liq_volume < normal_volume * 0.2)

    def test_flash_crash_scenario(self):
        """Test full flash crash scenario backtest"""
        results = self.backtester.run_scenario(
            strategy=self.agent,
            scenario_type="flash_crash",
            length=1000,
            crash_at=500,
            crash_size=0.15
        )

        self.assertIn("scenario_metrics", results)
        self.assertIn("max_drawdown_idx", results["scenario_metrics"])
        self.assertIn("recovery_time_periods", results["scenario_metrics"])
        self.assertIn("survived_crash", results["scenario_metrics"])

    def test_low_liquidity_scenario(self):
        """Test full low liquidity scenario backtest"""
        results = self.backtester.run_scenario(
            strategy=self.agent,
            scenario_type="low_liquidity",
            length=1000,
            low_liq_start=300,
            low_liq_length=100
        )

        self.assertIn("scenario_metrics", results)
        self.assertIn("avg_trade_cost", results["scenario_metrics"])
        self.assertIn("trade_count_low_liq", results["scenario_metrics"])


if __name__ == "__main__":
    unittest.main()
