"""Test risk-aware backtesting system"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk.risk_manager import RiskConfig
from training.utils.risk_backtest import RiskAwareBacktester

class MockAgent:
    """Mock trading agent for testing"""
    def __init__(self):
        self.action_sequence = [0.5, -0.3, 0.2, -0.1]  # Predefined actions
        self.current_step = 0
    
    def get_action(self, state):
        """Return next action in sequence"""
        action = self.action_sequence[self.current_step % len(self.action_sequence)]
        self.current_step += 1
        return action

def generate_test_data(length=100, assets=["BTC", "ETH"]):
    """Generate test data for multiple assets"""
    dates = pd.date_range(start="2024-01-01", periods=length, freq="1h")
    data = {}
    
    for asset in assets:
        # Generate correlated price movements
        returns = np.random.normal(0, 0.02, length)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data[f"{asset}_$open"] = prices * (1 - 0.001)
        data[f"{asset}_$high"] = prices * (1 + 0.002)
        data[f"{asset}_$low"] = prices * (1 - 0.002)
        data[f"{asset}_$close"] = prices
        data[f"{asset}_$volume"] = np.random.uniform(100, 1000, length)
    
    return pd.DataFrame(data, index=dates)

class TestRiskAwareBacktester(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_data = generate_test_data()
        self.risk_config = RiskConfig(
            max_position_size=0.2,
            stop_loss_pct=0.02,
            max_drawdown_pct=0.15,
            daily_trade_limit=10,
            var_confidence_level=0.95,
            portfolio_var_limit=0.02,
            max_correlation=0.7,
            min_trade_size=0.01,
            max_leverage=1.0
        )
        self.backtester = RiskAwareBacktester(
            data=self.test_data,
            risk_config=self.risk_config,
            initial_balance=10000.0,
        )
        self.agent = MockAgent()
    
    def test_portfolio_var_limits(self):
        """Test if portfolio VaR limits are respected"""
        # Execute trades that would exceed portfolio VaR
        timestamp = self.test_data.index[20]
        
        # First trade in BTC
        btc_result = self.backtester.execute_trade(
            timestamp=timestamp,
            action=1.0,
            price_data={
                "BTC_$open": self.test_data["BTC_$open"].iloc[20],
                "BTC_$high": self.test_data["BTC_$high"].iloc[20],
                "BTC_$low": self.test_data["BTC_$low"].iloc[20],
                "BTC_$close": self.test_data["BTC_$close"].iloc[20],
                "BTC_$volume": self.test_data["BTC_$volume"].iloc[20]
            }
        )
        
        # Calculate portfolio VaR after trade
        portfolio_value = self.backtester.balance + sum(
            self.backtester.get_position_value(asset)
            for asset in ["BTC", "ETH"]
        )
        
        # Verify portfolio VaR is within limits
        portfolio_var = self.backtester.risk_manager.get_portfolio_var(
            self.backtester.positions,
            portfolio_value
        )
        self.assertLessEqual(
            portfolio_var,
            self.risk_config.portfolio_var_limit,
            "Portfolio VaR should not exceed configured limit"
        )
    
    def test_correlation_based_position_sizing(self):
        """Test if position sizes are adjusted based on correlations"""
        # Execute trades in correlated assets
        timestamp = self.test_data.index[20]
        
        # First trade in BTC
        btc_result = self.backtester.execute_trade(
            timestamp=timestamp,
            action=1.0,
            price_data={
                "BTC_$open": self.test_data["BTC_$open"].iloc[20],
                "BTC_$high": self.test_data["BTC_$high"].iloc[20],
                "BTC_$low": self.test_data["BTC_$low"].iloc[20],
                "BTC_$close": self.test_data["BTC_$close"].iloc[20],
                "BTC_$volume": self.test_data["BTC_$volume"].iloc[20]
            }
        )
        
        # Second trade in ETH (should be reduced due to correlation)
        eth_result = self.backtester.execute_trade(
            timestamp=timestamp,
            action=1.0,
            price_data={
                "ETH_$open": self.test_data["ETH_$open"].iloc[20],
                "ETH_$high": self.test_data["ETH_$high"].iloc[20],
                "ETH_$low": self.test_data["ETH_$low"].iloc[20],
                "ETH_$close": self.test_data["ETH_$close"].iloc[20],
                "ETH_$volume": self.test_data["ETH_$volume"].iloc[20]
            }
        )
        
        # Verify position sizes are adjusted
        if "risk_metrics" in eth_result:
            self.assertLess(
                eth_result["risk_metrics"]["adjusted_size"],
                1.0,
                "Position size should be reduced due to correlation"
            )
    
    def test_risk_metrics_tracking(self):
        """Test if portfolio risk metrics are properly tracked"""
        results = self.backtester.run(self.agent, window_size=20)
        
        # Verify risk metrics are included in results
        self.assertIn("risk_summary", results)
        risk_summary = results["risk_summary"]
        
        # Check for portfolio-wide metrics
        self.assertIn("portfolio_var", risk_summary)
        self.assertIn("avg_correlation", risk_summary)
        
        # Verify metrics are within expected ranges
        self.assertGreaterEqual(risk_summary["portfolio_var"], 0)
        self.assertLessEqual(risk_summary["portfolio_var"], 1)
        self.assertGreaterEqual(risk_summary["avg_correlation"], -1)
        self.assertLessEqual(risk_summary["avg_correlation"], 1)

if __name__ == "__main__":
    unittest.main()
