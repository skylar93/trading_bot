"""Test risk-aware backtesting system"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.utils.risk_backtest import RiskAwareBacktester
from training.utils.risk_management import RiskConfig

class MockAgent:
    """Mock trading agent for testing"""
    def get_action(self, observation):
        return np.random.uniform(-1, 1)

def generate_test_data(length: int = 1000) -> pd.DataFrame:
    """Generate test market data"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='h')
    
    # Generate price process
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, length)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame with $ prefixed columns
    data = pd.DataFrame({
        '$open': price,
        '$high': price * (1 + np.random.uniform(0, 0.001, length)),
        '$low': price * (1 - np.random.uniform(0, 0.001, length)),
        '$close': price,
        '$volume': np.random.uniform(1000, 5000, length)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['$high'] = data[['$open', '$high', '$low', '$close']].max(axis=1)
    data['$low'] = data[['$open', '$high', '$low', '$close']].min(axis=1)
    
    return data

class TestRiskAwareBacktester(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_data = generate_test_data()
        self.risk_config = RiskConfig(
            max_position_size=0.2,
            stop_loss_pct=0.02,
            max_drawdown_pct=0.15,
            daily_trade_limit=10
        )
        self.backtester = RiskAwareBacktester(
            data=self.test_data,
            risk_config=self.risk_config,
            initial_balance=10000.0
        )
        self.agent = MockAgent()
        
    def test_volatility_calculation(self):
        """Test volatility estimation"""
        vol = self.backtester.calculate_volatility()
        self.assertTrue(0 <= vol <= 1)  # Reasonable volatility range
        
    def test_leverage_calculation(self):
        """Test leverage calculation"""
        leverage = self.backtester.get_current_leverage()
        self.assertEqual(leverage, 0.0)  # Should be zero at start
        
    def test_risk_adjusted_trading(self):
        """Test trading with risk management"""
        timestamp = self.test_data.index[0]
        price_data = {
            '$open': 100,
            '$high': 101,
            '$low': 99,
            '$close': 100,
            '$volume': 1000
        }
        
        # Test large trade gets adjusted
        result = self.backtester.execute_trade(timestamp, 1.0, price_data)
        self.assertIn('risk_metrics', result)
        self.assertTrue(result['risk_metrics']['adjusted_size'] <= 
                       self.risk_config.max_position_size)
        
    def test_backtest_with_risk_management(self):
        """Test full backtest with risk management"""
        results = self.backtester.run(self.agent, window_size=20)
        
        # Check risk summary exists
        self.assertIn('risk_summary', results)
        
        # Verify risk limits were respected
        self.assertTrue(results['risk_summary']['max_leverage'] <= 
                       self.risk_config.max_leverage)
        self.assertTrue(results['risk_summary']['max_drawdown'] <= 
                       self.risk_config.max_drawdown_pct)
        
        # Check trade sizes
        for trade in results['trades']:
            if 'risk_metrics' in trade:
                self.assertTrue(trade['risk_metrics']['adjusted_size'] <= 
                              self.risk_config.max_position_size)

if __name__ == '__main__':
    unittest.main()