"""Test risk management system"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from training.utils.risk_management import RiskManager, RiskConfig

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.config = RiskConfig(
            max_position_size=0.2,
            stop_loss_pct=0.02,
            max_drawdown_pct=0.15,
            daily_trade_limit=10,
            min_trade_size=0.01,
            max_leverage=1.0
        )
        self.risk_manager = RiskManager(self.config)
    
    def test_position_sizing(self):
        """Test position size calculation"""
        portfolio_value = 10000
        price = 100
        
        # Basic position size
        size = self.risk_manager.calculate_position_size(portfolio_value, price)
        self.assertEqual(size, portfolio_value * self.config.max_position_size)
        
        # Position size with volatility scaling
        vol_size = self.risk_manager.calculate_position_size(portfolio_value, price, volatility=0.5)
        self.assertTrue(vol_size < size)  # Higher volatility should reduce size
        
        # Minimum size check
        small_portfolio = 100
        small_size = self.risk_manager.calculate_position_size(small_portfolio, price)
        self.assertEqual(small_size, 0.0)  # Below minimum, should return 0
    
    def test_trade_limits(self):
        """Test daily trade limits"""
        timestamp = pd.Timestamp('2024-01-01 10:00:00')
        
        # Should allow trades up to limit
        for _ in range(self.config.daily_trade_limit):
            self.assertTrue(self.risk_manager.check_trade_limits(timestamp))
            self.risk_manager.update_trade_counter(timestamp)
        
        # Should reject after limit reached
        self.assertFalse(self.risk_manager.check_trade_limits(timestamp))
        
        # Should reset on new day
        next_day = timestamp + timedelta(days=1)
        self.assertTrue(self.risk_manager.check_trade_limits(next_day))
    
    def test_stop_loss(self):
        """Test stop loss functionality"""
        trade_id = 'test_trade'
        entry_price = 100
        
        # Test long position stop loss
        stop_price = self.risk_manager.set_stop_loss(trade_id, entry_price, 'long')
        expected_stop = entry_price * (1 - self.config.stop_loss_pct)
        self.assertEqual(stop_price, expected_stop)
        
        # Test stop loss hit
        self.assertTrue(self.risk_manager.check_stop_loss(trade_id, expected_stop - 1))
        self.assertFalse(self.risk_manager.check_stop_loss(trade_id, expected_stop + 1))
        
        # Test short position stop loss
        short_stop = self.risk_manager.set_stop_loss(trade_id, entry_price, 'short')
        expected_short_stop = entry_price * (1 + self.config.stop_loss_pct)
        self.assertEqual(short_stop, expected_short_stop)
    
    def test_drawdown_monitoring(self):
        """Test drawdown calculations"""
        # Initial portfolio value
        initial_value = 10000
        drawdown, exceeded = self.risk_manager.update_drawdown(initial_value)
        self.assertEqual(drawdown, 0.0)
        self.assertFalse(exceeded)
        
        # Small drawdown
        small_drop = initial_value * 0.95
        drawdown, exceeded = self.risk_manager.update_drawdown(small_drop)
        self.assertEqual(drawdown, 0.05)
        self.assertFalse(exceeded)
        
        # Exceeding max drawdown
        big_drop = initial_value * 0.8
        drawdown, exceeded = self.risk_manager.update_drawdown(big_drop)
        self.assertEqual(drawdown, 0.2)
        self.assertTrue(exceeded)
    
    def test_leverage_limits(self):
        """Test leverage limits"""
        position_size = 1000
        
        # No leverage
        adjusted = self.risk_manager.adjust_for_leverage(position_size, current_leverage=0.0)
        self.assertEqual(adjusted, position_size)
        
        # At max leverage
        adjusted = self.risk_manager.adjust_for_leverage(position_size, current_leverage=1.0)
        self.assertEqual(adjusted, 0.0)
        
        # Partial leverage available
        adjusted = self.risk_manager.adjust_for_leverage(position_size, current_leverage=0.5)
        self.assertEqual(adjusted, position_size * 0.5)
    
    def test_trade_signal_processing(self):
        """Test complete trade signal processing"""
        timestamp = pd.Timestamp('2024-01-01 10:00:00')
        portfolio_value = 10000
        price = 100
        
        # Normal trade
        result = self.risk_manager.process_trade_signal(
            timestamp, portfolio_value, price
        )
        self.assertTrue(result['allowed'])
        self.assertTrue('position_size' in result)
        
        # Excessive drawdown
        self.risk_manager.update_drawdown(portfolio_value * 0.8)  # Create large drawdown
        result = self.risk_manager.process_trade_signal(
            timestamp, portfolio_value * 0.8, price
        )
        self.assertFalse(result['allowed'])
        self.assertEqual(result['reason'], 'max_drawdown_exceeded')

if __name__ == '__main__':
    unittest.main()