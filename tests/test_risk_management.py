"""Test risk management system"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from risk.risk_manager import RiskManager, RiskConfig


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.config = RiskConfig(
            max_position_size=0.2,
            stop_loss_pct=0.02,
            max_drawdown_pct=0.15,
            daily_trade_limit=10,
            min_trade_size=0.01,
            max_leverage=1.0,
            volatility_lookback=20,
            risk_free_rate=0.02,
        )
        self.risk_manager = RiskManager(self.config)

        # Create sample price data for volatility calculation
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        self.sample_prices = pd.Series(
            np.random.lognormal(0, 0.02, 100), index=dates
        )

    def test_position_sizing(self):
        """Test position size calculation"""
        portfolio_value = 10000
        price = 100

        # Calculate volatility
        returns = self.sample_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        # Basic position size without volatility
        size = self.risk_manager.calculate_position_size(
            portfolio_value=portfolio_value, price=price, volatility=None
        )
        expected_size = portfolio_value * self.config.max_position_size
        self.assertAlmostEqual(size, expected_size, delta=0.01)

        # Position size with volatility scaling
        vol_size = self.risk_manager.calculate_position_size(
            portfolio_value=portfolio_value, price=price, volatility=volatility
        )
        self.assertTrue(
            vol_size < size
        )  # Higher volatility should reduce size

        # Test minimum size threshold
        small_portfolio = 100  # Very small portfolio
        small_size = self.risk_manager.calculate_position_size(
            portfolio_value=small_portfolio, price=price, volatility=None
        )
        self.assertTrue(
            small_size <= small_portfolio * self.config.max_position_size
        )

    def test_trade_limits(self):
        """Test daily trade limits"""
        timestamp = pd.Timestamp("2024-01-01 10:00:00")

        # Should allow trades up to limit
        for i in range(self.config.daily_trade_limit):
            self.assertTrue(
                self.risk_manager.check_trade_limits(timestamp),
                f"Trade {i+1} should be allowed",
            )
            self.risk_manager.update_trade_counter(timestamp)

        # Should reject after limit reached
        self.assertFalse(
            self.risk_manager.check_trade_limits(timestamp),
            "Trade should be rejected after limit reached",
        )

        # Should reset on new day
        next_day = timestamp + pd.Timedelta(days=1)
        self.assertTrue(
            self.risk_manager.check_trade_limits(next_day),
            "Trade counter should reset on new day",
        )

    def test_stop_loss(self):
        """Test stop loss calculation"""
        entry_price = 100
        position_size = 1.0

        # Long position stop loss
        long_stop = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price, position_size=position_size, is_long=True
        )
        expected_long_stop = entry_price * (1 - self.config.stop_loss_pct)
        self.assertAlmostEqual(
            long_stop,
            expected_long_stop,
            delta=0.01,
            msg="Long position stop loss calculation incorrect",
        )

        # Short position stop loss
        short_stop = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price, position_size=position_size, is_long=False
        )
        expected_short_stop = entry_price * (1 + self.config.stop_loss_pct)
        self.assertAlmostEqual(
            short_stop,
            expected_short_stop,
            delta=0.01,
            msg="Short position stop loss calculation incorrect",
        )

    def test_drawdown_monitoring(self):
        """Test drawdown monitoring"""
        initial_value = 10000
        current_value = initial_value

        # No drawdown
        self.assertFalse(
            self.risk_manager.check_max_drawdown(initial_value, current_value),
            "Should not trigger max drawdown when no drawdown exists",
        )

        # Small drawdown
        current_value = initial_value * 0.95  # 5% drawdown
        self.assertFalse(
            self.risk_manager.check_max_drawdown(initial_value, current_value),
            "Should not trigger max drawdown for small drawdown",
        )

        # Max drawdown exceeded
        current_value = initial_value * (
            1 - self.config.max_drawdown_pct * 1.1
        )
        self.assertTrue(
            self.risk_manager.check_max_drawdown(initial_value, current_value),
            "Should trigger max drawdown when threshold exceeded",
        )

    def test_leverage_limits(self):
        """Test leverage limits"""
        portfolio_value = 10000
        position_value = portfolio_value * self.config.max_leverage * 0.5

        # Test within limits
        self.assertTrue(
            self.risk_manager.check_leverage_limits(
                portfolio_value, position_value
            ),
            "Should allow position within leverage limits",
        )

        # Test exceeding limits
        large_position = portfolio_value * self.config.max_leverage * 1.1
        self.assertFalse(
            self.risk_manager.check_leverage_limits(
                portfolio_value, large_position
            ),
            "Should reject position exceeding leverage limits",
        )

    def test_trade_signal_processing(self):
        """Test trade signal processing"""
        # Valid signal
        valid_signal = {
            "timestamp": pd.Timestamp("2024-01-01 10:00:00"),
            "type": "buy",
            "price": 100.0,
            "size": 1.0,
        }
        self.assertTrue(
            self.risk_manager.process_trade_signal(valid_signal),
            "Valid trade signal should be accepted",
        )

        # Invalid signal (missing fields)
        invalid_signal = {
            "timestamp": pd.Timestamp("2024-01-01 10:00:00"),
            "type": "buy",
        }
        self.assertFalse(
            self.risk_manager.process_trade_signal(invalid_signal),
            "Invalid trade signal should be rejected",
        )

        # Invalid signal (negative price)
        invalid_price_signal = {
            "timestamp": pd.Timestamp("2024-01-01 10:00:00"),
            "type": "buy",
            "price": -100.0,
            "size": 1.0,
        }
        self.assertFalse(
            self.risk_manager.process_trade_signal(invalid_price_signal),
            "Signal with negative price should be rejected",
        )

        # Invalid signal (zero size)
        invalid_size_signal = {
            "timestamp": pd.Timestamp("2024-01-01 10:00:00"),
            "type": "buy",
            "price": 100.0,
            "size": 0.0,
        }
        self.assertFalse(
            self.risk_manager.process_trade_signal(invalid_size_signal),
            "Signal with zero size should be rejected",
        )


if __name__ == "__main__":
    unittest.main()
