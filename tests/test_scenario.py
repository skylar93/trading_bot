"""Tests for scenario-based backtesting"""

import unittest
import numpy as np
import pandas as pd
from training.backtesting.base_backtester import BaseBacktester
from training.backtesting.risk_manager import RiskConfig
from training.backtesting.scenario import (
    generate_flash_crash_data,
    generate_low_liquidity_data,
    calculate_scenario_metrics,
    generate_flash_crash_data_deterministic
)
from training.backtesting.risk_aware_backtester import RiskAwareBacktester

class DummyAgent:
    """Dummy agent for testing that implements a simple strategy"""
    def get_action(self, state):
        """Simple strategy: buy dips, sell rips"""
        returns = state['$close'].pct_change()
        if returns.iloc[-1] < -0.05:  # Buy the dip
            return 0.1
        elif returns.iloc[-1] > 0.05:  # Sell the rip
            return -0.1
        return 0.0

class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.risk_config = RiskConfig(
            max_position_size=1.0,
            stop_loss_pct=0.02,
            max_drawdown_pct=0.15,
            daily_trade_limit=1000
        )
        self.backtester = BaseBacktester(
            initial_capital=10000.0,
            risk_config=self.risk_config
        )

    def test_flash_crash_scenario(self):
        """Test flash crash scenario generation and backtesting"""
        crash_params = {
            "length": 1000,
            "crash_at": 500,
            "crash_size": 0.3,  # 30% drop
            "base_price": 100.0
        }
        
        data = generate_flash_crash_data_deterministic(**crash_params)  # Use deterministic version
        self.backtester.data = data
        
        # Verify crash characteristics
        pre_crash = data.iloc[crash_params["crash_at"] - 1]["$close"]  # Price right before crash
        crash_bottom = data.iloc[crash_params["crash_at"]]["$close"]  # Price at start of crash
        self.assertTrue(
            crash_bottom <= pre_crash * (1 - crash_params["crash_size"]),
            "Price should drop by at least the crash size"
        )
        
        # Use a modified strategy that maintains full position until crash
        class CrashTestStrategy(DummyStrategy):
            def get_action(self, window_data):
                """Always maintain full long position until crash"""
                if len(window_data) < 2:
                    return 1.0  # Start with full position
                    
                current_price = window_data["$close"].iloc[-1]
                prev_price = window_data["$close"].iloc[-2]
                price_change = (current_price - prev_price) / prev_price
                
                if price_change < -0.1:  # If big drop detected
                    return -1.0  # Exit position
                return 1.0  # Otherwise maintain full position
        
        # Run backtest with scenario data
        results = self.backtester.run_scenario(
            strategy=CrashTestStrategy(),
            scenario_type="flash_crash",
            **crash_params
        )
        
        self.assertIn("metrics", results)
        self.assertIn("max_drawdown", results["metrics"])
        self.assertTrue(
            results["metrics"]["max_drawdown"] > 0.1,  # Significant drawdown during crash
            "Drawdown should be significant during crash"
        )

    def test_low_liquidity_scenario(self):
        """Test low liquidity scenario generation and backtesting"""
        liq_params = {
            "length": 1000,
            "low_liq_start": 300,
            "low_liq_length": 100,
            "base_price": 100.0,
            "base_volume": 1000.0,
            "volume_reduction": 0.8  # 80% reduction in volume
        }
        
        data = generate_low_liquidity_data(**liq_params)
        self.backtester.data = data
        
        # Verify low liquidity characteristics
        normal_volume = data.iloc[:liq_params["low_liq_start"]]["$volume"].mean()
        low_liq_volume = data.iloc[
            liq_params["low_liq_start"]:
            liq_params["low_liq_start"] + liq_params["low_liq_length"]
        ]["$volume"].mean()
        
        expected_reduction = normal_volume * (1 - liq_params["volume_reduction"])
        self.assertTrue(
            low_liq_volume <= expected_reduction,
            "Volume should drop by specified reduction factor during low liquidity"
        )
        
        # Run backtest with scenario data
        results = self.backtester.run_scenario(
            strategy=DummyStrategy(),
            scenario_type="low_liquidity",
            **liq_params
        )
        
        self.assertIn("metrics", results)
        self.assertIn("trades", results)
        
        # Check trade behavior during low liquidity
        low_liq_trades = [
            t for t in results["trades"]
            if liq_params["low_liq_start"] <= results["timestamps"].index(t["timestamp"]) < 
               liq_params["low_liq_start"] + liq_params["low_liq_length"]
        ]
        
        if low_liq_trades:
            avg_trade_size = np.mean([abs(t["amount"]) for t in low_liq_trades])
            normal_trades = [
                t for t in results["trades"]
                if results["timestamps"].index(t["timestamp"]) < liq_params["low_liq_start"]
            ]
            normal_avg_size = np.mean([abs(t["amount"]) for t in normal_trades]) if normal_trades else 0
            
            self.assertTrue(
                avg_trade_size < normal_avg_size,
                "Trade sizes should be reduced during low liquidity"
            )

    def test_risk_limits_in_crash(self):
        """Test risk management during flash crash"""
        crash_params = {
            "length": 1000,
            "crash_at": 500,
            "crash_size": 0.3,  # Larger crash
            "crash_duration": 3,
            "recovery_duration": 6,
            "base_price": 100.0
        }
        
        data = generate_flash_crash_data(**crash_params)
        self.backtester.data = data
        
        results = self.backtester.run_scenario(
            strategy=DummyStrategy(),
            scenario_type="flash_crash",
            **crash_params
        )
        
        # Verify risk management prevented excessive losses
        max_drawdown = results["metrics"]["max_drawdown"]
        self.assertTrue(
            abs(max_drawdown) <= self.risk_config.max_drawdown_pct,
            f"Risk limits should prevent drawdown exceeding {self.risk_config.max_drawdown_pct}"
        )

    def test_risk_limits_in_low_liquidity(self):
        """Test risk management during low liquidity"""
        liq_params = {
            "length": 1000,
            "low_liq_start": 300,
            "low_liq_length": 100,
            "base_price": 100.0,
            "base_volume": 1000.0,
            "volume_reduction": 0.8
        }
        
        data = generate_low_liquidity_data(**liq_params)
        self.backtester.data = data
        
        results = self.backtester.run_scenario(
            strategy=DummyStrategy(),
            scenario_type="low_liquidity",
            **liq_params
        )
        
        # Verify reduced position sizes during low liquidity
        low_liq_trades = [
            t for t in results["trades"]
            if liq_params["low_liq_start"] <= results["timestamps"].index(t["timestamp"]) < 
               liq_params["low_liq_start"] + liq_params["low_liq_length"]
        ]
        
        if low_liq_trades:
            max_trade_size = max(abs(t["amount"]) for t in low_liq_trades)
            self.assertTrue(
                max_trade_size <= self.risk_config.max_position_size * (1 - liq_params["volume_reduction"]),
                "Position sizes should be reduced proportionally to volume reduction"
            )

class DummyStrategy:
    """A strategy that takes smaller positions and is more responsive to drawdowns"""
    def __init__(self):
        self.last_action = -0.5  # Start with a half short position
        
    def get_action(self, window_data):
        """Return alternating buy/sell actions with smaller sizes"""
        # Check for significant drawdown and exit positions
        if 'portfolio_value' in window_data:
            returns = pd.Series(window_data['portfolio_value']).pct_change()
            drawdown = (returns.max() - returns.iloc[-1]) / returns.max()
            if drawdown > 0.1:  # Exit positions if drawdown > 10%
                return -self.last_action  # Exit current position
        
        self.last_action *= -0.5  # Alternate between 0.5 and -0.5
        return self.last_action

def test_flash_crash_scenario():
    """Test flash crash scenario generation and backtesting"""
    # Generate deterministic flash crash data
    crash_params = {
        "length": 1000,
        "crash_at": 500,
        "crash_size": 0.3,  # 30% drop
        "base_price": 100.0
    }
    
    data = generate_flash_crash_data_deterministic(**crash_params)  # Use deterministic version
    
    # Check crash characteristics
    pre_crash = data.iloc[crash_params["crash_at"] - 1]["$close"]  # Price right before crash
    crash_bottom = data.iloc[crash_params["crash_at"]]["$close"]  # Price at start of crash
    assert crash_bottom <= pre_crash * (1 - crash_params["crash_size"]), "Price should drop by at least the crash size"
    
    # Run backtest with scenario data
    risk_config = RiskConfig(
        max_position_size=1.0,
        stop_loss_pct=0.1,
        max_drawdown_pct=0.3,
        daily_trade_limit=1000
    )
    backtester = BaseBacktester(
        data=data,
        risk_config=risk_config,
        initial_capital=10000.0
    )
    
    # Use a modified DummyStrategy that maintains full position until crash
    class CrashTestStrategy(DummyStrategy):
        def get_action(self, window_data):
            """Always maintain full long position until crash"""
            if len(window_data) < 2:
                return 1.0  # Start with full position
                
            current_price = window_data["$close"].iloc[-1]
            prev_price = window_data["$close"].iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            
            if price_change < -0.1:  # If big drop detected
                return -1.0  # Exit position
            return 1.0  # Otherwise maintain full position
    
    results = backtester.run_scenario(
        strategy=CrashTestStrategy(),
        scenario_type="flash_crash",
        **crash_params
    )
    
    assert "metrics" in results
    assert "max_drawdown" in results["metrics"]
    assert results["metrics"]["max_drawdown"] > 0.2, "Drawdown should be significant during crash"

def test_low_liquidity_scenario():
    """Test low liquidity scenario generation and backtesting"""
    # Generate low liquidity data
    liq_params = {
        "length": 1000,
        "low_liq_start": 300,
        "low_liq_length": 100,
        "base_price": 100.0,
        "base_volume": 1000.0,
        "volume_reduction": 0.8  # 80% reduction in volume
    }
    
    data = generate_low_liquidity_data(**liq_params)
    
    # Check liquidity characteristics
    normal_volume = data.iloc[:liq_params["low_liq_start"]]["$volume"].mean()
    low_liq_volume = data.iloc[
        liq_params["low_liq_start"]:
        liq_params["low_liq_start"] + liq_params["low_liq_length"]
    ]["$volume"].mean()
    
    expected_reduction = normal_volume * (1 - liq_params["volume_reduction"])
    assert low_liq_volume <= expected_reduction, "Volume should drop by specified reduction factor during low liquidity"
    
    # Run backtest with scenario data
    backtester = BaseBacktester(data=data)
    results = backtester.run_scenario(
        strategy=DummyStrategy(),
        scenario_type="low_liquidity",
        **liq_params
    )
    
    assert "metrics" in results
    assert "trades" in results
    
    # Check trade behavior during low liquidity
    low_liq_trades = [
        t for t in results["trades"]
        if liq_params["low_liq_start"] <= results["timestamps"].index(t["timestamp"]) < 
           liq_params["low_liq_start"] + liq_params["low_liq_length"]
    ]
    
    if low_liq_trades:
        avg_trade_size = np.mean([abs(t["amount"]) for t in low_liq_trades])
        normal_trades = [
            t for t in results["trades"]
            if results["timestamps"].index(t["timestamp"]) < liq_params["low_liq_start"]
        ]
        normal_avg_size = np.mean([abs(t["amount"]) for t in normal_trades]) if normal_trades else 0
        
        assert avg_trade_size < normal_avg_size, "Trade sizes should be reduced during low liquidity"

def test_risk_limits_in_crash():
    """Test that risk limits are respected during flash crash"""
    crash_params = {
        "length": 1000,
        "crash_at": 500,
        "crash_size": 0.3,
        "base_price": 100.0
    }
    
    risk_config = RiskConfig(
        max_position_size=1.0,
        stop_loss_pct=0.1,  # 10% stop loss
        max_drawdown_pct=0.2,  # 20% max drawdown
        daily_trade_limit=1000  # Changed from max_trades_per_day
    )
    
    data = generate_flash_crash_data(**crash_params)
    backtester = BaseBacktester(data=data, risk_config=risk_config)
    
    results = backtester.run_scenario(
        strategy=DummyStrategy(),
        scenario_type="flash_crash",
        **crash_params
    )
    
    assert results["metrics"]["max_drawdown"] <= 0.2, "Max drawdown limit should be respected"

def test_risk_limits_in_low_liquidity():
    """Test that risk limits are respected during low liquidity"""
    liq_params = {
        "length": 1000,
        "low_liq_start": 300,
        "low_liq_length": 100,
        "base_price": 100.0,
        "base_volume": 1000.0,
        "volume_reduction": 0.8
    }
    
    risk_config = RiskConfig(
        max_position_size=1.0,
        stop_loss_pct=0.1,  # 10% stop loss
        max_drawdown_pct=0.2,  # 20% max drawdown
        daily_trade_limit=1000  # Changed from max_trades_per_day
    )
    
    data = generate_low_liquidity_data(**liq_params)
    backtester = BaseBacktester(data=data, risk_config=risk_config)
    
    results = backtester.run_scenario(
        strategy=DummyStrategy(),
        scenario_type="low_liquidity",
        **liq_params
    )
    
    assert results["metrics"]["max_drawdown"] <= 0.2, "Max drawdown limit should be respected"
    
    # Check that position sizes are reduced during low liquidity
    low_liq_trades = [
        t for t in results["trades"]
        if 300 <= results["timestamps"].index(t["timestamp"]) < 400
    ]
    
    if low_liq_trades:
        avg_trade_size = np.mean([abs(t["amount"]) for t in low_liq_trades])
        assert avg_trade_size < backtester.max_position * 0.5, "Position sizes should be reduced in low liquidity" 