"""
Advanced backtesting scenarios for trading bot.
Implements complex market conditions and edge cases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class ScenarioGenerator:
    def __init__(self, base_price: float = 100.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
    
    def generate_flash_crash(self, 
                           length: int = 100, 
                           crash_idx: int = 50, 
                           crash_size: float = 0.15) -> pd.DataFrame:
        """Generate a flash crash scenario
        
        Args:
            length: Number of periods
            crash_idx: When the crash occurs
            crash_size: Size of the crash as a percentage
        
        Returns:
            DataFrame with OHLCV data
        """
        # Generate base prices
        prices = np.random.normal(0, self.volatility, length).cumsum()
        prices = self.base_price * np.exp(prices)
        
        # Add flash crash
        crash_impact = prices[crash_idx] * crash_size
        prices[crash_idx:crash_idx+3] -= crash_impact
        prices[crash_idx+3:] -= crash_impact * 0.7  # Partial recovery
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='1min'),
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, length)),
            'low': prices * (1 - np.random.uniform(0, 0.01, length)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, length)
        })
        
        return df
    
    def generate_low_liquidity(self,
                             length: int = 100,
                             low_liq_start: int = 30,
                             low_liq_length: int = 20) -> pd.DataFrame:
        """Generate a low liquidity scenario
        
        Args:
            length: Number of periods
            low_liq_start: When low liquidity starts
            low_liq_length: How long low liquidity lasts
        
        Returns:
            DataFrame with OHLCV data
        """
        # Generate base prices with higher volatility during low liquidity
        prices = np.zeros(length)
        volatility = np.ones(length) * self.volatility
        volatility[low_liq_start:low_liq_start+low_liq_length] *= 3
        
        for i in range(1, length):
            prices[i] = prices[i-1] + np.random.normal(0, volatility[i])
        
        prices = self.base_price * np.exp(prices)
        
        # Generate volumes with low liquidity period
        volumes = np.random.uniform(100, 1000, length)
        volumes[low_liq_start:low_liq_start+low_liq_length] *= 0.1
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='1min'),
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, length)),
            'low': prices * (1 - np.random.uniform(0, 0.01, length)),
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    def generate_choppy_market(self,
                             length: int = 100,
                             chop_intensity: float = 2.0) -> pd.DataFrame:
        """Generate a choppy market scenario with rapid price reversals
        
        Args:
            length: Number of periods
            chop_intensity: Intensity of the choppy behavior
        
        Returns:
            DataFrame with OHLCV data
        """
        # Generate oscillating prices
        t = np.linspace(0, 4*np.pi, length)
        trend = self.base_price + np.sin(t) * self.base_price * 0.05
        noise = np.random.normal(0, self.volatility * chop_intensity, length)
        prices = trend + noise
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='1min'),
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, length)),
            'low': prices * (1 - np.random.uniform(0, 0.01, length)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, length)
        })
        
        return df
    
    def combine_scenarios(self, scenarios: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple scenario DataFrames into one"""
        return pd.concat(scenarios, ignore_index=True)

class ScenarioTester:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.scenario_generator = ScenarioGenerator()
    
    def test_scenario(self, 
                     scenario_data: pd.DataFrame, 
                     initial_balance: float = 10000.0) -> Dict:
        """Test a specific scenario
        
        Args:
            scenario_data: DataFrame with OHLCV data
            initial_balance: Starting balance
            
        Returns:
            Dict with test results
        """
        # Initialize environment with scenario data
        self.env.reset(
            data=scenario_data,
            initial_balance=initial_balance
        )
        
        done = False
        total_reward = 0
        trades = []
        
        while not done:
            state = self.env.get_observation()
            action = self.agent.compute_action(state)
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if info.get('trade_executed'):
                trades.append(info['trade_info'])
        
        return {
            'total_reward': total_reward,
            'final_balance': self.env.balance,
            'trades': trades,
            'max_drawdown': self.env.max_drawdown,
            'sharpe_ratio': self.env.sharpe_ratio
        }
    
    def run_all_scenarios(self) -> Dict:
        """Run all available test scenarios"""
        scenarios = {
            'flash_crash': self.scenario_generator.generate_flash_crash(),
            'low_liquidity': self.scenario_generator.generate_low_liquidity(),
            'choppy_market': self.scenario_generator.generate_choppy_market()
        }
        
        results = {}
        for name, scenario_data in scenarios.items():
            results[name] = self.test_scenario(scenario_data)
        
        return results