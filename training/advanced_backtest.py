import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
from .backtest import Backtester
from .evaluation import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScenarioBacktester(Backtester):
    """Advanced backtester with scenario generation capabilities"""
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialize with no data - will be generated per scenario"""
        self.initial_balance = initial_balance
        self.transaction_fee = 0.001
        self.slippage = 0.001
    
    def generate_flash_crash_data(self, 
                                length: int = 1000,
                                crash_at: int = 500,
                                crash_size: float = 0.15,
                                base_price: float = 100.0) -> pd.DataFrame:
        """Generate flash crash scenario data"""
        # Generate base price series with some volatility
        timestamps = pd.date_range(start='2024-01-01', periods=length, freq='5min')
        returns = np.random.normal(0, 0.001, size=length)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add flash crash
        crash_impact = prices[crash_at] * crash_size
        prices[crash_at:crash_at+3] -= crash_impact
        prices[crash_at+3:] -= crash_impact * 0.7  # Partial recovery
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.002, length)),
            'low': prices * (1 - np.random.uniform(0, 0.002, length)),
            'close': prices,
            'volume': np.random.uniform(100000, 200000, length)
        }, index=timestamps)
        
        return data
    
    def generate_low_liquidity_data(self,
                                  length: int = 1000,
                                  low_liq_start: int = 300,
                                  low_liq_length: int = 100,
                                  base_price: float = 100.0) -> pd.DataFrame:
        """Generate low liquidity scenario data"""
        timestamps = pd.date_range(start='2024-01-01', periods=length, freq='5min')
        returns = np.random.normal(0, 0.001, size=length)
        
        # Increase volatility during low liquidity period
        returns[low_liq_start:low_liq_start+low_liq_length] *= 3
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate volumes with low liquidity period
        volumes = np.random.uniform(100000, 200000, length)
        volumes[low_liq_start:low_liq_start+low_liq_length] *= 0.1
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.002, length)),
            'low': prices * (1 - np.random.uniform(0, 0.002, length)),
            'close': prices,
            'volume': volumes
        }, index=timestamps)
        
        return data
    
    def run_flash_crash_scenario(self,
                              agent: Any,
                              window_size: int = 20,
                              verbose: bool = True) -> Dict:
        """Run backtest with flash crash scenario"""
        # Generate flash crash data
        self.data = self.generate_flash_crash_data()
        
        # Run standard backtest
        results = self.run(agent, window_size, verbose)
        
        # Add scenario-specific metrics
        max_drawdown_idx = np.argmax(np.maximum.accumulate(results['portfolio_values']) - results['portfolio_values'])
        recovery_time = len(results['portfolio_values']) - max_drawdown_idx
        
        results['scenario_metrics'] = {
            'max_drawdown_idx': max_drawdown_idx,
            'recovery_time_periods': recovery_time,
            'survived_crash': results['portfolio_values'][-1] > self.initial_balance * 0.5
        }
        
        return results
    
    def run_low_liquidity_scenario(self,
                                agent: Any,
                                window_size: int = 20,
                                verbose: bool = True) -> Dict:
        """Run backtest with low liquidity scenario"""
        # Generate low liquidity data
        self.data = self.generate_low_liquidity_data()
        
        # Run standard backtest
        results = self.run(agent, window_size, verbose)
        
        # Calculate liquidity-specific metrics
        trade_costs = [trade.get('costs', 0) for trade in results['trades']]
        avg_cost = np.mean(trade_costs) if trade_costs else 0
        
        results['scenario_metrics'] = {
            'avg_trade_cost': avg_cost,
            'trade_count_low_liq': len([t for t in results['trades'] 
                                      if 300 <= results['timestamps'].index(t['exit_time']) < 400])
        }
        
        return results

    def plot_scenario_results(self, results: Dict, scenario_type: str, save_path: Optional[str] = None):
        """Enhanced plotting for scenario results"""
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Create base plots using parent class
        super().plot_results(results)
        
        # Add scenario-specific annotations
        if scenario_type == 'flash_crash':
            crash_idx = results['scenario_metrics']['max_drawdown_idx']
            plt.axvline(x=results['timestamps'][crash_idx], color='r', linestyle='--', 
                       label='Flash Crash')
            plt.text(results['timestamps'][crash_idx], plt.ylim()[1], 
                    'Flash Crash', rotation=90)
            
        elif scenario_type == 'low_liquidity':
            plt.axvspan(results['timestamps'][300], results['timestamps'][400], 
                       alpha=0.2, color='yellow', label='Low Liquidity')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def save_scenario_results(self, results: Dict, scenario_type: str, save_dir: str):
        """Save scenario-specific results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save standard results
        super().save_results(results, save_dir)
        
        # Save scenario-specific metrics
        scenario_df = pd.DataFrame([results['scenario_metrics']])
        scenario_df.to_csv(save_dir / f'{scenario_type}_metrics.csv', index=False)
        
        # Save enhanced plots
        self.plot_scenario_results(results, scenario_type, 
                                 save_path=str(save_dir / f'{scenario_type}_plot.png'))
        
        logger.info(f"Scenario results saved to {save_dir}")