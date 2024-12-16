import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
from .evaluation import TradingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """Cryptocurrency trading strategy backtester"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 transaction_fee: float = 0.001,
                 slippage: float = 0.001):
        """
        Initialize Backtester
        
        Args:
            data: DataFrame with OHLCV data and features
            initial_balance: Initial portfolio balance
            transaction_fee: Trading fee as a decimal
            slippage: Expected slippage as a decimal
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        self.current_trade = None
    
    def _calculate_transaction_costs(self, price: float, size: float) -> float:
        """Calculate transaction costs including fees and slippage"""
        fee = price * size * self.transaction_fee
        slip = price * size * self.slippage
        return fee + slip
    
    def execute_trade(self, 
                     timestamp: datetime,
                     action: float,  # -1 to 1
                     price_data: Dict[str, float]) -> Dict:
        """
        Execute a trade
        
        Args:
            timestamp: Current timestamp
            action: Trading action (-1 to 1, where -1 is full sell, 1 is full buy)
            price_data: Dictionary with current prices (open, high, low, close)
            
        Returns:
            Dictionary with trade results
        """
        if abs(action) < 1e-5:  # No trade
            return {
                'timestamp': timestamp,
                'action': 0,
                'price': price_data['close'],
                'pnl': 0,
                'costs': 0,
                'balance': self.balance,
                'position': self.position
            }
        
        # Determine trade direction and size
        is_buy = action > 0
        price = price_data['close']  # Using close price for simplicity
        
        if is_buy:
            # Calculate maximum possible position size
            max_size = self.balance / (price * (1 + self.transaction_fee + self.slippage))
            size = max_size * abs(action)
            
            # Calculate costs and update position
            costs = self._calculate_transaction_costs(price, size)
            total_cost = (price * size) + costs
            
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.position += size
                
                # Record trade entry
                self.current_trade = {
                    'entry_time': timestamp,
                    'entry_price': price,
                    'size': size,
                    'type': 'long'
                }
        else:
            # Selling
            size = self.position * abs(action)
            costs = self._calculate_transaction_costs(price, size)
            
            self.position -= size
            proceeds = (price * size) - costs
            self.balance += proceeds
            
            # Record trade exit if there was an entry
            if self.current_trade is not None:
                trade = {
                    'entry_time': self.current_trade['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': self.current_trade['entry_price'],
                    'exit_price': price,
                    'size': size,
                    'type': self.current_trade['type'],
                    'pnl': proceeds - (self.current_trade['entry_price'] * size),
                    'return': (price / self.current_trade['entry_price'] - 1) * 100
                }
                self.trades.append(trade)
                self.current_trade = None
        
        # Calculate current portfolio value
        portfolio_value = self.balance + (self.position * price)
        self.portfolio_values.append(portfolio_value)
        
        return {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'size': size,
            'costs': costs,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value
        }
    
    def run(self, 
            agent: Any,
            window_size: int = 20,
            verbose: bool = True) -> Dict:
        """
        Run backtest
        
        Args:
            agent: Trading agent with get_action(state) method
            window_size: Size of the observation window
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results
        """
        self.reset()
        
        timestamps = []
        portfolio_values_with_time = []
        
        for i in range(window_size, len(self.data)):
            # Prepare state
            observation = self.data.iloc[i-window_size:i]
            timestamp = self.data.index[i]
            
            # Get action from agent
            action = agent.get_action(observation)
            
            # Execute trade
            price_data = {
                'open': self.data.iloc[i]['open'],
                'high': self.data.iloc[i]['high'],
                'low': self.data.iloc[i]['low'],
                'close': self.data.iloc[i]['close']
            }
            
            result = self.execute_trade(timestamp, action, price_data)
            
            # Record timestamp and portfolio value
            timestamps.append(timestamp)
            portfolio_values_with_time.append(
                (timestamp, result['portfolio_value'])
            )
            
            if verbose and i % 100 == 0:
                logger.info(f"Progress: {i}/{len(self.data)} - "
                          f"Portfolio Value: {result['portfolio_value']:.2f}")
        
        # Calculate performance metrics
        portfolio_values = np.array(self.portfolio_values)
        metrics = TradingMetrics.evaluate_strategy(portfolio_values, self.trades)
        
        return {
            'metrics': metrics,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'portfolio_values_with_time': portfolio_values_with_time,
            'timestamps': timestamps
        }
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot backtest results
        
        Args:
            results: Dictionary with backtest results
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
            
            # Prepare time series data
            times = [t for t, _ in results['portfolio_values_with_time']]
            values = [v for _, v in results['portfolio_values_with_time']]
            
            # Plot portfolio value
            ax1.plot(times, values, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value ($)')
            ax1.legend()
            ax1.grid(True)
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Plot drawdown
            running_max = np.maximum.accumulate(values)
            drawdown = np.array(values) / running_max - 1
            ax2.fill_between(times, drawdown, 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True)
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Plot trade points
            for trade in results['trades']:
                if trade['pnl'] > 0:
                    color = 'g'
                    marker = '^'
                else:
                    color = 'r'
                    marker = 'v'
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                          color=color, marker=marker, s=100)
            
            # Plot daily returns
            returns = pd.Series(values).pct_change().fillna(0)
            # Ensure times and returns have the same length
            ax3.bar(times, returns, color='blue', alpha=0.5)
            ax3.set_title('Returns')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Return %')
            ax3.grid(True)
            ax3.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib is required for plotting")
            
    def save_results(self, results: Dict, save_dir: str):
        """
        Save backtest results to files
        
        Args:
            results: Dictionary with backtest results
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(save_dir / 'metrics.csv', index=False)
        
        # Save trades
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(save_dir / 'trades.csv', index=False)
        
        # Save portfolio values
        portfolio_df = pd.DataFrame(
            results['portfolio_values_with_time'],
            columns=['timestamp', 'portfolio_value']
        )
        portfolio_df.to_csv(save_dir / 'portfolio_values.csv', index=False)
        
        # Save plots
        self.plot_results(results, save_path=str(save_dir / 'results_plot.png'))
        
        logger.info(f"Results saved to {save_dir}")
    
    def _record_trade(self, action, price, timestamp):
        """Record a trade with proper datetime handling"""
        if isinstance(timestamp, (int, float)):
            timestamp = pd.Timestamp.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        trade = {
            'action': action,
            'price': price,
            'entry_time': timestamp,
            'exit_time': None,
            'profit': 0.0
        }
        self.trades.append(trade)