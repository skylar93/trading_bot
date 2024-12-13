import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TradingVisualizer:
    """Visualization tools for trading results"""
    
    def __init__(self, figsize: tuple = (15, 10)):
        self.figsize = figsize
        sns.set_theme()  # Use seaborn's default theme
    
    def plot_portfolio_performance(self, 
                                 portfolio_values: np.ndarray,
                                 returns: np.ndarray,
                                 trades: List[Dict],
                                 metrics: Dict,
                                 save_path: Optional[str] = None):
        """Create comprehensive performance visualization"""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        
        # Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_portfolio_value(ax1, portfolio_values)
        self._add_trade_markers(ax1, portfolio_values, trades)
        
        # Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_returns_dist(ax2, returns)
        
        # Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax3, portfolio_values)
        
        # Rolling Metrics
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_rolling_metrics(ax4, returns)
        
        # Trade Analysis
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_trade_analysis(ax5, trades)
        
        # Add performance metrics as text
        self._add_metrics_text(fig, metrics)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance visualization to {save_path}")
        plt.close()
    
    def _plot_portfolio_value(self, ax, portfolio_values: np.ndarray):
        """Plot portfolio value over time"""
        ax.plot(portfolio_values, linewidth=2)
        ax.set_title('Portfolio Value Over Time', fontsize=12)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
    
    def _add_trade_markers(self, ax, portfolio_values: np.ndarray, trades: List[Dict]):
        """Add trade entry/exit markers to portfolio value plot"""
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                color = 'g'
            else:
                color = 'r'
            ax.scatter(trade.get('entry_time', 0), portfolio_values[trade.get('entry_time', 0)], 
                      marker='^', color=color, alpha=0.6)
            ax.scatter(trade.get('exit_time', 0), portfolio_values[trade.get('exit_time', 0)], 
                      marker='v', color=color, alpha=0.6)
    
    def _plot_returns_dist(self, ax, returns: np.ndarray):
        """Plot returns distribution"""
        if len(returns) > 0:
            sns.histplot(returns, ax=ax, bins=50)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Returns Distribution', fontsize=12)
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
    
    def _plot_drawdown(self, ax, portfolio_values: np.ndarray):
        """Plot drawdown over time"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        ax.fill_between(np.arange(len(drawdown)), drawdown, 0, 
                       color='r', alpha=0.3)
        ax.set_title('Drawdown', fontsize=12)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Drawdown %')
        ax.grid(True)
    
    def _plot_rolling_metrics(self, ax, returns: np.ndarray):
        """Plot rolling Sharpe ratio and volatility"""
        if len(returns) > 30:  # Need at least 30 points for rolling window
            df = pd.Series(returns)
            rolling_sharpe = df.rolling(window=30).mean() / df.rolling(window=30).std() * np.sqrt(252)
            rolling_vol = df.rolling(window=30).std() * np.sqrt(252)
            
            ax.plot(rolling_sharpe, label='Rolling Sharpe', color='b')
            ax2 = ax.twinx()
            ax2.plot(rolling_vol, label='Rolling Vol', color='r', alpha=0.5)
            
            ax.set_title('Rolling Metrics (30-day)', fontsize=12)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Sharpe Ratio')
            ax2.set_ylabel('Volatility')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_trade_analysis(self, ax, trades: List[Dict]):
        """Plot trade PnL distribution"""
        if trades:
            pnls = [t.get('pnl', 0) for t in trades]
            if len(pnls) > 0:
                sns.histplot(pnls, ax=ax, bins=30)
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Trade PnL Distribution', fontsize=12)
        ax.set_xlabel('PnL')
        ax.set_ylabel('Frequency')
    
    def _add_metrics_text(self, fig, metrics: Dict):
        """Add performance metrics as text"""
        text = (
            f"Total Return: {metrics.get('total_return', 0):.1f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
            f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
            f"Profit/Loss Ratio: {metrics.get('profit_loss_ratio', 0):.2f}\n"
            f"Total Trades: {metrics.get('total_trades', 0)}"
        )
        fig.text(0.02, 0.02, text, fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))