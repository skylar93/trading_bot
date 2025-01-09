"""
Backtest utilities for the Trading Bot UI
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from training.utils.risk_backtest import RiskAwareBacktester
from risk.risk_manager import RiskManager, RiskConfig
import random

logger = logging.getLogger(__name__)

class DummyAgent:
    """Dummy agent for testing that makes small random trades"""
    
    def __init__(self):
        self.step_count = -1  # Start at -1 so first increment gives 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_action(self, state: Dict[str, Any]) -> float:
        """Generate small random actions for testing"""
        self.step_count += 1
        
        # First action should be non-zero but small for consistency test
        if self.step_count == 0:
            return 0.5
            
        # Trade every 5 steps with small magnitude
        if self.step_count % 5 == 0:
            action = 0.5 if (self.step_count // 5) % 2 == 0 else -0.5
            self.logger.info(f"DummyAgent taking action: {action}")
            return action
            
        return 0.0

class BacktestManager:
    """Manage backtest execution and results"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing BacktestManager with settings: %s", settings)
        
        self.risk_config = RiskConfig(
            max_position_size=settings["max_position_size"] / 100.0,  # Convert from percentage
            stop_loss_pct=settings["stop_loss"] / 100.0,  # Convert from percentage
            max_drawdown_pct=0.15,
            daily_trade_limit=1000,
            var_confidence_level=0.95,
            portfolio_var_limit=0.02,
            max_correlation=0.7
        )
        self.logger.info("Risk config initialized: %s", vars(self.risk_config))
        
        self.agent = DummyAgent()
        self.logger.info("DummyAgent initialized")
        
    def load_market_data(self) -> Optional[pd.DataFrame]:
        """Load market data for backtesting
        
        Returns:
            DataFrame with OHLCV data or None if loading fails
        """
        try:
            import ccxt
            
            # Initialize exchange
            exchange = ccxt.binance()
            
            # Get timeframe in milliseconds
            timeframe_ms = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000
            }
            
            # Calculate timestamps
            start_timestamp = int(pd.Timestamp(self.settings["start_date"]).timestamp() * 1000)
            end_timestamp = int(pd.Timestamp(self.settings["end_date"]).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                self.logger.info(f"Fetching data from {pd.Timestamp(current_timestamp, unit='ms')}")
                chunk = exchange.fetch_ohlcv(
                    symbol=self.settings["trading_pair"],
                    timeframe="1h",  # Use 1h timeframe for now
                    since=current_timestamp,
                    limit=1000  # Maximum limit for most exchanges
                )
                
                if not chunk:
                    break
                    
                ohlcv.extend(chunk)
                current_timestamp = chunk[-1][0] + timeframe_ms["1h"]
            
            if not ohlcv:
                self.logger.error("No data fetched from exchange")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "$open", "$high", "$low", "$close", "$volume"]
            )
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            self.logger.info(f"Successfully loaded {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}", exc_info=True)
            return None
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with the current agent"""
        try:
            self.logger.info("Starting backtest with data shape: %s", data.shape)
            
            # Initialize backtester with risk config
            backtester = RiskAwareBacktester(
                data=data,
                risk_config=self.risk_config,
                initial_balance=self.settings.get("initial_balance", 10000.0),
                trading_fee=self.settings.get("trading_fee", 0.001)
            )
            
            # Run backtest
            results = backtester.run(
                agent=self.agent,
                window_size=20,
                verbose=True
            )
            
            # Process results to ensure all required metrics
            if "metrics" not in results:
                results["metrics"] = {}
            
            # Update portfolio values with timestamps
            portfolio_values = results.get("portfolio_values", [])
            if portfolio_values:
                # Create timestamps for each portfolio value
                timestamps = pd.date_range(
                    start=data.index[0],
                    end=data.index[-1],
                    periods=len(portfolio_values)
                )
                results["portfolio_history"] = [
                    {"timestamp": ts, "value": val}
                    for ts, val in zip(timestamps, portfolio_values)
                ]
            
            # Update metrics
            results["metrics"].update(self._process_results(results))
            
            # Log final results
            self.logger.info("Backtest completed")
            self.logger.info("Final portfolio value: %.2f", results.get("portfolio_values", [])[-1])
            self.logger.info("Total trades: %d", len(results.get("trades", [])))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "portfolio_values": [],
                "trades": [],
                "metrics": self._process_results({"trades": [], "portfolio_values": []})
            }
    
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process backtest results and calculate metrics"""
        metrics = {}
        
        if "trades" in results and results["trades"]:
            trades = results["trades"]
            pnls = [t.get("pnl", 0.0) for t in trades]
            
            # Calculate basic metrics
            total_pnl = sum(pnls)
            profitable_trades = sum(1 for pnl in pnls if pnl > 0)
            total_trades = len(trades)
            
            # Calculate profit factor
            winning_pnls = [pnl for pnl in pnls if pnl > 0]
            losing_pnls = [abs(pnl) for pnl in pnls if pnl < 0]
            total_profits = sum(winning_pnls) if winning_pnls else 0
            total_losses = sum(losing_pnls) if losing_pnls else 0
            profit_factor = total_profits / total_losses if total_losses > 0 else float('inf') if total_profits > 0 else 0.0
            
            # Calculate average trade
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0
            
            metrics.update({
                "total_return": total_pnl / self.settings.get("initial_balance", 10000.0),
                "sharpe_ratio": self._calculate_sharpe_ratio(pnls) if pnls else 0.0,
                "max_drawdown": self._calculate_max_drawdown(results.get("portfolio_values", [])),
                "win_rate": (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "total_pnl": total_pnl,
                "profit_factor": profit_factor,
                "avg_trade": avg_trade
            })
        
        return metrics
    
    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        """Calculate Sharpe ratio from PnL values"""
        if not pnls or len(pnls) < 2:
            return 0.0
            
        returns = pd.Series(pnls)
        std = returns.std()
        if std == 0:
            return 0.0
            
        return (returns.mean() / std) * np.sqrt(252)  # Annualized
        
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values:
            return 0.0
            
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd 