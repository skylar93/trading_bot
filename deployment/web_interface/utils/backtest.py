"""
Backtest utilities for the Trading Bot UI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from training.backtesting.risk_aware_backtester import RiskAwareBacktester
from training.backtesting.risk_manager import RiskManager, RiskConfig
from agents.strategies.agent_factory import create_agent

def setup_logging():
    """Configure logging with proper handlers and levels"""
    # Use a flag to check if logging has already been set up
    logger_name = "backtest_logger"
    logger = logging.getLogger(logger_name)
    
    # If logger already has handlers, assume it's configured
    if logger.handlers:
        return logger
        
    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Configure console handler for INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure file handler for DEBUG level
    file_handler = logging.FileHandler('backtest_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Set up logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger

# Initialize logger only once
logger = setup_logging()

class BacktestManager:
    """Manage backtest execution and results"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.logger = logger  # Use the singleton logger instance
        self.logger.debug("Initializing BacktestManager with settings: %s", settings)
        
        self.risk_config = RiskConfig(
            max_position_size=settings["max_position_size"] / 100.0,
            stop_loss_pct=settings["stop_loss"] / 100.0,
            max_drawdown_pct=0.15,
            daily_trade_limit=1000,
            var_confidence_level=0.95,
            portfolio_var_limit=0.02,
            max_correlation=0.7
        )
        self.logger.debug("Risk config initialized: %s", vars(self.risk_config))
        
        # Create agent using factory
        agent_name = settings.get("agent_name", "Dummy")
        agent_config = settings.get("agent_config", {})
        self.logger.debug("Creating agent: %s with config: %s", agent_name, agent_config)
        self.agent = create_agent(agent_name, config=agent_config)
        self.logger.info("Agent created: %s", agent_name)
        
    def load_market_data(self) -> Optional[pd.DataFrame]:
        """Load market data for backtesting"""
        try:
            self.logger.info("Loading market data for %s from %s to %s", 
                           self.settings["trading_pair"],
                           self.settings["start_date"],
                           self.settings["end_date"])
            
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
            
            self.logger.info("Successfully loaded %d data points", len(df))
            self.logger.debug("Data head: %s", df.head())
            return df
            
        except Exception as e:
            self.logger.error("Error loading market data: %s", str(e), exc_info=True)
            return None
    
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with the current agent"""
        try:
            self.logger.info("Starting backtest - Agent: %s, Data period: %s to %s",
                           self.settings.get("agent_name"),
                           data.index[0].strftime('%Y-%m-%d'),
                           data.index[-1].strftime('%Y-%m-%d'))
            
            # Initialize backtester with risk config
            risk_config_summary = {
                "max_position": f"{self.risk_config.max_position_size*100}%",
                "stop_loss": f"{self.risk_config.stop_loss_pct*100}%",
                "max_drawdown": f"{self.risk_config.max_drawdown_pct*100}%"
            }
            self.logger.info("Risk config: %s", risk_config_summary)
            
            backtester = RiskAwareBacktester(
                data=data,
                risk_config=self.risk_config,
                initial_capital=self.settings.get("initial_balance", 10000.0),
                trading_fee=self.settings.get("trading_fee", 0.001)
            )
            
            # Run backtest
            results = backtester.run(
                strategy=self.agent,
                window_size=20,
                verbose=True
            )
            
            # Process results
            portfolio_values = results.get("portfolio_values", [])
            if portfolio_values:
                self.logger.info("Backtest completed - Portfolio values: %d, Final value: $%.2f",
                               len(portfolio_values),
                               portfolio_values[-1])
                
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
            else:
                self.logger.warning("No portfolio values in results")
            
            # Update and log metrics
            results["metrics"] = results.get("metrics", {})
            results["metrics"].update(self._process_results(results))
            
            # Log key performance metrics
            metrics = results["metrics"]
            self.logger.info("Performance Metrics:")
            self.logger.info("- Total Return: %.2f%%", metrics.get("total_return", 0) * 100)
            self.logger.info("- Sharpe Ratio: %.3f", metrics.get("sharpe_ratio", 0))
            self.logger.info("- Max Drawdown: %.2f%%", metrics.get("max_drawdown", 0) * 100)
            self.logger.info("- Total Trades: %d", metrics.get("total_trades", 0))
            self.logger.info("- Win Rate: %.2f%%", metrics.get("win_rate", 0) * 100)
            
            return results
            
        except Exception as e:
            self.logger.error("Backtest failed: %s", str(e), exc_info=True)
            raise
    
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process backtest results and calculate additional metrics"""
        try:
            # Start with existing metrics
            metrics = results.get("metrics", {}).copy()
            trades = results.get("trades", [])
            portfolio_values = results.get("portfolio_values", [])
            
            if trades:
                # Only calculate these if not already present
                if "avg_trade" not in metrics:
                    total_trades = len(trades)
                    metrics["avg_trade"] = (
                        sum(t.get("pnl", 0) for t in trades) / total_trades 
                        if total_trades > 0 else 0
                    )
            
            if portfolio_values and "sharpe_ratio" not in metrics:
                # Calculate portfolio metrics only if not present
                returns = pd.Series([float(v) for v in portfolio_values]).pct_change().dropna()
                if len(returns) > 0:
                    metrics["sharpe_ratio"] = (
                        np.sqrt(252) * (returns.mean() / returns.std()) 
                        if returns.std() != 0 else 0
                    )
            
            self.logger.info("Additional metrics processed: %s", 
                           {k: v for k, v in metrics.items() 
                            if k not in results.get("metrics", {})})
            
            return metrics
            
        except Exception as e:
            self.logger.error("Error processing additional metrics: %s", str(e), exc_info=True)
            return results.get("metrics", {}).copy()  # Return original metrics on error 