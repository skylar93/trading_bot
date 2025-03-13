"""
Backtest presenter for web interface.

This module separates backtesting logic from UI presentation in the Streamlit interface.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

from deployment.web_interface.utils.backtest import BacktestManager
from training.backtesting.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)

class BacktestPresenter:
    """
    Presenter for backtest results that handles:
    1. Loading and preprocessing market data
    2. Managing different backtest scenarios
    3. Running backtests with appropriate parameters
    4. Processing and storing backtest results
    
    This class is responsible for:
    - Loading historical market data
    - Applying scenario modifications (flash crash, low liquidity, etc.)
    - Coordinating actual backtesting through BacktestManager
    - Processing and storing results for UI presentation
    
    Features:
    - Separation of backtest logic from UI presentation
    - Support for multiple scenario types
    - Result data preparation for visualization
    
    Implementation Notes:
    - Delegates actual backtesting to BacktestManager
    - Scenario modification handled by ScenarioManager
    - Maintains state of current backtest results
    
    Recent Changes:
    - Created as part of UI/logic separation refactoring
    """
    
    def __init__(self):
        """Initialize the backtest presenter"""
        # Create default settings for BacktestManager initialization
        default_settings = {
            "agent_name": "Dummy",
            "agent_config": {},
            "max_position_size": 50,  # 50%
            "stop_loss": 5,          # 5%
            "initial_balance": 10000.0,
        }
        self.backtest_manager = BacktestManager(settings=default_settings)
        self.scenario_manager = ScenarioManager()
        self.results = None
        self.scenario_metrics = {}
        self.scenario_type = None
        self.data = None
        self.modified_data = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("BacktestPresenter initialized with default settings")
    
    def load_market_data(self, symbol: str, timeframe: str, start_date: datetime, 
                         end_date: datetime) -> bool:
        """
        Load historical market data for backtesting
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1h")
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            bool: True if data loading was successful
        """
        self.logger.info(f"Loading market data for {symbol} {timeframe} from {start_date} to {end_date}")
        
        try:
            # Update BacktestManager settings before loading data
            self.backtest_manager.settings.update({
                "trading_pair": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            })
            
            # Use BacktestManager to load data
            self.data = self.backtest_manager.load_market_data()
            
            if self.data is None or self.data.empty:
                self.logger.error("Failed to load market data")
                return False
                
            self.logger.info(f"Successfully loaded {len(self.data)} data points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}", exc_info=True)
            return False
    
    def apply_scenario(self, scenario_type: str, scenario_params: Dict[str, Any]) -> bool:
        """
        Apply a scenario modification to the loaded market data
        
        Args:
            scenario_type: Type of scenario (e.g., "flash_crash", "low_liquidity")
            scenario_params: Parameters for the scenario
            
        Returns:
            bool: True if scenario application was successful
        """
        if self.data is None or self.data.empty:
            self.logger.error("Cannot apply scenario: No market data loaded")
            return False
            
        self.logger.info(f"Applying {scenario_type} scenario with params: {scenario_params}")
        
        try:
            # Use ScenarioManager to modify data
            self.modified_data = self.scenario_manager.apply_scenario(
                data=self.data,
                scenario_type=scenario_type,
                params=scenario_params
            )
            
            if self.modified_data is None or self.modified_data.empty:
                self.logger.error("Failed to apply scenario")
                return False
                
            self.scenario_type = scenario_type
            self.logger.info(f"Successfully applied {scenario_type} scenario")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying scenario: {str(e)}", exc_info=True)
            return False
    
    def run_backtest(self, agent_type: str, risk_params: Dict[str, Any], 
                    initial_balance: float, model_path: Optional[str] = None) -> bool:
        """
        Run a backtest with the current market data and scenario
        
        Args:
            agent_type: Type of trading agent to use
            risk_params: Risk management parameters
            initial_balance: Initial balance for the backtest
            model_path: Optional path to a trained model checkpoint
            
        Returns:
            bool: True if backtest was successful
        """
        # Determine which data to use (original or modified)
        data_to_use = self.modified_data if self.modified_data is not None else self.data
        
        if data_to_use is None or data_to_use.empty:
            self.logger.error("Cannot run backtest: No market data available")
            return False
            
        self.logger.info(f"Running backtest with agent: {agent_type}, initial balance: {initial_balance}")
        self.logger.info(f"Risk parameters: {risk_params}")
        if model_path:
            self.logger.info(f"Using trained model from: {model_path}")
        
        try:
            # Update the BacktestManager settings with the new parameters
            self.backtest_manager.settings.update({
                "agent_name": agent_type,
                "initial_balance": initial_balance,
                "risk_params": risk_params,
                "model_path": model_path  # Add model path to settings
            })
            
            # Directly update the risk_config object with new parameters
            if hasattr(self.backtest_manager, 'risk_config'):
                if 'min_trade_size' in risk_params:
                    self.backtest_manager.risk_config.min_trade_size = risk_params['min_trade_size']
                if 'max_position_size' in risk_params:
                    self.backtest_manager.risk_config.max_position_size = risk_params['max_position_size']
                if 'stop_loss' in risk_params:
                    self.backtest_manager.risk_config.stop_loss_pct = risk_params['stop_loss']
                
                self.logger.info(f"Updated risk config: {vars(self.backtest_manager.risk_config)}")
            
            # Use BacktestManager to run the backtest
            self.results = self.backtest_manager.run_backtest(
                data=data_to_use
            )
            
            if self.results is None:
                self.logger.error("Failed to run backtest")
                return False
                
            # Calculate scenario-specific metrics if a scenario was applied
            if self.scenario_type:
                self.scenario_metrics = self._calculate_scenario_metrics()
                
            self.logger.info("Successfully ran backtest")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}", exc_info=True)
            return False
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all backtest results for UI presentation
        
        Returns:
            Dict containing all results data needed for UI
        """
        if self.results is None:
            return {
                "success": False,
                "message": "No backtest results available"
            }
            
        # Extract portfolio values and timestamps if available
        portfolio_values = []
        timestamps = []
        
        if "portfolio_data" in self.results:
            portfolio_data = self.results.get("portfolio_data")
            if portfolio_data is not None and not isinstance(portfolio_data, dict) and not portfolio_data.empty:
                # If we have a DataFrame with portfolio history
                if "portfolio_value" in portfolio_data.columns and "timestamp" in portfolio_data.columns:
                    portfolio_values = portfolio_data["portfolio_value"].tolist()
                    timestamps = portfolio_data["timestamp"].tolist()
        elif "portfolio_values" in self.results and "timestamps" in self.results:
            # Direct access if already available as lists
            portfolio_values = self.results.get("portfolio_values", [])
            timestamps = self.results.get("timestamps", [])
            
        return {
            "success": True,
            "portfolio_data": self.results.get("portfolio_data"),
            "trade_list": self.results.get("trade_list"), 
            "trades": self.results.get("trades", []),  # For compatibility with original code
            "portfolio_values": portfolio_values,
            "timestamps": timestamps,
            "metrics": self.results.get("metrics"),
            "scenario_type": self.scenario_type,
            "scenario_metrics": self.scenario_metrics
        }
    
    def _calculate_scenario_metrics(self) -> Dict[str, float]:
        """Calculate scenario-specific metrics based on the current scenario type"""
        if not self.scenario_type or not self.results:
            return {}
            
        metrics = {}
        
        if self.scenario_type == "flash_crash":
            # Calculate flash crash specific metrics
            portfolio_data = self.results.get("portfolio_data")
            trade_list = self.results.get("trade_list")
            
            if portfolio_data is not None:
                # Find the crash point (largest drawdown)
                portfolio_values = portfolio_data["portfolio_value"].values
                max_drawdown_idx = np.argmin(portfolio_values / np.maximum.accumulate(portfolio_values))
                
                # Calculate recovery metrics
                pre_crash_value = np.max(portfolio_values[:max_drawdown_idx]) if max_drawdown_idx > 0 else portfolio_values[0]
                crash_value = portfolio_values[max_drawdown_idx]
                post_crash_max = np.max(portfolio_values[max_drawdown_idx:])
                
                metrics["drawdown_depth"] = (1 - crash_value / pre_crash_value) * 100
                metrics["recovery_percentage"] = (post_crash_max / crash_value - 1) * 100
                metrics["recovery_speed"] = len(portfolio_values) - max_drawdown_idx
                
                # Calculate trade efficacy during crash
                if trade_list is not None:
                    crash_trades = [t for t in trade_list if t["timestamp"] >= portfolio_data.index[max_drawdown_idx]]
                    metrics["crash_trade_efficacy"] = np.mean([t["pnl"] for t in crash_trades]) if crash_trades else 0
            
        elif self.scenario_type == "low_liquidity":
            # Calculate low liquidity specific metrics
            trade_list = self.results.get("trade_list")
            
            if trade_list is not None:
                # Extract metrics from trade list
                fill_rates = [t.get("fill_rate", 100) for t in trade_list]
                trade_costs = [t.get("trade_cost", 0) for t in trade_list]
                spreads = [t.get("spread", 0) for t in trade_list]
                execution_delays = [t.get("execution_delay", 0) for t in trade_list]
                
                metrics["fill_rate"] = np.mean(fill_rates) if fill_rates else 100
                metrics["avg_trade_cost"] = np.mean(trade_costs) if trade_costs else 0
                metrics["avg_spread"] = np.mean(spreads) if spreads else 0
                metrics["execution_delay"] = np.mean(execution_delays) if execution_delays else 0
        
        return metrics 