"""
Backward Compatibility Module for Risk Management in Backtesting

This module maintains backward compatibility with the old risk manager interface
while using the new abstracted risk management system.

Recent Changes:
- Refactored to use the new risk management system
- Maintains backward compatibility with existing code
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, List, Tuple, Set

from risk_management import create_risk_manager, create_risk_config
from risk_management.backtesting_risk_manager import StopLossConfig


@dataclass
class RiskConfig:
    """
    Merged configuration for risk management.
    For backward compatibility with existing code.
    """
    # Basic risk
    max_position_size: float = 0.2       # fraction of portfolio
    stop_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    daily_trade_limit: int = 10
    min_trade_size: float = 0.01        # fraction of portfolio
    max_leverage: float = 1.0
    
    # Advanced portfolio risk
    volatility_lookback: int = 20
    risk_free_rate: float = 0.02
    var_confidence_level: float = 0.95
    correlation_window: int = 30
    max_correlation: float = 0.7
    portfolio_var_limit: float = 0.02   # fraction of portfolio value
    
    # New stop loss parameters
    trailing_stop_pct: float = 0.05    # Default trailing stop percentage
    enable_stop_loss: bool = True      # Whether to enforce stop losses
    enable_trailing_stop: bool = False  # Whether to use trailing stops
    
    # New forced liquidation parameters
    enable_forced_liquidation: bool = False  # Whether to force liquidation on max drawdown
    forced_liquidation_drawdown: float = 0.15  # Drawdown threshold for forced liquidation
    
    # New time-based constraints
    close_positions_on_friday: bool = False  # Whether to close positions at Friday end of day
    max_holding_period_days: int = 0   # Maximum holding period (0 = no limit)
    
    # New execution constraints
    enable_partial_fills: bool = False  # Whether to simulate partial fills
    max_partial_fill_pct: float = 0.8   # Maximum percentage of order to fill (if partial fills enabled)
    min_partial_fill_pct: float = 0.5   # Minimum percentage of order to fill (if partial fills enabled)
    slippage_std: float = 0.001         # Standard deviation for slippage simulation


class RiskManager:
    """
    Unified Risk Manager for backtesting environments.
    Backward compatibility wrapper for new BacktestingRiskManager.
    """
    def __init__(self, config: RiskConfig):
        """
        Initialize the RiskManager with the given configuration.
        
        Args:
            config (RiskConfig): Risk management configuration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Convert RiskConfig to dictionary for factory
        config_dict = {k: getattr(config, k) for k in dir(config) 
                      if not k.startswith('_') and not callable(getattr(config, k))}
        
        # For legacy compatibility, map some config parameters
        if "stop_loss_pct" in config_dict:
            config_dict["stop_loss_threshold"] = config_dict["stop_loss_pct"]
        
        if "enable_stop_loss" in config_dict:
            config_dict["use_stop_loss"] = config_dict["enable_stop_loss"]
            
        if "enable_trailing_stop" in config_dict:
            config_dict["use_trailing_stop"] = config_dict["enable_trailing_stop"]
        
        if "trailing_stop_pct" in config_dict:
            config_dict["trailing_stop_buffer"] = config_dict["trailing_stop_pct"]
        
        if "enable_forced_liquidation" in config_dict:
            config_dict["use_forced_liquidation"] = config_dict["enable_forced_liquidation"]
            
        # Create the actual risk manager using the factory
        self._risk_manager = create_risk_manager("backtesting", config_dict)
        
        # For backward compatibility
        self.config = config
    
    def __getattr__(self, name):
        """
        Delegate method calls to the underlying risk manager.
        
        Args:
            name: Attribute name
            
        Returns:
            The requested attribute from the underlying risk manager
        """
        return getattr(self._risk_manager, name)
    
    # Keep StopLossConfig class accessible for backward compatibility 
    StopLossConfig = StopLossConfig 
    
    def _apply_partial_fill(self, trade_size):
        """
        Delegate _apply_partial_fill to the underlying risk manager.
        
        Args:
            trade_size (float): The original requested trade size
            
        Returns:
            float: The adjusted trade size after applying partial fill simulation
        """
        return self._risk_manager._apply_partial_fill(trade_size) 