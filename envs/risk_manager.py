"""
Risk Manager for Multi-Agent Trading Environment - Backward Compatibility Module

This module maintains backward compatibility with the old risk manager interface
while using the new abstracted risk management system.

Recent Changes:
- Refactored to use the new risk management system
- Maintains backward compatibility with existing code
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from collections import deque
from scipy.stats import norm

from risk_management import create_risk_manager, RLRiskConfig


@dataclass
class RiskConfig:
    """
    Configuration for risk management in the trading environment.
    For backward compatibility with existing code.
    """
    # Stop loss settings
    use_stop_loss: bool = True
    stop_loss_threshold: float = 0.1  # 10% loss triggers stop loss
    
    # Trailing stop settings
    use_trailing_stop: bool = False  
    trailing_stop_buffer: float = 0.05  # 5% drop from highest point
    
    # VaR settings
    use_var: bool = False
    var_confidence_level: float = 0.95
    rolling_var_window: int = 100
    action_on_var_exceed: str = "reduce_position"  # "reduce_position" or "close_position"
    
    # Drawdown protection
    max_drawdown_pct: float = 0.15  # 15% maximum drawdown
    use_forced_liquidation: bool = False  # Force liquidation on max drawdown
    
    # Application frequency settings
    check_frequency: int = 1  # Check every n steps
    
    # Correlation settings
    use_correlation: bool = False
    correlation_window: int = 50  # Window for correlation calculation
    correlation_threshold: float = 0.7  # Threshold to consider high correlation
    correlation_risk_reduction: float = 0.5  # Position size multiplier when correlation exceeds threshold
    
    # Portfolio-level stop loss
    use_portfolio_stop_loss: bool = False
    portfolio_stop_loss_threshold: float = 0.15  # 15% portfolio loss triggers stop loss
    
    # Portfolio-level trailing stop
    use_portfolio_trailing_stop: bool = False
    portfolio_trailing_stop_buffer: float = 0.08  # 8% drop from portfolio high water mark
    
    # Multi-asset VaR settings
    use_portfolio_var: bool = False
    portfolio_var_threshold: float = 0.02  # Maximum acceptable portfolio VaR (2%)
    use_parametric_var: bool = True  # Use parametric (True) or historical (False) VaR calculation


class RiskManager:
    """
    Risk manager for RL trading environment.
    Backward compatibility wrapper for new RLRiskManager.
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize the risk manager with the given configuration.
        
        Args:
            config: Risk management configuration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Convert RiskConfig to dictionary for factory
        config_dict = {k: getattr(config, k) for k in dir(config) 
                      if not k.startswith('_') and not callable(getattr(config, k))}
        
        # Create the actual risk manager using the factory
        self._risk_manager = create_risk_manager("rl", config_dict)
        
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
    
    # Add any methods needed for backward compatibility that aren't in the new system 