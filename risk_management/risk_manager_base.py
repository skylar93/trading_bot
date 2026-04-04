"""
Risk Manager Base Abstract Class

This module provides a base abstract class for risk management functionality
that can be extended for different use cases (RL environments, backtesting, etc.)

Features:
- Abstract methods for risk management core functionality
- Common interfaces for different risk management implementations
- Type definitions and documentation for risk management components

Implementation Notes:
- This is an abstract base class and should not be instantiated directly
- Concrete implementations should be created for specific use cases
- Designed to unify risk management across different trading contexts
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Set


@dataclass
class RiskConfigBase:
    """
    Base configuration for risk management.
    
    Features:
    - Core risk parameters common to all risk management contexts
    - Foundation for specialized risk configurations
    
    Implementation Notes:
    - All percentages are expressed as decimals (0.01 = 1%)
    - Default values are set to conservative levels
    - Should be extended by specific implementations
    """
    # Stop loss settings
    use_stop_loss: bool = True
    stop_loss_threshold: float = 0.1  # 10% loss triggers stop loss
    
    # Trailing stop settings
    use_trailing_stop: bool = False  
    trailing_stop_buffer: float = 0.05  # 5% drop from highest point
    
    # Drawdown protection
    max_drawdown_pct: float = 0.15  # 15% maximum drawdown
    use_forced_liquidation: bool = False  # Force liquidation on max drawdown
    
    # VaR settings
    use_var: bool = False
    var_confidence_level: float = 0.95
    rolling_var_window: int = 100


class RiskManagerBase(ABC):
    """
    Abstract base class for risk management functionality.
    
    Provides common interface for risk management operations across
    different contexts (RL environment, backtesting, live trading, etc.)
    """
    
    def __init__(self, config: RiskConfigBase):
        """
        Initialize the risk manager with the given configuration.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def reset(self):
        """Reset all risk manager state."""
        pass
    
    @abstractmethod
    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """
        Check if maximum drawdown has been exceeded.
        
        Args:
            peak_value: Historical peak portfolio value
            current_value: Current portfolio value
            
        Returns:
            bool: True if max drawdown exceeded, False otherwise
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(
        self, entry_price: float, position_size: float, is_long: bool = True
    ) -> float:
        """
        Calculate the stop loss price based on configuration.
        
        Args:
            entry_price: The entry price of the position
            position_size: The size of the position (positive for long, negative for short)
            is_long: Whether the position is long (True) or short (False)
            
        Returns:
            float: The stop loss price
        """
        pass
    
    @abstractmethod
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss has been triggered for the given symbol.
        
        Args:
            symbol: The symbol/asset to check
            current_price: The current price of the asset
            
        Returns:
            bool: True if stop loss triggered, False otherwise
        """
        pass
    
    @abstractmethod
    def update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """
        Update the trailing stop level for a symbol based on current price.
        
        Args:
            symbol: The symbol/asset to update
            current_price: The current price of the asset
        """
        pass
    
    @abstractmethod
    def calculate_var(self, returns: np.ndarray) -> float:
        """
        Calculate Value at Risk (VaR) based on historical returns.
        
        Args:
            returns: Array of historical returns
            
        Returns:
            float: Value at Risk at the configured confidence level
        """
        pass
    
    @abstractmethod
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of risk metrics
        """
        pass