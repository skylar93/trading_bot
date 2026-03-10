"""
Risk Manager Factory Module

This module provides a factory function for creating different types of risk managers.

Features:
- Create risk managers based on type (RL, backtesting, etc.)
- Load risk config from dictionaries or config files
- Provide sensible defaults for different risk manager types

Implementation Notes:
- Factory pattern to abstract risk manager implementation details
- Centralizes risk manager creation logic
- Handles configuration mapping and validation
"""

import logging
from typing import Dict, Any, Union, Optional, Type

from risk_management.risk_manager_base import RiskManagerBase, RiskConfigBase
from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
from risk_management.backtesting_risk_manager import BacktestingRiskManager, BacktestingRiskConfig


def create_risk_manager(
    risk_type: str,
    config: Optional[Dict[str, Any]] = None
) -> RiskManagerBase:
    """
    Factory function to create risk managers of different types.
    
    Args:
        risk_type: Type of risk manager to create ("rl", "backtesting")
        config: Optional configuration dictionary for the risk manager
        
    Returns:
        RiskManagerBase: An instance of the appropriate risk manager
        
    Raises:
        ValueError: If risk_type is not recognized
    """
    logger = logging.getLogger("RiskManagerFactory")
    config = config or {}
    
    if risk_type.lower() == "rl":
        risk_config = RLRiskConfig(**config)
        logger.info(f"Creating RL risk manager with config: {risk_config}")
        return RLRiskManager(risk_config)
    elif risk_type.lower() in ["backtest", "backtesting"]:
        risk_config = BacktestingRiskConfig(**config)
        logger.info(f"Creating backtesting risk manager with config: {risk_config}")
        return BacktestingRiskManager(risk_config)
    else:
        raise ValueError(f"Unknown risk manager type: {risk_type}")


def create_risk_config(
    risk_type: str,
    config: Optional[Dict[str, Any]] = None
) -> RiskConfigBase:
    """
    Create a risk configuration object for the specified risk manager type.
    
    Args:
        risk_type: Type of risk manager config to create ("rl", "backtesting")
        config: Optional configuration dictionary for the risk config
        
    Returns:
        RiskConfigBase: An instance of the appropriate risk config
        
    Raises:
        ValueError: If risk_type is not recognized
    """
    config = config or {}
    
    if risk_type.lower() == "rl":
        return RLRiskConfig(**config)
    elif risk_type.lower() in ["backtest", "backtesting"]:
        return BacktestingRiskConfig(**config)
    else:
        raise ValueError(f"Unknown risk manager type: {risk_type}") 