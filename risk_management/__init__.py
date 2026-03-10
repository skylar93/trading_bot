"""
Risk Management Package

This package provides risk management functionality for trading environments,
including backtesting and RL environments.
"""

from risk_management.risk_manager_base import RiskManagerBase, RiskConfigBase
from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
from risk_management.backtesting_risk_manager import BacktestingRiskManager, BacktestingRiskConfig, StopLossConfig
from risk_management.factory import create_risk_manager, create_risk_config 