"""
Risk Manager for Multi-Agent Trading Environment

This module provides risk management functionality for the RL trading environment,
including stop-loss, trailing stop, VaR/CVaR, and other risk controls.

Features:
- Position-level risk management (stop-loss, trailing stop)
- Portfolio-level risk management (VaR, CVaR)
- Drawdown monitoring and forced liquidation
- Risk metrics tracking and reporting

Implementation Notes:
- Designed specifically for integration with MultiAgentTradingEnv
- Supports both per-agent and portfolio-wide risk constraints
- Provides signals for position adjustment or liquidation based on risk thresholds
- Maintains historical risk metrics for analysis and visualization

Recent Changes:
- Initial implementation with stop-loss and trailing stop functionality
- Added VaR/CVaR calculation for portfolio risk management
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from collections import deque


@dataclass
class RiskConfig:
    """
    Configuration for risk management in the trading environment.
    
    Features:
    - Stop loss and trailing stop settings
    - VaR/CVaR configuration
    - Drawdown limits and liquidation triggers
    
    Implementation Notes:
    - All percentages are expressed as decimals (0.01 = 1%)
    - Default values are set to conservative levels
    - Can be loaded from environment configuration
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


class RiskManager:
    """
    Risk manager for RL trading environment.
    
    Provides risk management functionality including stop-loss, trailing stop,
    VaR/CVaR calculation and position management based on risk thresholds.
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize the risk manager with the given configuration.
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Portfolio tracking
        self.peak_values = {}  # Dict[agent_id, peak_value]
        self.current_values = {}  # Dict[agent_id, current_value]
        self.position_highest_values = {}  # Dict[agent_id, Dict[asset, highest_value]]
        self.liquidation_triggered = {}  # Dict[agent_id, bool]
        
        # VaR tracking
        self.returns_history = {}  # Dict[agent_id, deque]
        
        # Metrics for logging
        self.stop_loss_events = 0
        self.trailing_stop_events = 0
        self.var_exceed_events = 0
        self.forced_liquidation_events = 0
    
    def reset(self):
        """Reset all risk manager state."""
        self.peak_values = {}
        self.current_values = {}
        self.position_highest_values = {}
        self.liquidation_triggered = {}
        self.returns_history = {}
        self.stop_loss_events = 0
        self.trailing_stop_events = 0
        self.var_exceed_events = 0
        self.forced_liquidation_events = 0
    
    def update_portfolio_values(self, portfolio_values: Dict[str, float]):
        """
        Update portfolio values for each agent and track peak values.
        
        Args:
            portfolio_values: Dictionary mapping agent_id to portfolio value
        """
        for agent_id, value in portfolio_values.items():
            self.current_values[agent_id] = value
            
            # Initialize or update peak value
            if agent_id not in self.peak_values or value > self.peak_values[agent_id]:
                self.peak_values[agent_id] = value
                
            # Initialize liquidation flag if not exists
            if agent_id not in self.liquidation_triggered:
                self.liquidation_triggered[agent_id] = False
    
    def record_returns(self, returns: Dict[str, float]):
        """
        Record returns for VaR calculation.
        
        Args:
            returns: Dictionary mapping agent_id to return value
        """
        if not self.config.use_var:
            return
            
        for agent_id, ret in returns.items():
            if agent_id not in self.returns_history:
                self.returns_history[agent_id] = deque(maxlen=self.config.rolling_var_window)
            self.returns_history[agent_id].append(ret)
    
    def check_stop_loss(self, agent_id: str, position_size: float, 
                       entry_price: float, current_price: float) -> bool:
        """
        Check if stop loss has been triggered for a position.
        
        Args:
            agent_id: Identifier for the agent
            position_size: Size of the position (positive for long, negative for short)
            entry_price: Entry price of the position
            current_price: Current market price
            
        Returns:
            bool: True if stop loss triggered, False otherwise
        """
        if not self.config.use_stop_loss:
            return False
            
        # Calculate percentage loss
        if position_size > 0:  # Long position
            pct_change = (current_price - entry_price) / entry_price
            is_loss = pct_change < 0
            loss_exceeded = abs(pct_change) > self.config.stop_loss_threshold
        else:  # Short position
            pct_change = (entry_price - current_price) / entry_price
            is_loss = pct_change < 0
            loss_exceeded = abs(pct_change) > self.config.stop_loss_threshold
        
        if is_loss and loss_exceeded:
            self.logger.warning(
                f"Stop loss triggered for {agent_id}: {pct_change:.2%} exceeds threshold "
                f"{self.config.stop_loss_threshold:.2%}"
            )
            self.stop_loss_events += 1
            return True
            
        return False
    
    def check_trailing_stop(self, agent_id: str, asset: str, 
                           position_size: float, current_price: float) -> bool:
        """
        Check if trailing stop has been triggered.
        
        Args:
            agent_id: Identifier for the agent
            asset: Asset symbol
            position_size: Size of the position (positive for long, negative for short)
            current_price: Current market price
            
        Returns:
            bool: True if trailing stop triggered, False otherwise
        """
        if not self.config.use_trailing_stop:
            return False
            
        # Initialize tracking for this position if needed
        position_key = f"{agent_id}_{asset}"
        if position_key not in self.position_highest_values:
            self.position_highest_values[position_key] = current_price
            return False
            
        highest_price = self.position_highest_values[position_key]
        
        # Update highest price if current price is higher (for long) or lower (for short)
        if position_size > 0 and current_price > highest_price:
            self.position_highest_values[position_key] = current_price
            return False
        elif position_size < 0 and current_price < highest_price:
            self.position_highest_values[position_key] = current_price
            return False
            
        # Check if price has moved against position by more than trailing_stop_buffer
        if position_size > 0:  # Long position
            price_drop = (highest_price - current_price) / highest_price
            if price_drop > self.config.trailing_stop_buffer:
                self.logger.warning(
                    f"Trailing stop triggered for {agent_id} {asset}: Price dropped "
                    f"{price_drop:.2%} from high of {highest_price:.2f}"
                )
                self.trailing_stop_events += 1
                return True
        else:  # Short position
            price_rise = (current_price - highest_price) / highest_price
            if price_rise > self.config.trailing_stop_buffer:
                self.logger.warning(
                    f"Trailing stop triggered for {agent_id} {asset}: Price rose "
                    f"{price_rise:.2%} from low of {highest_price:.2f}"
                )
                self.trailing_stop_events += 1
                return True
                
        return False
    
    def calculate_var(self, agent_id: str) -> Optional[float]:
        """
        Calculate Value at Risk for an agent's returns.
        
        Args:
            agent_id: Identifier for the agent
            
        Returns:
            float: VaR value, or None if insufficient data
        """
        if not self.config.use_var:
            return None
            
        if agent_id not in self.returns_history or len(self.returns_history[agent_id]) < 2:
            return None
            
        returns = list(self.returns_history[agent_id])
        sorted_returns = sorted(returns)
        index = int((1 - self.config.var_confidence_level) * len(sorted_returns))
        var_estimate = abs(sorted_returns[index])
        
        return var_estimate
    
    def check_var_exceed(self, agent_id: str, current_return: float) -> Optional[str]:
        """
        Check if current return exceeds VaR threshold.
        
        Args:
            agent_id: Identifier for the agent
            current_return: Current period return
            
        Returns:
            str or None: Action to take if VaR is exceeded, None otherwise
        """
        var_value = self.calculate_var(agent_id)
        if var_value is None:
            return None
        
        # VaR is exceeded if the current loss (negative return) is greater than VaR
        if current_return < -var_value:
            self.logger.warning(
                f"VaR exceeded for {agent_id}: Current loss {-current_return:.2%} > "
                f"VaR {var_value:.2%}"
            )
            self.var_exceed_events += 1
            return self.config.action_on_var_exceed
            
        return None
    
    def check_max_drawdown(self, agent_id: str) -> bool:
        """
        Check if maximum drawdown has been exceeded.
        
        Args:
            agent_id: Identifier for the agent
            
        Returns:
            bool: True if max drawdown exceeded, False otherwise
        """
        if agent_id not in self.peak_values or agent_id not in self.current_values:
            return False
            
        peak_value = self.peak_values[agent_id]
        current_value = self.current_values[agent_id]
        
        drawdown = (peak_value - current_value) / peak_value
        
        if drawdown > self.config.max_drawdown_pct:
            if self.config.use_forced_liquidation and not self.liquidation_triggered.get(agent_id, False):
                self.liquidation_triggered[agent_id] = True
                self.forced_liquidation_events += 1
                self.logger.warning(
                    f"Forced liquidation for {agent_id}: Drawdown {drawdown:.2%} exceeds "
                    f"threshold {self.config.max_drawdown_pct:.2%}"
                )
                return True
                
        return False
    
    def get_risk_events_info(self) -> Dict[str, int]:
        """
        Get information about risk events that have occurred.
        
        Returns:
            dict: Dictionary with risk event counts
        """
        return {
            "stop_loss_events": self.stop_loss_events,
            "trailing_stop_events": self.trailing_stop_events,
            "var_exceed_events": self.var_exceed_events,
            "forced_liquidation_events": self.forced_liquidation_events
        } 