"""
Capital management for multi-asset reinforcement learning trading environment.

This module provides capital allocation and management functionality for trading environments,
supporting both shared and isolated capital modes.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np

logger = logging.getLogger(__name__)

class CapitalManager:
    """
    Capital manager for handling fund allocation across multiple assets or agents.
    
    Features:
    - Supports both shared and isolated capital modes
    - Manages allocations, margin requirements, and capital efficiency
    - Tracks performance metrics by asset/agent
    - Implements capital protection mechanisms
    
    Implementation Notes:
    - Shared mode: All agents/assets share a single capital pool
    - Isolated mode: Each agent/asset has its own dedicated capital
    - Compatible with both MultiAssetTradingEnv and MultiAgentTradingEnv
    
    Recent Changes:
    - Initial implementation with shared and isolated capital modes
    - Added capital efficiency tracking and rebalancing
    - Implemented maximum drawdown protection
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        mode: str = "shared",  # "shared" or "isolated"
        assets: Optional[List[str]] = None,
        allocation_weights: Optional[Dict[str, float]] = None,
        max_leverage: float = 1.0,
        drawdown_limit: float = 0.2,  # 20% max drawdown before reducing exposure
        auto_rebalance: bool = False,
        rebalance_threshold: float = 0.1  # 10% deviation from target allocation
    ):
        """
        Initialize the capital manager.
        
        Args:
            initial_capital: Starting capital
            mode: Capital management mode ("shared" or "isolated")
            assets: List of asset identifiers
            allocation_weights: Initial capital allocation weights by asset
            max_leverage: Maximum leverage allowed
            drawdown_limit: Maximum drawdown before reducing exposure
            auto_rebalance: Whether to automatically rebalance allocations
            rebalance_threshold: Threshold for automatic rebalancing
        """
        self.initial_capital = initial_capital
        self.total_capital = initial_capital
        self.mode = mode.lower()
        self.assets = assets or []
        self.max_leverage = max_leverage
        self.drawdown_limit = drawdown_limit
        self.auto_rebalance = auto_rebalance
        self.rebalance_threshold = rebalance_threshold
        
        # Initialize allocation weights
        if allocation_weights is None:
            # Default to equal allocation
            if len(self.assets) > 0:
                equal_weight = 1.0 / len(self.assets)
                self.allocation_weights = {asset: equal_weight for asset in self.assets}
            else:
                self.allocation_weights = {}
        else:
            # Normalize provided weights
            total_weight = sum(allocation_weights.values())
            self.allocation_weights = {
                asset: weight / total_weight for asset, weight in allocation_weights.items()
            }
        
        # Initialize capital, allocation, and usage tracking
        if self.mode == "isolated":
            # Each asset gets its own capital pool
            self.allocated_capital = {
                asset: self.initial_capital * self.allocation_weights.get(asset, 0.0)
                for asset in self.assets
            }
            # Track maximum allocated capital for drawdown calculation
            self.max_allocated_capital = self.allocated_capital.copy()
        else:
            # Shared capital pool
            self.allocated_capital = {"shared": self.initial_capital}
            self.max_allocated_capital = {"shared": self.initial_capital}
        
        # Track capital usage
        self.used_capital = {asset: 0.0 for asset in self.assets}
        self.current_leverage = 0.0
        
        # Performance tracking
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.returns_history = []
        
        # Position value tracking
        self.position_values = {asset: 0.0 for asset in self.assets}
        
        logger.info(
            f"Initialized {mode} mode capital manager with {initial_capital:.2f} "
            f"initial capital for {len(self.assets)} assets"
        )
    
    def get_available_capital(self, asset: str) -> float:
        """
        Get available capital for a specific asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Available capital for the asset
        """
        if self.mode == "isolated":
            # In isolated mode, each asset has its own capital pool
            return self.allocated_capital.get(asset, 0.0) - self.used_capital.get(asset, 0.0)
        else:
            # In shared mode, all assets share the capital pool,
            # but we need to account for capital already used by other assets
            total_used = sum(self.used_capital.values())
            return self.allocated_capital.get("shared", 0.0) - total_used
    
    def allocate_capital(self, asset: str, amount: float) -> float:
        """
        Allocate capital to a specific asset.
        
        Args:
            asset: Asset identifier
            amount: Amount to allocate (positive) or deallocate (negative)
            
        Returns:
            Actually allocated amount (may be less than requested if insufficient funds)
        """
        if asset not in self.assets:
            logger.warning(f"Attempting to allocate capital to unknown asset: {asset}")
            return 0.0
        
        available = self.get_available_capital(asset)
        
        if amount > available:
            # Cap at available capital
            actual_amount = available
            logger.warning(
                f"Requested capital allocation of {amount:.2f} for {asset} exceeds "
                f"available capital of {available:.2f}, allocating {actual_amount:.2f}"
            )
        else:
            actual_amount = amount
        
        # Update used capital
        self.used_capital[asset] = self.used_capital.get(asset, 0.0) + actual_amount
        
        # Update leverage
        self._update_leverage()
        
        return actual_amount
    
    def release_capital(self, asset: str, amount: float) -> float:
        """
        Release previously allocated capital.
        
        Args:
            asset: Asset identifier
            amount: Amount to release
            
        Returns:
            Actually released amount
        """
        if asset not in self.assets:
            logger.warning(f"Attempting to release capital from unknown asset: {asset}")
            return 0.0
        
        used = self.used_capital.get(asset, 0.0)
        
        if amount > used:
            # Can't release more than what's used
            actual_amount = used
            logger.warning(
                f"Requested capital release of {amount:.2f} for {asset} exceeds "
                f"used capital of {used:.2f}, releasing {actual_amount:.2f}"
            )
        else:
            actual_amount = amount
        
        # Update used capital
        self.used_capital[asset] = used - actual_amount
        
        # Update leverage
        self._update_leverage()
        
        return actual_amount
    
    def update_capital(self, capital_changes: Dict[str, float]) -> float:
        """
        Update capital based on trading results.
        
        Args:
            capital_changes: Capital changes by asset
            
        Returns:
            Net capital change
        """
        net_change = 0.0
        
        if self.mode == "isolated":
            # Update each asset's capital separately
            for asset, change in capital_changes.items():
                if asset in self.allocated_capital:
                    self.allocated_capital[asset] += change
                    # Update max allocated capital for drawdown calculation
                    self.max_allocated_capital[asset] = max(
                        self.max_allocated_capital[asset], self.allocated_capital[asset]
                    )
                    net_change += change
                else:
                    logger.warning(f"Attempting to update capital for unknown asset: {asset}")
        else:
            # Update shared capital pool
            for change in capital_changes.values():
                net_change += change
            
            self.allocated_capital["shared"] += net_change
            # Update max allocated capital for drawdown calculation
            self.max_allocated_capital["shared"] = max(
                self.max_allocated_capital["shared"], self.allocated_capital["shared"]
            )
        
        # Update total capital
        self.total_capital += net_change
        
        # Update peak capital and drawdown
        self.peak_capital = max(self.peak_capital, self.total_capital)
        self.current_drawdown = 1.0 - (self.total_capital / self.peak_capital)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check if we should reduce exposure due to drawdown
        if self.current_drawdown > self.drawdown_limit:
            self._reduce_exposure_for_drawdown()
        
        # Track return
        if len(self.returns_history) == 0:
            prev_capital = self.initial_capital
        else:
            prev_capital = self.total_capital - net_change
        
        if prev_capital > 0:
            self.returns_history.append(net_change / prev_capital)
        
        # Check if rebalancing is needed
        if self.auto_rebalance:
            self._check_rebalance()
        
        return net_change
    
    def update_position_values(self, position_values: Dict[str, float]):
        """
        Update the current value of positions.
        
        Args:
            position_values: Current value of positions by asset
        """
        self.position_values = position_values.copy()
        self._update_leverage()
    
    def _update_leverage(self):
        """Update the current leverage ratio."""
        total_position_value = sum(self.position_values.values())
        if self.total_capital > 0:
            self.current_leverage = total_position_value / self.total_capital
        else:
            self.current_leverage = 0.0
    
    def _reduce_exposure_for_drawdown(self):
        """Reduce exposure when drawdown exceeds limit."""
        reduction_factor = 1.0 - (self.current_drawdown / self.drawdown_limit)
        
        logger.warning(
            f"Reducing exposure to {reduction_factor:.2f} due to drawdown of "
            f"{self.current_drawdown:.2%} exceeding limit of {self.drawdown_limit:.2%}"
        )
        
        # Adjust max leverage based on drawdown
        self.max_leverage *= reduction_factor
    
    def _check_rebalance(self):
        """Check if rebalancing is needed and perform it if necessary."""
        if len(self.assets) <= 1:
            # No need to rebalance with only one asset
            return
        
        # Calculate current allocation percentages
        current_allocations = {}
        
        if self.mode == "isolated":
            # For isolated mode, use allocated capital directly
            total_allocated = sum(self.allocated_capital.values())
            if total_allocated > 0:
                for asset in self.assets:
                    current_allocations[asset] = self.allocated_capital.get(asset, 0.0) / total_allocated
        else:
            # For shared mode, use position values
            total_position_value = sum(self.position_values.values())
            if total_position_value > 0:
                for asset in self.assets:
                    current_allocations[asset] = self.position_values.get(asset, 0.0) / total_position_value
        
        # Check for deviations from target allocation
        needs_rebalance = False
        for asset in self.assets:
            target = self.allocation_weights.get(asset, 0.0)
            current = current_allocations.get(asset, 0.0)
            
            if abs(target - current) > self.rebalance_threshold:
                needs_rebalance = True
                break
        
        if needs_rebalance:
            logger.info("Auto-rebalancing triggered due to allocation deviation")
            self.rebalance()
    
    def rebalance(self):
        """
        Rebalance allocations to match target weights.
        
        Returns:
            Dictionary of suggested position adjustments by asset
        """
        position_adjustments = {}
        
        if self.mode == "isolated":
            # For isolated mode, directly adjust allocated capital
            total_capital = sum(self.allocated_capital.values())
            
            for asset in self.assets:
                target_capital = total_capital * self.allocation_weights.get(asset, 0.0)
                current_capital = self.allocated_capital.get(asset, 0.0)
                adjustment = target_capital - current_capital
                
                if abs(adjustment) > 0.01:  # Skip tiny adjustments
                    self.allocated_capital[asset] = target_capital
                    position_adjustments[asset] = adjustment
        else:
            # For shared mode, suggest position value adjustments
            total_position_value = sum(self.position_values.values())
            
            if total_position_value > 0:
                for asset in self.assets:
                    target_value = total_position_value * self.allocation_weights.get(asset, 0.0)
                    current_value = self.position_values.get(asset, 0.0)
                    adjustment = target_value - current_value
                    
                    if abs(adjustment) > 0.01:  # Skip tiny adjustments
                        position_adjustments[asset] = adjustment
        
        logger.info(f"Rebalance adjustments: {position_adjustments}")
        return position_adjustments
    
    def get_allocation_status(self) -> Dict[str, Any]:
        """
        Get the current allocation status.
        
        Returns:
            Dictionary with allocation status information
        """
        status = {
            "total_capital": self.total_capital,
            "initial_capital": self.initial_capital,
            "mode": self.mode,
            "current_leverage": self.current_leverage,
            "max_leverage": self.max_leverage,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "allocation_weights": self.allocation_weights.copy(),
            "position_values": self.position_values.copy(),
            "used_capital": self.used_capital.copy()
        }
        
        if self.mode == "isolated":
            status["allocated_capital"] = self.allocated_capital.copy()
        else:
            status["shared_capital_pool"] = self.allocated_capital.get("shared", 0.0)
        
        if len(self.returns_history) > 0:
            status["avg_return"] = np.mean(self.returns_history)
            status["return_volatility"] = np.std(self.returns_history)
        
        return status
    
    def reset(self):
        """Reset the capital manager to initial state."""
        self.total_capital = self.initial_capital
        
        if self.mode == "isolated":
            # Each asset gets its own capital pool
            self.allocated_capital = {
                asset: self.initial_capital * self.allocation_weights.get(asset, 0.0)
                for asset in self.assets
            }
            self.max_allocated_capital = self.allocated_capital.copy()
        else:
            # Shared capital pool
            self.allocated_capital = {"shared": self.initial_capital}
            self.max_allocated_capital = {"shared": self.initial_capital}
        
        # Reset tracking variables
        self.used_capital = {asset: 0.0 for asset in self.assets}
        self.position_values = {asset: 0.0 for asset in self.assets}
        self.current_leverage = 0.0
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.returns_history = []
        
        logger.info(f"Reset capital manager to initial state with {self.initial_capital:.2f} capital")


# Adapter class to integrate with MultiAssetTradingEnv
class MultiAssetCapitalManager(CapitalManager):
    """
    Capital manager adapter for MultiAssetTradingEnv.
    
    Features:
    - Seamlessly integrates with MultiAssetTradingEnv
    - Handles position updates and capital allocation
    - Provides capital constraints for trading decisions
    
    Implementation Notes:
    - Extends CapitalManager with environment-specific functionality
    - Tracks position values based on environment state
    - Updates capital based on trading results
    """
    
    def __init__(
        self,
        env,  # MultiAssetTradingEnv instance
        mode: str = "shared",
        allocation_weights: Optional[Dict[str, float]] = None,
        max_leverage: float = 1.0,
        drawdown_limit: float = 0.2,
        auto_rebalance: bool = False
    ):
        """
        Initialize the MultiAssetCapitalManager.
        
        Args:
            env: MultiAssetTradingEnv instance
            mode: Capital management mode ("shared" or "isolated")
            allocation_weights: Initial capital allocation weights by asset
            max_leverage: Maximum leverage allowed
            drawdown_limit: Maximum drawdown before reducing exposure
            auto_rebalance: Whether to automatically rebalance allocations
        """
        self.env = env
        
        super().__init__(
            initial_capital=env.initial_balance,
            mode=mode,
            assets=env.assets,
            allocation_weights=allocation_weights,
            max_leverage=max_leverage,
            drawdown_limit=drawdown_limit,
            auto_rebalance=auto_rebalance
        )
    
    def update_from_env_state(self):
        """Update capital manager state from environment state."""
        # Update position values based on current prices and positions
        position_values = {}
        for asset in self.assets:
            position = self.env.positions.get(asset, 0.0)
            price = self.env.prices.get(asset, 0.0)
            position_values[asset] = position * price
        
        self.update_position_values(position_values)
        
        # Update total capital based on environment's portfolio value
        capital_change = self.env.portfolio_value - self.total_capital
        capital_changes = {asset: 0.0 for asset in self.assets}
        
        # Distribute the change proportionally to position values
        total_position_value = sum(position_values.values())
        if total_position_value > 0:
            for asset in self.assets:
                asset_proportion = position_values.get(asset, 0.0) / total_position_value
                capital_changes[asset] = capital_change * asset_proportion
        
        self.update_capital(capital_changes)
    
    def check_capital_constraints(self, asset: str, position_change: float) -> float:
        """
        Check if a position change satisfies capital constraints.
        
        Args:
            asset: Asset identifier
            position_change: Requested position change
            
        Returns:
            Adjusted position change that satisfies constraints
        """
        price = self.env.prices.get(asset, 0.0)
        if price <= 0:
            return 0.0
        
        capital_required = abs(position_change) * price
        available_capital = self.get_available_capital(asset)
        
        if capital_required > available_capital:
            # Scale down the position change
            adjusted_position_change = np.sign(position_change) * (available_capital / price)
            logger.warning(
                f"Position change for {asset} scaled down from {position_change:.6f} to "
                f"{adjusted_position_change:.6f} due to capital constraints"
            )
            return adjusted_position_change
        
        return position_change
    
    def allocate_for_position(self, asset: str, position_change: float) -> float:
        """
        Allocate capital for a position change.
        
        Args:
            asset: Asset identifier
            position_change: Position change amount
            
        Returns:
            Amount of capital allocated
        """
        price = self.env.prices.get(asset, 0.0)
        if price <= 0:
            return 0.0
        
        capital_required = abs(position_change) * price
        return self.allocate_capital(asset, capital_required)
    
    def get_max_position_size(self, asset: str) -> float:
        """
        Get the maximum position size for an asset based on available capital.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Maximum position size
        """
        price = self.env.prices.get(asset, 0.0)
        if price <= 0:
            return 0.0
        
        available_capital = self.get_available_capital(asset)
        return available_capital / price
    
    def get_max_leverage_position(self, asset: str) -> float:
        """
        Get the maximum position size for an asset based on leverage constraints.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Maximum position size with leverage
        """
        price = self.env.prices.get(asset, 0.0)
        if price <= 0:
            return 0.0
        
        max_position_value = self.total_capital * self.max_leverage
        current_position_value = sum(self.position_values.values())
        available_position_value = max_position_value - current_position_value
        
        if available_position_value <= 0:
            return 0.0
        
        return available_position_value / price


# Adapter class to integrate with MultiAgentTradingEnv
class MultiAgentCapitalManager(CapitalManager):
    """
    Capital manager adapter for MultiAgentTradingEnv.
    
    Features:
    - Seamlessly integrates with MultiAgentTradingEnv
    - Manages capital allocation across multiple agents
    - Provides capital constraints for agent trading decisions
    
    Implementation Notes:
    - Extends CapitalManager with agent-specific functionality
    - Maps agents to assets for capital tracking
    - Updates capital based on agent performance
    """
    
    def __init__(
        self,
        env,  # MultiAgentTradingEnv instance
        mode: str = "isolated",  # Default to isolated mode for agents
        allocation_weights: Optional[Dict[str, float]] = None,
        max_leverage: float = 1.0,
        drawdown_limit: float = 0.2,
        auto_rebalance: bool = False
    ):
        """
        Initialize the MultiAgentCapitalManager.
        
        Args:
            env: MultiAgentTradingEnv instance
            mode: Capital management mode ("shared" or "isolated")
            allocation_weights: Initial capital allocation weights by agent
            max_leverage: Maximum leverage allowed
            drawdown_limit: Maximum drawdown before reducing exposure
            auto_rebalance: Whether to automatically rebalance allocations
        """
        self.env = env
        
        super().__init__(
            initial_capital=env.initial_balance,
            mode=mode,
            assets=env.agents,  # Use agent IDs as "assets"
            allocation_weights=allocation_weights,
            max_leverage=max_leverage,
            drawdown_limit=drawdown_limit,
            auto_rebalance=auto_rebalance
        )
        
        # Map between agent IDs and their asset assignments
        self.agent_assets = {}
        if hasattr(env, "agent_assets"):
            self.agent_assets = env.agent_assets
    
    def update_from_env_state(self):
        """Update capital manager state from environment state."""
        # Update position values based on current prices and positions
        position_values = {}
        agent_portfolios = {}
        
        # Get portfolio values for each agent
        if hasattr(self.env, "agent_portfolio_values"):
            agent_portfolios = self.env.agent_portfolio_values
        else:
            # Fallback calculation
            for agent_id in self.assets:  # assets contains agent IDs
                if hasattr(self.env, "positions") and hasattr(self.env, "balances"):
                    position = self.env.positions.get(agent_id, 0.0)
                    balance = self.env.balances.get(agent_id, 0.0)
                    price = 0.0
                    
                    # Try to get price from the environment
                    if hasattr(self.env, "prices"):
                        if agent_id in self.env.prices:
                            price = self.env.prices[agent_id]
                        elif agent_id in self.agent_assets:
                            asset = self.agent_assets[agent_id]
                            price = self.env.prices.get(asset, 0.0)
                    
                    position_value = position * price
                    portfolio_value = balance + position_value
                    agent_portfolios[agent_id] = portfolio_value
        
        # Update position values
        for agent_id in self.assets:
            position_values[agent_id] = agent_portfolios.get(agent_id, 0.0) - self.allocated_capital.get(agent_id, 0.0)
            if position_values[agent_id] < 0:
                position_values[agent_id] = 0
        
        self.update_position_values(position_values)
        
        # Update total capital based on agent portfolio values
        capital_changes = {}
        for agent_id in self.assets:
            current_portfolio = agent_portfolios.get(agent_id, 0.0)
            prev_portfolio = self.allocated_capital.get(agent_id, 0.0)
            capital_changes[agent_id] = current_portfolio - prev_portfolio
        
        self.update_capital(capital_changes)
    
    def get_agent_allocation(self, agent_id: str) -> Dict[str, float]:
        """
        Get the capital allocation for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with allocation information
        """
        allocation = {
            "agent_id": agent_id,
            "total_capital": self.total_capital,
            "mode": self.mode
        }
        
        if self.mode == "isolated":
            allocation["allocated_capital"] = self.allocated_capital.get(agent_id, 0.0)
            allocation["used_capital"] = self.used_capital.get(agent_id, 0.0)
            allocation["available_capital"] = self.get_available_capital(agent_id)
        else:
            allocation["shared_capital_pool"] = self.allocated_capital.get("shared", 0.0)
            allocation["total_used_capital"] = sum(self.used_capital.values())
            allocation["available_capital"] = self.get_available_capital(agent_id)
        
        allocation["position_value"] = self.position_values.get(agent_id, 0.0)
        allocation["allocation_weight"] = self.allocation_weights.get(agent_id, 0.0)
        
        return allocation 