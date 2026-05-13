"""
Risk Manager for Multi-Agent Trading Environment

This module provides risk management functionality for the RL trading environment,
including stop-loss, trailing stop, VaR/CVaR, and other risk controls.

Features:
- Position-level risk management (stop-loss, trailing stop)
- Portfolio-level risk management (VaR, CVaR)
- Drawdown monitoring and forced liquidation
- Risk metrics tracking and reporting
- Cross-asset correlation monitoring and correlation-based position sizing
- Multi-asset portfolio VaR/CVaR calculation
- Portfolio-wide Stop-Loss/Trailing Stop

Implementation Notes:
- Designed specifically for integration with MultiAgentTradingEnv
- Supports both per-agent and portfolio-wide risk constraints
- Provides signals for position adjustment or liquidation based on risk thresholds
- Maintains historical risk metrics for analysis and visualization
- Uses correlation matrix to identify concentrated risks

Recent Changes:
- Refactored to inherit from RiskManagerBase abstract class
- Added cross-asset correlation tracking and position adjustment
- Implemented portfolio-level VaR/CVaR with covariance matrix
- Added portfolio-wide Stop-Loss and Trailing Stop
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
import threading
from collections import deque
from scipy.stats import norm

from risk_management.risk_manager_base import RiskManagerBase, RiskConfigBase
from risk_management.unified_risk_manager import UnifiedRiskManager

from deployment.monitoring.alerter import TradingAlerter


@dataclass
class RLRiskConfig(RiskConfigBase):
    """
    Configuration for risk management in the RL trading environment.

    Features:
    - Stop loss and trailing stop settings
    - VaR/CVaR configuration
    - Drawdown limits and liquidation triggers
    - Correlation-based risk adjustments
    - Portfolio-level risk controls

    Implementation Notes:
    - All percentages are expressed as decimals (0.01 = 1%)
    - Default values are set to conservative levels
    - Can be loaded from environment configuration
    """
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
    use_parametric_var: bool = False  # Use parametric (True) or historical (False) VaR calculation

    # Action on VaR exceeding threshold
    action_on_var_exceed: str = "reduce_position"  # "reduce_position" or "close_position"


class RLRiskManager(RiskManagerBase):
    """
    Risk manager for RL trading environment.

    Provides risk management functionality including stop-loss, trailing stop,
    VaR/CVaR calculation and position management based on risk thresholds.
    Also includes correlation-based risk management and portfolio-level controls.
    """

    def __init__(self, config: RLRiskConfig, alerter: Optional[TradingAlerter] = None, audit_logger=None):
        """
        Initialize the risk manager with the given configuration.

        Args:
            config: Risk management configuration
            alerter: Optional TradingAlerter for risk event notifications
            audit_logger: Optional AuditLogger for immutable risk event recording
        """
        import warnings
        warnings.warn(
            "RLRiskManager is deprecated and will be removed in a future phase. "
            "Use UnifiedRiskManager directly or via the factory.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)
        self.config = config
        self.alerter = alerter
        self._audit_logger = audit_logger
        self._lock = threading.Lock()
        # Composition: delegate core risk computations to UnifiedRiskManager
        _var_method = "parametric" if config.use_parametric_var else "historical"
        self._unified = UnifiedRiskManager(mode="live", var_method=_var_method)

        # Portfolio tracking
        self.peak_values = {}  # Dict[agent_id, peak_value]
        self.current_values = {}  # Dict[agent_id, current_value]
        self.position_highest_values = {}  # Dict[agent_id, Dict[asset, highest_value]]
        self.liquidation_triggered = {}  # Dict[agent_id, bool]

        # VaR tracking
        self.returns_history = {}  # Dict[agent_id, deque]

        # Portfolio tracking
        self.portfolio_peak_value = 0.0
        self.portfolio_current_value = 0.0

        # Multi-asset returns history for correlation and portfolio VaR
        self.asset_returns_history = {}  # Dict[asset, deque]
        self.asset_prices_history = {}   # Dict[asset, deque]
        self.correlation_matrix = None
        self.covariance_matrix = None

        # Metrics for logging
        self.stop_loss_events = 0
        self.trailing_stop_events = 0
        self.var_exceed_events = 0
        self.forced_liquidation_events = 0
        self.correlation_adjustment_events = 0
        self.portfolio_stop_loss_events = 0
        self.portfolio_var_exceed_events = 0

    def reset(self):
        """Reset all risk manager state."""
        with self._lock:
            self.peak_values = {}
            self.current_values = {}
            self.position_highest_values = {}
            self.liquidation_triggered = {}
            self.returns_history = {}

            self.portfolio_peak_value = 0.0
            self.portfolio_current_value = 0.0

            self.asset_returns_history = {}
            self.asset_prices_history = {}
            self.correlation_matrix = None
            self.covariance_matrix = None

            self.stop_loss_events = 0
            self.trailing_stop_events = 0
            self.var_exceed_events = 0
            self.forced_liquidation_events = 0
            self.correlation_adjustment_events = 0
            self.portfolio_stop_loss_events = 0
            self.portfolio_var_exceed_events = 0

    def calculate_stop_loss(self, entry_price: float, position_size: float, is_long: bool = True) -> float:
        """
        Calculate stop loss price based on entry price and position direction.

        Args:
            entry_price: Entry price of the position
            position_size: Size of the position (positive for long, negative for short)
            is_long: Whether the position is long (True) or short (False)

        Returns:
            float: Stop loss price
        """
        if is_long:
            return entry_price * (1 - self.config.stop_loss_threshold)
        else:
            return entry_price * (1 + self.config.stop_loss_threshold)

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
            with self._lock:
                self.stop_loss_events += 1
            if self.alerter is not None:
                self.alerter.send_alert(
                    f"Stop loss triggered for agent '{agent_id}': "
                    f"entry={entry_price:.4f} current={current_price:.4f} "
                    f"loss={abs(pct_change):.1%}",
                    level="WARNING",
                )
            if self._audit_logger is not None:
                self._audit_logger.log_risk_event({
                    "event": "stop_loss",
                    "agent_id": agent_id,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pct_loss": abs(pct_change),
                })
            return True

        return False

    def update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """Update trailing stop high-water-mark for a symbol.

        Uses position_highest_values with key format matching check_trailing_stop().
        When called without agent_id context, uses "_default_{symbol}" as key.
        """
        key = f"_default_{symbol}"
        with self._lock:
            if key not in self.position_highest_values or current_price > self.position_highest_values[key]:
                self.position_highest_values[key] = current_price

    def compute_var(self, agent_id_or_returns: Union[str, np.ndarray]) -> Optional[float]:
        """
        Compute Value at Risk (VaR).

        Supports both:
        1. Passing an agent_id to compute VaR from stored returns
        2. Passing a returns array directly

        Args:
            agent_id_or_returns: Agent ID (str) or returns array (np.ndarray)

        Returns:
            Optional[float]: VaR at the configured confidence level, or None if insufficient data
        """
        returns = None

        if isinstance(agent_id_or_returns, str):
            agent_id = agent_id_or_returns
            if agent_id in self.returns_history and len(self.returns_history[agent_id]) >= 10:
                returns = np.array(list(self.returns_history[agent_id]))
            else:
                return None
        else:
            returns = agent_id_or_returns

        if len(returns) < 10:
            return None

        return self._unified.compute_var(
            returns,
            confidence_level=self.config.var_confidence_level,
            var_method="parametric" if self.config.use_parametric_var else "historical",
        )

    def calculate_var(self, *args, **kwargs) -> Optional[float]:
        """Deprecated. Use compute_var() instead."""
        import warnings
        warnings.warn(
            "calculate_var() is deprecated; use compute_var()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_var(*args, **kwargs)

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns:
            Dict[str, Any]: Dictionary of current risk metrics
        """
        return {
            "stop_loss_events": self.stop_loss_events,
            "trailing_stop_events": self.trailing_stop_events,
            "var_exceed_events": self.var_exceed_events,
            "forced_liquidation_events": self.forced_liquidation_events,
            "correlation_adjustment_events": self.correlation_adjustment_events,
            "portfolio_stop_loss_events": self.portfolio_stop_loss_events,
            "portfolio_var_exceed_events": self.portfolio_var_exceed_events
        }

    def update_portfolio_values(self, portfolio_values: Dict[str, float]):
        """
        Update portfolio values for each agent and track peak values.

        Args:
            portfolio_values: Dictionary mapping agent_id to portfolio value
        """
        with self._lock:
            for agent_id, value in portfolio_values.items():
                self.current_values[agent_id] = value

                # Initialize or update peak value
                if agent_id not in self.peak_values or value > self.peak_values[agent_id]:
                    self.peak_values[agent_id] = value

                # Initialize liquidation flag if not exists
                if agent_id not in self.liquidation_triggered:
                    self.liquidation_triggered[agent_id] = False

            # Update portfolio total value
            self.portfolio_current_value = sum(portfolio_values.values())

            # Update portfolio peak value
            if self.portfolio_peak_value == 0 or self.portfolio_current_value > self.portfolio_peak_value:
                self.portfolio_peak_value = self.portfolio_current_value

    # Add other RL-specific risk management methods below
    # (These are methods from the original RiskManager class in envs/risk_manager.py)

    def record_returns(self, returns: Dict[str, float]):
        """
        Record returns for VaR calculation.

        Args:
            returns: Dictionary mapping agent_id to return value
        """
        if not self.config.use_var:
            return

        with self._lock:
            for agent_id, ret in returns.items():
                if agent_id not in self.returns_history:
                    self.returns_history[agent_id] = deque(maxlen=self.config.rolling_var_window)
                self.returns_history[agent_id].append(ret)

    def _record_asset_data(self, asset_prices: Dict[str, float], asset_returns: Dict[str, float]):
        """
        Record asset prices and returns for correlation and portfolio VaR calculation.

        Args:
            asset_prices: Dictionary mapping asset to current price
            asset_returns: Dictionary mapping asset to current return
        """
        if not (self.config.use_correlation or self.config.use_portfolio_var):
            return

        with self._lock:
            # Record prices
            for asset, price in asset_prices.items():
                if asset not in self.asset_prices_history:
                    self.asset_prices_history[asset] = deque(maxlen=self.config.correlation_window)
                self.asset_prices_history[asset].append(price)

            # Record returns
            for asset, ret in asset_returns.items():
                if asset not in self.asset_returns_history:
                    self.asset_returns_history[asset] = deque(maxlen=self.config.correlation_window)
                self.asset_returns_history[asset].append(ret)

        # Update correlation and covariance matrices if we have enough data
        self._update_correlation_matrix()

    def _update_correlation_matrix(self) -> None:
        """Update correlation and covariance matrices based on asset return histories."""
        if not self.config.use_correlation:
            return

        with self._lock:
            # Check if we have enough assets with enough data
            assets_with_data = [
                asset for asset, returns in self.asset_returns_history.items()
                if len(returns) >= 10  # Need at least 10 data points for meaningful correlation
            ]

            if len(assets_with_data) < 2:
                return  # Need at least 2 assets for correlation

            # Create DataFrame from return histories
            returns_data = {}

            # Find minimum length across all assets to ensure equal length arrays
            min_length = min(len(self.asset_returns_history[asset]) for asset in assets_with_data)

            for asset in assets_with_data:
                # Take only the last min_length elements to ensure all arrays have the same length
                returns_data[asset] = list(self.asset_returns_history[asset])[-min_length:]

        # Create DataFrame and calculate correlation/covariance matrices (outside lock — CPU-bound)
        try:
            df = pd.DataFrame(returns_data)
            corr = df.corr()
            cov = df.cov()
            with self._lock:
                self.correlation_matrix = corr
                self.covariance_matrix = cov
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            # Keep existing matrices if calculation fails

    # Legacy compatibility methods
    def _get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """
        Get the current correlation matrix.

        Returns:
            Optional[pd.DataFrame]: Correlation matrix or None if not available
        """
        return self.correlation_matrix

    def get_correlation_adjustment(self, asset1: str, asset2_or_positions: Union[str, Dict[str, float]]) -> float:
        """
        Calculate position size adjustment factor based on correlations.

        This method supports both:
        1. (asset1, asset2) to compare two specific assets
        2. (asset, position_sizes) to check one asset against all assets with positions

        Args:
            asset1: First asset to check
            asset2_or_positions: Either:
                - Second asset name (str) to check correlation with asset1
                - Dictionary of position sizes for all assets to check against asset1

        Returns:
            float: Adjustment factor (1.0 = no adjustment, < 1.0 = reduce position)
        """
        # If correlation tracking is disabled
        if not self.config.use_correlation:
            return 1.0

        # Check if correlation matrix exists
        if self.correlation_matrix is None:
            return 1.0

        # Old interface: asset and position sizes dictionary
        if isinstance(asset2_or_positions, dict):
            # Count how many highly correlated assets we have positions in
            correlated_assets = 0
            for other_asset, size in asset2_or_positions.items():
                if abs(size) < 1e-8 or other_asset == asset1:
                    continue

                # Check correlation between assets
                if other_asset in self.correlation_matrix.columns and self._check_correlation(asset1, other_asset):
                    correlated_assets += 1

            # If we have positions in correlated assets, reduce position size
            if correlated_assets > 0:
                self.correlation_adjustment_events += 1
                return self.config.correlation_risk_reduction
            return 1.0

        # New interface: two asset names
        # Check if correlation is above threshold
        if self._check_correlation(asset1, asset2_or_positions):
            self.correlation_adjustment_events += 1
            return self.config.correlation_risk_reduction
        return 1.0

    def _check_correlation(self, asset1: str, asset2: str) -> bool:
        """
        Check if correlation between two assets exceeds the threshold.

        Args:
            asset1: First asset name
            asset2: Second asset name

        Returns:
            bool: True if correlation exceeds threshold, False otherwise
        """
        if self.correlation_matrix is None:
            return False

        if asset1 not in self.correlation_matrix.index or asset2 not in self.correlation_matrix.columns:
            return False

        correlation = float(self.correlation_matrix.loc[asset1, asset2])
        # Delegate to UnifiedRiskManager
        return self._unified.check_correlation(correlation, self.config.correlation_threshold)

    def _check_portfolio_stop_loss(self) -> bool:
        """
        Check if portfolio-wide stop loss has been triggered.

        Returns:
            bool: True if portfolio stop loss triggered, False otherwise
        """
        if (not self.config.use_portfolio_stop_loss or
            self.portfolio_peak_value == 0 or
            self.portfolio_current_value == 0):
            return False

        # Calculate portfolio drawdown
        drawdown = (self.portfolio_peak_value - self.portfolio_current_value) / self.portfolio_peak_value

        if drawdown > self.config.portfolio_stop_loss_threshold:
            self.portfolio_stop_loss_events += 1
            if self._audit_logger is not None:
                self._audit_logger.log_risk_event({
                    "event": "portfolio_stop_loss",
                    "portfolio_peak": self.portfolio_peak_value,
                    "portfolio_current": self.portfolio_current_value,
                    "drawdown_pct": drawdown,
                    "threshold_pct": self.config.portfolio_stop_loss_threshold,
                })
            return True

        return False

    def _check_portfolio_trailing_stop(self) -> bool:
        """
        Check if portfolio-wide trailing stop has been triggered.

        Returns:
            bool: True if portfolio trailing stop triggered, False otherwise
        """
        if (not self.config.use_portfolio_trailing_stop or
            self.portfolio_peak_value == 0 or
            self.portfolio_current_value == 0):
            return False

        # Calculate portfolio drawdown from peak
        drawdown = (self.portfolio_peak_value - self.portfolio_current_value) / self.portfolio_peak_value

        return drawdown > self.config.portfolio_trailing_stop_buffer

    def _calculate_portfolio_var(self, position_sizes: Dict[str, float], prices: Dict[str, float]) -> Optional[float]:
        """
        Calculate portfolio Value at Risk using the covariance matrix.

        Args:
            position_sizes: Current position sizes for all assets
            prices: Current prices for all assets

        Returns:
            float: Portfolio VaR, or None if insufficient data
        """
        # Filter to assets with nonzero positions
        active_assets = [a for a, s in position_sizes.items() if abs(s) > 1e-8]
        if not active_assets:
            return 0.0

        # Filter further to assets with enough return history
        available = [
            a for a in active_assets
            if a in self.asset_returns_history and len(self.asset_returns_history[a]) >= 10
        ]

        # Single-asset fallback: use individual historical VaR
        if len(available) == 1:
            asset = available[0]
            returns = np.array(list(self.asset_returns_history[asset]))
            var = -np.percentile(returns, 100 * (1 - self.config.var_confidence_level))
            return max(0.0, float(var))

        if len(available) < 2:
            return getattr(self.config, 'portfolio_var_threshold', 0.02)  # not enough history

        min_len = min(len(self.asset_returns_history[a]) for a in available)
        returns_matrix = np.column_stack([
            list(self.asset_returns_history[a])[-min_len:] for a in available
        ])  # shape: (min_len, n_assets)

        # Position weights by market value
        pos_values = np.array([abs(position_sizes[a]) * prices.get(a, 1.0) for a in available])
        total_value = pos_values.sum()
        if total_value < 1e-8:
            return getattr(self.config, 'portfolio_var_threshold', 0.02)
        w = pos_values / total_value

        if self.config.use_parametric_var:
            cov = np.cov(returns_matrix.T)  # (n_assets, n_assets)
            portfolio_variance = float(w @ cov @ w)
            portfolio_std = np.sqrt(max(portfolio_variance, 0.0))
            portfolio_mean = float(w @ returns_matrix.mean(axis=0))
            var = -(portfolio_mean + norm.ppf(1 - self.config.var_confidence_level) * portfolio_std)
            return max(0.0, float(var))
        else:
            # Historical: reconstruct portfolio return series
            portfolio_returns = returns_matrix @ w
            var = -np.percentile(portfolio_returns, 100 * (1 - self.config.var_confidence_level))
            return max(0.0, float(var))

    def check_portfolio_var_exceed(self, position_sizes: Dict[str, float], prices: Dict[str, float],
                                  current_portfolio_return: float) -> bool:
        """
        Check if portfolio VaR exceeds threshold.

        Args:
            position_sizes: Current position sizes for all assets
            prices: Current prices for all assets
            current_portfolio_return: Current portfolio return

        Returns:
            bool: True if portfolio VaR is exceeded, False otherwise
        """
        if not self.config.use_portfolio_var:
            return False

        # Simple implementation for compatibility
        if current_portfolio_return < -self.config.portfolio_var_threshold:
            self.portfolio_var_exceed_events += 1
            return True

        return False

    def update_asset_price(self, asset: str, price: float) -> None:
        """
        Update price history for a single asset.

        Args:
            asset: Asset to update
            price: Current price
        """
        if asset not in self.asset_prices_history:
            self.asset_prices_history[asset] = deque(maxlen=self.config.correlation_window)

        self.asset_prices_history[asset].append(price)

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

        position_key = f"{agent_id}_{asset}"
        with self._lock:
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
                with self._lock:
                    self.trailing_stop_events += 1
                if self.alerter is not None:
                    self.alerter.send_alert(
                        f"Trailing stop triggered for agent '{agent_id}' asset '{asset}': "
                        f"high={highest_price:.4f} current={current_price:.4f} "
                        f"drop={price_drop:.1%}",
                        level="WARNING",
                    )
                if self._audit_logger is not None:
                    self._audit_logger.log_risk_event({
                        "event": "trailing_stop",
                        "agent_id": agent_id,
                        "asset": asset,
                        "highest_price": highest_price,
                        "current_price": current_price,
                        "pct_drop": price_drop,
                        "direction": "long",
                    })
                return True
        else:  # Short position
            price_rise = (current_price - highest_price) / highest_price
            if price_rise > self.config.trailing_stop_buffer:
                with self._lock:
                    self.trailing_stop_events += 1
                if self.alerter is not None:
                    self.alerter.send_alert(
                        f"Trailing stop triggered for agent '{agent_id}' asset '{asset}': "
                        f"low={highest_price:.4f} current={current_price:.4f} "
                        f"rise={price_rise:.1%}",
                        level="WARNING",
                    )
                if self._audit_logger is not None:
                    self._audit_logger.log_risk_event({
                        "event": "trailing_stop",
                        "agent_id": agent_id,
                        "asset": asset,
                        "highest_price": highest_price,
                        "current_price": current_price,
                        "pct_rise": price_rise,
                        "direction": "short",
                    })
                return True

        return False

    def _get_risk_events_info(self) -> Dict[str, int]:
        """
        Get information about risk events that have occurred.

        Returns:
            dict: Dictionary with risk event counts
        """
        with self._lock:
            return {
                "stop_loss_events": self.stop_loss_events,
                "trailing_stop_events": self.trailing_stop_events,
                "var_exceed_events": self.var_exceed_events,
                "forced_liquidation_events": self.forced_liquidation_events,
                "correlation_adjustment_events": self.correlation_adjustment_events,
                "portfolio_stop_loss_events": self.portfolio_stop_loss_events,
                "portfolio_var_exceed_events": self.portfolio_var_exceed_events,
            }

    def check_drawdown(self, agent_id_or_peak, peak_value=None, current_value=None) -> bool:
        """
        Check if drawdown limit has been exceeded.

        Supports three call patterns:
        1. check_drawdown(peak, current)          — 2-arg float
        2. check_drawdown("agent", peak, current) — 3-arg string agent_id
        3. check_drawdown("agent")                — lookup from stored values

        Returns:
            bool: True if drawdown limit exceeded, False otherwise
        """
        # Pattern 1: called with (peak_float, current_float)
        if isinstance(agent_id_or_peak, (int, float)):
            peak = agent_id_or_peak
            current = peak_value  # second positional arg
            breached = self._unified.check_drawdown(peak, current, self.config.max_drawdown_pct)
            if breached:
                if self.alerter is not None:
                    self.alerter.check_drawdown(current=current, peak=peak)
                if self._audit_logger is not None:
                    drawdown = (peak - current) / peak if peak > 0 else 0.0
                    self._audit_logger.log_risk_event({
                        "event": "drawdown_breach",
                        "agent_id": None,
                        "peak": peak,
                        "current": current,
                        "drawdown_pct": drawdown,
                        "threshold_pct": self.config.max_drawdown_pct,
                    })
            return breached

        # Pattern 2 & 3: string agent_id
        agent_id = agent_id_or_peak
        if peak_value is not None and current_value is not None:
            breached = self._unified.check_drawdown(peak_value, current_value, self.config.max_drawdown_pct)
            if breached and self.alerter is not None:
                self.alerter.check_drawdown(current=current_value, peak=peak_value)
            return breached

        # Pattern 3: lookup from stored values
        with self._lock:
            if agent_id not in self.peak_values or agent_id not in self.current_values:
                return False

            peak = self.peak_values[agent_id]
            current = self.current_values[agent_id]

            if peak <= 0:
                return False

            # Delegate core drawdown check to UnifiedRiskManager
            breached = self._unified.check_drawdown(peak, current, self.config.max_drawdown_pct)

            if breached:
                if self.config.use_forced_liquidation:
                    if agent_id not in self.liquidation_triggered:
                        self.liquidation_triggered[agent_id] = False
                    if not self.liquidation_triggered[agent_id]:
                        self.liquidation_triggered[agent_id] = True
                        self.forced_liquidation_events += 1

        if breached:
            if self.alerter is not None:
                self.alerter.check_drawdown(current=current, peak=peak)
            if self._audit_logger is not None:
                drawdown = (peak - current) / peak if peak > 0 else 0.0
                self._audit_logger.log_risk_event({
                    "event": "drawdown_breach",
                    "agent_id": agent_id if not isinstance(agent_id_or_peak, (int, float)) else None,
                    "peak": peak,
                    "current": current,
                    "drawdown_pct": drawdown,
                    "threshold_pct": self.config.max_drawdown_pct,
                })
            return True

        return False

    def check_max_drawdown(self, *args, **kwargs) -> bool:
        """Deprecated. Use check_drawdown() instead."""
        import warnings
        warnings.warn(
            "check_max_drawdown() is deprecated; use check_drawdown()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.check_drawdown(*args, **kwargs)

    def adjust_for_regime(self, action: float, regime_probs: np.ndarray) -> float:
        """Deprecated — semantic bug (vol_factor double-clips after max_position_size).
        Phase 8-Gamma uses pre-computed regime_track + bear_gate instead."""
        warnings.warn(
            "adjust_for_regime() is deprecated and must not be called; "
            "use regime_track/bear_gate in SingleAssetRLTradingEnv instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "adjust_for_regime() is blocked (deprecated, semantic bug). "
            "See RLRiskManager docstring for replacement."
        )

    def check_var_exceed(self, agent_id: str, current_return: float) -> Optional[str]:
        """
        Check if current return exceeds VaR threshold.

        Args:
            agent_id: Identifier for the agent
            current_return: Current period return

        Returns:
            str or None: Action to take if VaR is exceeded, None otherwise
        """
        var_value = None

        # Try to calculate VaR from returns history
        if agent_id in self.returns_history:
            returns = np.array(list(self.returns_history[agent_id]))
            if len(returns) >= 10:
                var_value = self.compute_var(returns)

        if var_value is None:
            return None

        # VaR is exceeded if the current loss (negative return) is greater than VaR
        if current_return < -var_value:
            self.var_exceed_events += 1
            return self.config.action_on_var_exceed

        return None
