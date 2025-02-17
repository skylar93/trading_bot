"""
Unified RiskManager that merges basic risk checks (drawdown, position size, daily trades)
with advanced portfolio risk (VaR, correlation, CVaR).
Positions are assumed to be Dict[str, Dict[str, float]]: 
  positions[symbol] = {"units": float, "avg_price": float, "cost_basis": float}
Also includes legacy methods (check_max_drawdown, calculate_stop_loss, etc.)
to satisfy older tests in test_risk_management.py
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Any, Union


@dataclass
class RiskConfig:
    """Merged configuration for risk management."""
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


class RiskManager:
    """
    Unified Risk Manager that handles:
      - daily trade limits
      - max drawdown
      - max leverage
      - position size clamp
      - min trade size
      - VaR / CVaR
      - correlation checks
      - plus legacy methods (check_max_drawdown, calculate_stop_loss, etc.)
        to satisfy older tests
    """
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    def reset(self):
        """Reset all internal state."""
        # For daily trade limits
        self.trade_counter = {}  # {date: int}
        
        # For drawdown tracking
        self.peak_value: Optional[float] = None
        self.current_drawdown: float = 0.0
        
        # For advanced portfolio risk
        self._asset_returns: Dict[str, pd.Series] = {}  # price/returns data
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_correlation_update: Optional[pd.Timestamp] = None

    # -----------------------------------------------------------------------
    #  Legacy or Additional methods (for older tests)
    # -----------------------------------------------------------------------
    def check_trade_limits(self, timestamp: pd.Timestamp) -> bool:
        """
        Return True if we have not exceeded the daily trade limit on this date.
        test_trade_limits() calls this.
        """
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        # if we've hit the limit, return False
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return False
        return True

    def update_trade_counter(self, timestamp: pd.Timestamp) -> None:
        """
        Legacy method for older tests that directly increment the trade counter.
        test_trade_limits calls this method by name.
        """
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1

    def check_max_drawdown(self, peak_value: float, current_value: float) -> bool:
        """
        Return True if drawdown is beyond config.max_drawdown_pct.
        test_drawdown_monitoring() calls this with (initial_value, current_value).
        """
        if peak_value <= 0:
            return False
        dd = (peak_value - current_value) / peak_value
        return dd > self.config.max_drawdown_pct

    def check_leverage_limits(self, portfolio_value: float, position_value: float) -> bool:
        """
        Return True if position_value/portfolio_value <= max_leverage.
        test_leverage_limits() calls this.
        """
        if portfolio_value <= 0:
            return False
        leverage = position_value / portfolio_value
        return leverage <= self.config.max_leverage

    def calculate_stop_loss(
        self, entry_price: float, position_size: float, is_long: bool = True
    ) -> float:
        """
        Return the stop loss price based on config.stop_loss_pct.
        test_stop_loss() calls this.
        """
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def process_trade_signal(self, signal: dict) -> bool:
        """
        Simple old interface: if trade signal has {timestamp, type, price, size}
        and price>0, size>0 => return True. Otherwise False.
        test_trade_signal_processing() calls this.
        """
        required_fields = ["timestamp", "type", "price", "size"]
        if not all(field in signal for field in required_fields):
            self.logger.warning("Invalid trade signal: missing required fields.")
            return False
        if signal["price"] <= 0 or signal["size"] <= 0:
            self.logger.warning("Invalid trade signal: negative price or zero size.")
            return False
        return True

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float] = None,
        current_positions: Optional[Dict[str, float]] = None,
        asset_name: Optional[str] = None
    ) -> float:
        """
        Return a NOTIONAL position size in dollars (not units).
        test_position_sizing() expects "portfolio_value * max_position_size" as a baseline.
        Also scale down if volatility is high or if correlation is too high.
        """
        # Basic position size limit in nominal terms
        position_size = portfolio_value * self.config.max_position_size

        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            vol_scalar = 1.0 / (1.0 + volatility)
            position_size *= vol_scalar

        # If we have correlation data and an asset_name & positions
        if current_positions and asset_name and self._correlation_matrix is not None:
            # If correlation is too high, reduce the size. (Simplistic approach)
            for pos_asset, pos_value in current_positions.items():
                if not self.check_correlation_limits(asset_name, pos_asset):
                    # E.g. reduce by half
                    position_size *= 0.5

        # Enforce minimum size (in notional)
        min_notional = portfolio_value * self.config.min_trade_size
        if position_size < min_notional:
            return 0.0

        return position_size

    def check_trade(
        self,
        timestamp: pd.Timestamp,
        portfolio_value: float,
        trade_size: float,
        price: float,
        positions: Dict[str, Dict[str, float]],
        asset: str,
        price_data: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Revised check_trade with a new order:
          1) daily limit
          2) drawdown
          3) position size clamp
          4) leverage clamp
          5) min trade
          6) advanced VaR/correlation
        """
        date_key = timestamp.date()
        
        # 1) daily limit - keep original reason
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        if self.trade_counter[date_key] >= self.config.daily_trade_limit:
            self.logger.warning(f"Daily trade limit reached for {date_key}")
            return {
                "allowed": False,
                "adjusted_size": 0.0,
                "reason": "Daily trade limit"
            }

        # 2) drawdown
        if self.peak_value is not None:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(
                    f"Max drawdown exceeded: {current_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
                )
                return {
                    "allowed": False,
                    "adjusted_size": 0.0,
                    "reason": f"Max drawdown exceeded: {current_drawdown:.2%}"
                }

        # Calculate current_exposure & proposed_value
        current_exposure = 0.0
        for sym, pos_dict in positions.items():
            if sym == asset:
                current_exposure += abs(pos_dict["units"] * price)
        proposed_value = abs(trade_size * price)

        # 5) Check minimum trade size first - if too small, reject immediately
        min_value = self.config.min_trade_size * portfolio_value
        if proposed_value < min_value:
            return {
                "allowed": False,
                "adjusted_size": 0.0,
                "reason": "trade_size_too_small"
            }

        # 3) position size clamp
        max_pos_value = self.config.max_position_size * portfolio_value
        if proposed_value > max_pos_value:
            # clamp but don't reject - this should succeed with adjusted size
            adjusted_units = max_pos_value / price
            # keep sign
            trade_size = adjusted_units if trade_size > 0 else -adjusted_units
            proposed_value = abs(trade_size * price)
            # Continue with adjusted size - don't return yet

        # 4) leverage clamp
        if portfolio_value > 1e-8:
            total_exposure = current_exposure + proposed_value
            leverage = total_exposure / portfolio_value
            if leverage > self.config.max_leverage:
                max_allowed_exposure = self.config.max_leverage * portfolio_value
                remain_exposure = max_allowed_exposure - current_exposure
                if remain_exposure < 0:
                    return {
                        "allowed": False,
                        "adjusted_size": 0.0,
                        "reason": "trade_size_too_small"
                    }
                # clamp but allow if size is still meaningful
                adjusted_units = remain_exposure / price
                if adjusted_units < self.config.min_trade_size * portfolio_value / price:
                    return {
                        "allowed": False,
                        "adjusted_size": 0.0,
                        "reason": "trade_size_too_small"
                    }
                # Update trade_size & proposed_value
                trade_size = adjusted_units if trade_size > 0 else -adjusted_units
                proposed_value = abs(trade_size * price)

        # 6) advanced VaR / correlation
        if price_data is not None:
            self.update_correlation_matrix(price_data)
            # hypothetical positions
            hypothetical_positions = {}
            for sym, pos_dict in positions.items():
                hypothetical_positions[sym] = pos_dict["units"]
            if asset not in hypothetical_positions:
                hypothetical_positions[asset] = 0.0
            hypothetical_positions[asset] += trade_size

            portfolio_var = self.get_portfolio_var(hypothetical_positions, portfolio_value)
            if portfolio_var > self.config.portfolio_var_limit:
                return {
                    "allowed": False,
                    "adjusted_size": 0.0,
                    "reason": "trade_size_too_small"
                }

        # Final zero-amount check after all clamps
        if abs(trade_size) < 1e-8:
            return {
                "allowed": False,
                "adjusted_size": 0.0,
                "reason": "trade_size_too_small"
            }

        # Trade is allowed with possibly adjusted size
        return {
            "allowed": True,
            "adjusted_size": trade_size,
            "reason": "OK"
        }

    def update_after_trade(self, timestamp: pd.Timestamp) -> None:
        """Call after a trade is executed successfully to update daily count, etc."""
        date_key = timestamp.date()
        if date_key not in self.trade_counter:
            self.trade_counter[date_key] = 0
        self.trade_counter[date_key] += 1
        self.logger.debug(f"Trade counter for {date_key}: {self.trade_counter[date_key]}")

    def update_drawdown(self, portfolio_value: float) -> float:
        """Update internal peak_value and current_drawdown, return current drawdown."""
        if self.peak_value is None:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        elif portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if self.current_drawdown > self.config.max_drawdown_pct:
                self.logger.warning(
                    f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
                )
        return self.current_drawdown

    # =====================
    # ADVANCED METHODS
    # =====================

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Value at Risk from historical returns at given confidence level.
        """
        if len(returns) < 2:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        """
        var = self.calculate_var(returns, confidence_level)
        if len(returns) < 2 or var <= 1e-12:
            return 0.0
        tail = returns[returns <= -var]
        if len(tail) == 0:
            return 0.0
        return abs(tail.mean())

    def update_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> None:
        """
        Recompute correlation matrix from the provided price_data.
        """
        if not price_data:
            return
        
        # Build returns
        for asset, prices in price_data.items():
            self._asset_returns[asset] = prices.pct_change().dropna()

        # Combine into dataframe w/ same window
        returns_df = {}
        for asset, rets in self._asset_returns.items():
            windowed = rets.tail(self.config.correlation_window)
            if len(windowed) > 1:
                returns_df[asset] = windowed
        if not returns_df:
            return
        
        df = pd.DataFrame(returns_df)
        if df.shape[1] < 2:
            return  # only 1 asset => no correlation to compute

        self._correlation_matrix = df.corr()
        self._last_correlation_update = pd.Timestamp.now()

    def check_correlation_limits(self, asset1: str, asset2: str) -> bool:
        """
        True if correlation between asset1 & asset2 is <= max_correlation
        Test expects a bool type (not numpy.bool_).
        """
        if self._correlation_matrix is None:
            return True
        if (asset1 not in self._correlation_matrix.columns
            or asset2 not in self._correlation_matrix.columns):
            return True

        corr = abs(self._correlation_matrix.loc[asset1, asset2])
        return bool(corr <= self.config.max_correlation)

    def get_portfolio_var(
        self,
        positions: Dict[str, float],
        portfolio_value: float
    ) -> float:
        """
        Approximate portfolio VaR by combining individual asset VaRs
        with correlation (if available).
        """
        if not positions or portfolio_value <= 0:
            return 0.0

        # 1) Compute weights (naive approach)
        weights = {}
        for asset, units in positions.items():
            position_value = abs(units) * 1.0
            weights[asset] = position_value / portfolio_value

        # 2) gather each asset's returns -> compute VaR
        asset_vars = []
        assets_in_matrix = []
        for asset, w in weights.items():
            if asset in self._asset_returns and w > 0:
                var = self.calculate_var(
                    self._asset_returns[asset],
                    self.config.var_confidence_level
                )
                asset_vars.append(var)
                assets_in_matrix.append(asset)
            else:
                asset_vars.append(0.0)
                assets_in_matrix.append(asset)

        if all(v == 0.0 for v in asset_vars):
            return 0.0

        var_diag = np.diag(asset_vars)
        weight_array = np.array([weights[a] for a in assets_in_matrix])
        
        # correlation
        if self._correlation_matrix is not None and len(assets_in_matrix) > 1:
            subset_corr = self._correlation_matrix.reindex(
                index=assets_in_matrix,
                columns=assets_in_matrix
            ).fillna(0.0).values
            portfolio_var = np.sqrt(weight_array.dot(var_diag).dot(subset_corr).dot(weight_array.T))
        else:
            portfolio_var = np.sqrt(weight_array.dot(var_diag).dot(weight_array.T))

        return min(float(portfolio_var), 1.0) 