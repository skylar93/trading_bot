"""
UnifiedRiskManager — Week 60, Track B

Single source of truth for shared risk computations used by both
BacktestingRiskManager and RLRiskManager.

Design principles:
- mode: "backtest" | "live" — controls behavioral defaults, not math
- var_method: "parametric" | "historical" — independent of mode
- All public methods are pure computations (explicit inputs → deterministic output)
- Thread-safe via threading.RLock (reentrant-safe for composing classes)
- Existing managers keep their own state; they delegate math to this class
"""

import threading
import numpy as np
import pandas as pd
from typing import Literal, Optional
from scipy.stats import norm


class UnifiedRiskManager:
    """
    Shared risk computation engine.

    Provides stateless (or near-stateless) implementations of common risk
    checks so that BacktestingRiskManager and RLRiskManager can delegate to a
    single, tested implementation instead of duplicating logic.

    Thread safety:
        All public methods acquire ``self._lock`` (a reentrant ``RLock``) so
        that composing classes that already hold their own lock can call these
        methods without deadlock.

    Parameters
    ----------
    mode : {"backtest", "live"}
        Execution context.  Does not change VaR math; affects only which
        defaults are applied when optional arguments are omitted.
    var_method : {"parametric", "historical"}
        VaR calculation strategy, independent of ``mode``.
    """

    def __init__(
        self,
        mode: Literal["backtest", "live"] = "backtest",
        var_method: Literal["parametric", "historical"] = "historical",
    ) -> None:
        self.mode = mode
        self.var_method = var_method
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # check_drawdown
    # ------------------------------------------------------------------
    def check_drawdown(
        self,
        peak_value: float,
        current_value: float,
        max_drawdown_pct: float,
    ) -> bool:
        """Return True if drawdown exceeds *max_drawdown_pct*.

        Parameters
        ----------
        peak_value : float
            Historical peak portfolio value (must be > 0).
        current_value : float
            Current portfolio value.
        max_drawdown_pct : float
            Maximum allowable drawdown as a fraction (e.g. 0.15 = 15%).

        Returns
        -------
        bool
            True  → drawdown limit breached.
            False → within limit, or peak_value <= 0.
        """
        with self._lock:
            if peak_value <= 0:
                return False
            drawdown = (peak_value - current_value) / peak_value
            return drawdown > max_drawdown_pct

    # ------------------------------------------------------------------
    # check_trailing_stop
    # ------------------------------------------------------------------
    def check_trailing_stop(
        self,
        current_price: float,
        reference_price: float,
        trailing_stop_buffer: float,
        is_long: bool = True,
    ) -> bool:
        """Return True if the trailing stop has been triggered.

        Parameters
        ----------
        current_price : float
            Latest market price.
        reference_price : float
            High-water mark (long) or low-water mark (short).
        trailing_stop_buffer : float
            Allowed adverse move as a fraction (e.g. 0.05 = 5%).
        is_long : bool
            True for long positions, False for short.

        Returns
        -------
        bool
            True  → stop triggered.
            False → price still within buffer.
        """
        with self._lock:
            if reference_price <= 0:
                return False
            if is_long:
                adverse_move = (reference_price - current_price) / reference_price
            else:
                adverse_move = (current_price - reference_price) / reference_price
            return adverse_move > trailing_stop_buffer

    # ------------------------------------------------------------------
    # compute_var
    # ------------------------------------------------------------------
    def compute_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        var_method: Optional[Literal["parametric", "historical"]] = None,
    ) -> Optional[float]:
        """Compute Value at Risk.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns (e.g. daily P&L / equity).  Requires at
            least 10 observations; returns ``None`` otherwise.
        confidence_level : float
            VaR confidence level (e.g. 0.95 → 95% VaR).
        var_method : {"parametric", "historical"} | None
            Override instance-level ``var_method``; uses instance default when
            None.

        Returns
        -------
        Optional[float]
            Non-negative VaR estimate, or ``None`` if insufficient data.

        Notes
        -----
        Parametric VaR
            ``VaR = -(μ + z_α · σ)``  where ``z_α = norm.ppf(1 - CL) < 0``.
            Equivalent to ``(z_α · σ - μ)`` expressed as a positive loss.
        Historical VaR
            ``VaR = -percentile(returns, (1 - CL) × 100)``.
            Left-tail quantile negated → positive loss amount.
        """
        with self._lock:
            arr = np.asarray(returns, dtype=float)
            if len(arr) < 10:
                return None

            method = var_method if var_method is not None else self.var_method

            if method == "parametric":
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                # norm.ppf(1 - CL) is negative; multiplying by std gives negative offset
                z_alpha = norm.ppf(1 - confidence_level)
                var = -(mean + z_alpha * std)
                return max(0.0, var)
            else:
                # historical
                var = -float(np.percentile(arr, (1 - confidence_level) * 100))
                return max(0.0, var)

    # ------------------------------------------------------------------
    # check_correlation
    # ------------------------------------------------------------------
    def check_correlation(
        self,
        correlation_value: float,
        threshold: float,
    ) -> bool:
        """Return True if absolute correlation exceeds *threshold*.

        Parameters
        ----------
        correlation_value : float
            Pearson correlation between two assets (range [-1, 1]).
        threshold : float
            Limit above which correlation is considered high-risk.

        Returns
        -------
        bool
            True  → risk exceeded (correlation is too high).
            False → within acceptable range.
        """
        with self._lock:
            return abs(correlation_value) > threshold

    # ------------------------------------------------------------------
    # check_position_limit
    # ------------------------------------------------------------------
    def check_position_limit(
        self,
        position_value: float,
        portfolio_value: float,
        max_position_fraction: float,
    ) -> bool:
        """Return True if position is within the allowed fraction of portfolio.

        Parameters
        ----------
        position_value : float
            Absolute notional value of the position (|units × price|).
        portfolio_value : float
            Total portfolio value (must be > 0).
        max_position_fraction : float
            Maximum allowed fraction (e.g. 0.2 = 20%).

        Returns
        -------
        bool
            True  → position is within limit.
            False → position exceeds limit, or portfolio_value <= 0.
        """
        with self._lock:
            if portfolio_value <= 0:
                return False
            return (position_value / portfolio_value) <= max_position_fraction
