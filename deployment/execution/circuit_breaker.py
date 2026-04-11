"""
VolatilityCircuitBreaker — Week 64, Track C (S43)

Halts new order submission when realized rolling volatility exceeds a
threshold.  Existing positions are **not** liquidated; only new orders
are blocked.

After ``cooldown`` seconds, the breaker re-evaluates current vol and
auto-resets if vol has dropped back below the threshold.

Config keys (risk.circuit_breaker)
-----------------------------------
  enabled       : bool    (default True)
  vol_threshold : float   realized vol (std of returns) to trigger (e.g. 0.05)
  window        : int     rolling window size for vol calculation (prices)
  cooldown      : float   seconds to keep tripped before re-evaluation
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VolatilityCircuitBreaker:
    """
    Rolling realized-volatility circuit breaker.

    Parameters
    ----------
    vol_threshold : float
        Std-dev of period returns above which new orders are blocked.
    window : int
        Number of price observations used for vol calculation (need ``window``
        prices to get ``window - 1`` returns).
    cooldown : float
        Minimum seconds to remain tripped before the next re-evaluation.
    """

    def __init__(
        self,
        vol_threshold: float = 0.05,
        window: int = 20,
        cooldown: float = 300.0,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if vol_threshold <= 0:
            raise ValueError(f"vol_threshold must be > 0, got {vol_threshold}")
        self._vol_threshold = vol_threshold
        self._window = window
        self._cooldown = cooldown
        # Store window+1 prices so we can compute window returns
        self._prices: Deque[float] = deque(maxlen=window + 1)
        self._tripped: bool = False
        self._tripped_at: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, price: float) -> None:
        """Feed a new price tick.  Recalculates vol and updates breaker state."""
        with self._lock:
            self._prices.append(price)
            self._recalculate_locked()

    def is_tripped(self) -> bool:
        """
        Return True if the circuit breaker is active (new orders blocked).

        After the cooldown elapses, the vol is re-evaluated and the breaker
        auto-resets if vol has fallen below threshold.
        """
        with self._lock:
            if self._tripped and self._tripped_at is not None:
                elapsed = time.monotonic() - self._tripped_at
                if elapsed >= self._cooldown:
                    self._recalculate_locked()
            return self._tripped

    @property
    def current_vol(self) -> Optional[float]:
        """Current rolling realized volatility, or None if insufficient data."""
        with self._lock:
            return self._compute_vol_locked()

    @property
    def vol_threshold(self) -> float:
        return self._vol_threshold

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _recalculate_locked(self) -> None:
        vol = self._compute_vol_locked()
        if vol is None:
            return
        if vol > self._vol_threshold:
            if not self._tripped:
                self._tripped = True
                self._tripped_at = time.monotonic()
                logger.warning(
                    "VolatilityCircuitBreaker TRIPPED: vol=%.4f > threshold=%.4f",
                    vol,
                    self._vol_threshold,
                )
        else:
            if self._tripped:
                logger.info(
                    "VolatilityCircuitBreaker RESET: vol=%.4f <= threshold=%.4f",
                    vol,
                    self._vol_threshold,
                )
            self._tripped = False
            self._tripped_at = None

    def _compute_vol_locked(self) -> Optional[float]:
        prices = list(self._prices)
        if len(prices) < 2:
            return None
        arr = np.array(prices, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        if len(returns) < 2:
            return None
        return float(np.std(returns, ddof=1))
