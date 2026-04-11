"""
FatFingerGuard — Week 64, Track C (S42)

Rejects abnormally large orders that could indicate typos or runaway
position sizing.

Rules
-----
1. Hard cap:        order size > hard_cap (absolute) → reject
2. Size multiplier: order size > mean(recent N orders) × multiplier_limit → reject
                    (only enforced once at least one historical fill exists)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Tuple

logger = logging.getLogger(__name__)


class FatFingerGuard:
    """
    Detect and reject abnormally large orders.

    Parameters
    ----------
    size_multiplier_limit : float
        Orders larger than ``mean(recent) × size_multiplier_limit`` are
        rejected.  Set to 0 to disable multiplier check.
    hard_cap : float
        Absolute maximum order size in base currency.  Set to 0 to disable.
    lookback : int
        Number of recent filled order sizes used for the multiplier baseline.
    """

    def __init__(
        self,
        size_multiplier_limit: float = 5.0,
        hard_cap: float = 0.0,
        lookback: int = 20,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self._multiplier = size_multiplier_limit
        self._hard_cap = hard_cap
        self._lookback = lookback
        self._history: Deque[float] = deque(maxlen=lookback)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, amount: float) -> Tuple[bool, str]:
        """
        Check whether *amount* is acceptable.

        Returns
        -------
        (ok, reason)
            ``ok == True``  → order passes.
            ``ok == False`` → order rejected; *reason* names the violated rule.
        """
        with self._lock:
            # 1. Hard cap
            if self._hard_cap > 0 and amount > self._hard_cap:
                reason = (
                    f"order size {amount:.6f} exceeds hard_cap {self._hard_cap:.6f}"
                )
                logger.warning("FatFingerGuard REJECT: %s", reason)
                return False, reason

            # 2. Multiplier check (only when history is populated)
            if self._multiplier > 0 and len(self._history) > 0:
                avg = sum(self._history) / len(self._history)
                limit = avg * self._multiplier
                if avg > 0 and amount > limit:
                    reason = (
                        f"order size {amount:.6f} exceeds "
                        f"avg {avg:.6f} × {self._multiplier} = {limit:.6f}"
                    )
                    logger.warning("FatFingerGuard REJECT: %s", reason)
                    return False, reason

            return True, ""

    def record_fill(self, amount: float) -> None:
        """Record a successfully filled order size for future multiplier checks."""
        if amount > 0:
            with self._lock:
                self._history.append(amount)

    @property
    def history_size(self) -> int:
        """Number of fills recorded so far (up to *lookback*)."""
        with self._lock:
            return len(self._history)
