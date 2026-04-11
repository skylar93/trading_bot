"""
RateLimiter — Week 64, Track C (S45)

Token-bucket rate limiter for exchange API calls.
Extracted from order_manager.py into its own module so it can be
tested and composed independently.
"""

from __future__ import annotations

import threading
import time
from typing import List


class RateLimiter:
    """
    Token-bucket rate limiter (thread-safe).

    Allows at most *max_calls* calls within any sliding *period*-second
    window.  Callers block in ``acquire()`` until a token is available.

    Parameters
    ----------
    max_calls : int
        Maximum calls allowed per *period*.
    period : float
        Length of the sliding time window in seconds.
    """

    def __init__(self, max_calls: int = 10, period: float = 1.0) -> None:
        if max_calls < 1:
            raise ValueError(f"max_calls must be >= 1, got {max_calls}")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self._max_calls = max_calls
        self._period = period
        self._calls: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a call token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return
            time.sleep(0.05)

    @property
    def max_calls(self) -> int:
        return self._max_calls

    @property
    def period(self) -> float:
        return self._period
