"""
ClockSync — Week 64, Track C (S45)

Detects clock skew between the local machine and the exchange server.

If drift exceeds *max_drift_sec*:
  - A warning is always logged.
  - If *halt_on_skew* is True, ``is_halted`` is set to True so the caller
    can prevent order submission until the drift resolves.

Design
------
- Accepts an optional ``time_fn: () -> float`` for unit-testing without a
  live exchange.  When provided, it is called instead of CCXT's
  ``fetch_time()``.
- Thread-safe; safe to call ``check()`` from multiple threads.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ClockSync:
    """
    Monitor clock drift between local wall clock and exchange server time.

    Parameters
    ----------
    max_drift_sec : float
        Maximum acceptable absolute drift in seconds before action is taken.
    halt_on_skew : bool
        If True, set ``is_halted = True`` when drift exceeds threshold.
        If False (default), only emit a warning log.
    time_fn : callable, optional
        ``() -> float`` returning remote server time as a Unix epoch float
        (seconds).  Injected for testing; overrides CCXT exchange lookup.
    """

    def __init__(
        self,
        max_drift_sec: float = 5.0,
        halt_on_skew: bool = False,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        if max_drift_sec <= 0:
            raise ValueError(f"max_drift_sec must be > 0, got {max_drift_sec}")
        self._max_drift = max_drift_sec
        self._halt_on_skew = halt_on_skew
        self._time_fn = time_fn
        self._exchange = None
        self._is_halted: bool = False
        self._last_drift: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_exchange(self, exchange) -> None:
        """Attach a CCXT exchange object for server-time queries."""
        with self._lock:
            self._exchange = exchange

    def check(self) -> float:
        """
        Measure and evaluate clock drift.

        Returns
        -------
        float
            Absolute drift in seconds.  Returns 0.0 if the server time
            could not be fetched.

        Side effects
        ------------
        - Logs a warning when drift > max_drift_sec.
        - Sets ``is_halted = True`` when halt_on_skew is True and drift
          exceeds threshold.
        """
        server_ts = self._fetch_server_time()
        if server_ts is None:
            return 0.0

        local_ts = time.time()
        drift = abs(local_ts - server_ts)

        with self._lock:
            self._last_drift = drift
            if drift > self._max_drift:
                logger.warning(
                    "ClockSync: drift=%.3fs exceeds max_drift=%.3fs "
                    "(local=%.3f server=%.3f)",
                    drift,
                    self._max_drift,
                    local_ts,
                    server_ts,
                )
                if self._halt_on_skew:
                    self._is_halted = True
                    logger.error(
                        "ClockSync: trading halted due to excessive clock skew (%.3fs)",
                        drift,
                    )
            else:
                logger.debug("ClockSync: drift=%.3fs OK", drift)

        return drift

    @property
    def is_halted(self) -> bool:
        """True if halt_on_skew is enabled and drift exceeded threshold."""
        with self._lock:
            return self._is_halted

    @property
    def last_drift(self) -> Optional[float]:
        """Most recently measured drift in seconds, or None if never checked."""
        with self._lock:
            return self._last_drift

    def reset_halt(self) -> None:
        """Manually clear the halt flag after resolving clock skew."""
        with self._lock:
            self._is_halted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_server_time(self) -> Optional[float]:
        """Return server time as Unix epoch seconds, or None on failure."""
        if self._time_fn is not None:
            try:
                return float(self._time_fn())
            except Exception as exc:
                logger.warning("ClockSync: time_fn failed: %s", exc)
                return None

        exchange = self._exchange
        if exchange is not None:
            try:
                # CCXT fetch_time() returns milliseconds
                ms = exchange.fetch_time()
                return ms / 1000.0
            except Exception as exc:
                logger.warning("ClockSync: exchange.fetch_time() failed: %s", exc)
                return None

        return None
