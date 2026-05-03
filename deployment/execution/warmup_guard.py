"""
WarmupGuard — E2/E3: cold-start and live-ramp protection.

E2 (paper/sandbox): size_fraction=0.5, no progress alerts
E3 (live):          size_fraction=0.3, 1-minute progress alerts

Both modes enforce max_qps=1 and fire a start alert and an end alert.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class WarmupGuard:
    """
    Caps order size and rate-limits submissions during the warmup window.

    Parameters
    ----------
    warmup_minutes :
        Length of the warmup window.
    size_fraction :
        Order size multiplier during warmup (0.5 for E2, 0.3 for E3).
    max_qps :
        Maximum orders allowed per second during warmup.
    progress_alerts :
        Emit a 1-minute progress alert to the alerter (True for E3 live ramp).
    alerter :
        Optional TradingAlerter for start / end / progress notifications.
    """

    def __init__(
        self,
        warmup_minutes: int = 30,
        size_fraction: float = 0.5,
        max_qps: float = 1.0,
        progress_alerts: bool = False,
        alerter=None,
    ) -> None:
        if warmup_minutes < 1:
            raise ValueError(f"warmup_minutes must be >= 1, got {warmup_minutes}")
        if not (0.0 < size_fraction <= 1.0):
            raise ValueError(f"size_fraction must be in (0, 1], got {size_fraction}")
        if max_qps <= 0:
            raise ValueError(f"max_qps must be > 0, got {max_qps}")

        self.warmup_minutes = warmup_minutes
        self.warmup_seconds = warmup_minutes * 60.0
        self.size_fraction = size_fraction
        self._min_order_interval: float = 1.0 / max_qps
        self.progress_alerts = progress_alerts
        self.alerter = alerter

        self._start_time: Optional[float] = None
        self._ended: bool = False
        self._last_order_time: float = 0.0
        self._last_progress_min: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Call once when the trading loop begins."""
        self._start_time = time.monotonic()
        msg = (
            f"Warmup mode ACTIVE — "
            f"size_fraction={self.size_fraction:.0%}, "
            f"max_qps={1.0/self._min_order_interval:.0f}, "
            f"duration={self.warmup_minutes} min"
        )
        logger.warning(msg)
        if self.alerter is not None:
            self.alerter.send_alert(msg, level="WARNING")

    @property
    def in_warmup(self) -> bool:
        """True while the warmup window is active."""
        if self._ended or self._start_time is None:
            return False
        return (time.monotonic() - self._start_time) < self.warmup_seconds

    def check(self, requested_size: float) -> Tuple[bool, float]:
        """
        Decide whether an order may proceed and return the allowed size.

        Should be called before every order submission.

        Parameters
        ----------
        requested_size :
            The fractional order size the agent requested (0–1).

        Returns
        -------
        (allowed, capped_size)
            allowed    — False when QPS limit is breached; skip the order.
            capped_size — requested_size * size_fraction while in warmup,
                         requested_size unchanged once warmup ends.
        """
        self._maybe_end()
        if not self.in_warmup:
            return True, requested_size

        self._maybe_progress_alert()

        now = time.monotonic()
        if (now - self._last_order_time) < self._min_order_interval:
            return False, 0.0

        self._last_order_time = now
        return True, requested_size * self.size_fraction

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_end(self) -> None:
        if self._ended or self._start_time is None:
            return
        if (time.monotonic() - self._start_time) >= self.warmup_seconds:
            self._ended = True
            msg = "Warmup ENDED — normal trading parameters restored"
            logger.info(msg)
            if self.alerter is not None:
                self.alerter.send_alert(msg, level="INFO")

    def _maybe_progress_alert(self) -> None:
        if not self.progress_alerts or self._start_time is None:
            return
        elapsed_min = int((time.monotonic() - self._start_time) / 60)
        if elapsed_min > self._last_progress_min and elapsed_min < self.warmup_minutes:
            self._last_progress_min = elapsed_min
            remaining = self.warmup_minutes - elapsed_min
            msg = (
                f"Live ramp progress: {elapsed_min}/{self.warmup_minutes} min elapsed, "
                f"{remaining} min remaining"
            )
            logger.info(msg)
            if self.alerter is not None:
                self.alerter.send_alert(msg, level="INFO")
