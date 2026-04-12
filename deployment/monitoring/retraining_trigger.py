"""Retraining trigger: emits events when model retraining is warranted.

Two conditions are monitored independently (S58):

Condition A — drawdown breach
    Current drawdown_pct exceeds ``drawdown_trigger_pct``.

Condition B — feature/reward drift accumulation
    Cumulative drift alarm count (from DriftDetector or
    FeatureDriftDetector) reaches ``drift_alarm_trigger_count``.

Each fired event is:
  1. Written to the AuditLogger (if attached) as type ``retraining_trigger``.
  2. Passed to an optional ``on_trigger`` callback for external notification.

Actual model retraining is NOT performed here — this class only signals
that it should happen.  The operator is expected to respond manually or
via an external scheduler.

Usage
-----
    trigger = RetrainingTrigger(config={
        "drawdown_trigger_pct": 0.15,
        "drift_alarm_trigger_count": 5,
    }, audit_logger=audit_logger)

    # Inside the trading loop:
    event = trigger.check(
        drawdown_pct=current_dd,
        drift_count=drift_detector.n_detections,
    )
    if event:
        logger.warning("Retraining suggested: %s", event)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrainingEvent:
    """Describes a single retraining trigger event."""
    condition: str          # "drawdown" | "drift"
    value: float            # the metric value that crossed the threshold
    threshold: float        # the configured threshold
    timestamp: float = field(default_factory=time.time)
    step: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "step": self.step,
            **self.extra,
        }

    def __str__(self) -> str:
        return (
            f"RetrainingEvent(condition={self.condition!r}, "
            f"value={self.value:.4f}, threshold={self.threshold:.4f}, "
            f"step={self.step})"
        )


class RetrainingTrigger:
    """Monitor trading metrics and emit retraining events when warranted.

    Parameters
    ----------
    config : dict
        ``drawdown_trigger_pct``   — drawdown fraction that triggers (default 0.15).
        ``drift_alarm_trigger_count`` — cumulative alarm count that triggers (default 5).
        ``cooldown_steps``         — min steps between successive triggers of the
                                     same condition (default 100).
    audit_logger : optional
        AuditLogger instance for writing ``retraining_trigger`` records.
    on_trigger : optional
        Callable invoked with ``(RetrainingEvent,)`` when a trigger fires.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        audit_logger=None,
        on_trigger: Optional[Callable[[RetrainingEvent], None]] = None,
    ) -> None:
        cfg = config or {}
        self.drawdown_trigger_pct: float = float(
            cfg.get("drawdown_trigger_pct", 0.15)
        )
        self.drift_alarm_trigger_count: int = int(
            cfg.get("drift_alarm_trigger_count", 5)
        )
        self.cooldown_steps: int = int(cfg.get("cooldown_steps", 100))

        self.audit_logger = audit_logger
        self.on_trigger = on_trigger

        self._lock = threading.Lock()
        self._events: List[RetrainingEvent] = []
        # Track last-trigger step per condition for cooldown
        self._last_trigger_step: Dict[str, int] = {
            "drawdown": -self.cooldown_steps - 1,
            "drift": -self.cooldown_steps - 1,
        }

    # ------------------------------------------------------------------ #
    # Core check
    # ------------------------------------------------------------------ #

    def check(
        self,
        drawdown_pct: float,
        drift_count: int,
        step: int = -1,
    ) -> Optional[RetrainingEvent]:
        """Evaluate conditions and return a RetrainingEvent if one fires.

        Only one event is returned per call (drawdown takes priority over
        drift).  Cooldown is enforced per condition independently.

        Parameters
        ----------
        drawdown_pct : float
            Current drawdown as a fraction (e.g. 0.18 for 18 %).
        drift_count : int
            Cumulative drift alarm count from the attached drift detector.
        step : int
            Current trading step (used for cooldown tracking).

        Returns
        -------
        RetrainingEvent or None
        """
        event = self._check_drawdown(drawdown_pct, step)
        if event is None:
            event = self._check_drift(drift_count, step)
        if event is not None:
            self._fire(event)
        return event

    # ------------------------------------------------------------------ #
    # Individual condition checks
    # ------------------------------------------------------------------ #

    def _check_drawdown(self, drawdown_pct: float, step: int) -> Optional[RetrainingEvent]:
        if drawdown_pct < self.drawdown_trigger_pct:
            return None
        with self._lock:
            last = self._last_trigger_step["drawdown"]
            if step >= 0 and (step - last) < self.cooldown_steps:
                return None
            self._last_trigger_step["drawdown"] = step
        return RetrainingEvent(
            condition="drawdown",
            value=drawdown_pct,
            threshold=self.drawdown_trigger_pct,
            step=step,
        )

    def _check_drift(self, drift_count: int, step: int) -> Optional[RetrainingEvent]:
        if drift_count < self.drift_alarm_trigger_count:
            return None
        with self._lock:
            last = self._last_trigger_step["drift"]
            if step >= 0 and (step - last) < self.cooldown_steps:
                return None
            self._last_trigger_step["drift"] = step
        return RetrainingEvent(
            condition="drift",
            value=float(drift_count),
            threshold=float(self.drift_alarm_trigger_count),
            step=step,
        )

    # ------------------------------------------------------------------ #
    # Event dispatch
    # ------------------------------------------------------------------ #

    def _fire(self, event: RetrainingEvent) -> None:
        logger.warning(
            "Retraining trigger fired: condition=%s value=%.4f threshold=%.4f step=%d",
            event.condition,
            event.value,
            event.threshold,
            event.step,
        )
        with self._lock:
            self._events.append(event)

        if self.audit_logger is not None:
            try:
                self.audit_logger.log_risk_event({
                    "type": "retraining_trigger",
                    **event.to_dict(),
                })
            except Exception as exc:
                logger.warning("AuditLogger write failed for retraining event: %s", exc)

        if self.on_trigger is not None:
            try:
                self.on_trigger(event)
            except Exception as exc:
                logger.warning("on_trigger callback raised: %s", exc)

    # ------------------------------------------------------------------ #
    # History access
    # ------------------------------------------------------------------ #

    @property
    def events(self) -> List[RetrainingEvent]:
        """All events fired since creation (copy)."""
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        """Clear event history and cooldown counters."""
        with self._lock:
            self._events.clear()
            for key in self._last_trigger_step:
                self._last_trigger_step[key] = -self.cooldown_steps - 1
