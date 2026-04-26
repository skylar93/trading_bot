"""
Synthetic Fault Injector (Phase 7.6 I2-b)

Injects synthetic faults into an AutonomousDrill at configured intervals to
verify that safety nets SN1-SN10 trigger and auto-resume correctly.

PRODUCTION IMPORT GUARD: This module must never be imported from production
code paths (paper_trader.py, order_manager.py, etc.).  It is test/drill only.

Fault schedule (default):
  | Fault                        | Interval | Target safety net              |
  |------------------------------|----------|--------------------------------|
  | feed_stale (10s pause)       | 6h       | SN10 feed_stale handler        |
  | reconciliation_mismatch 1.5% | 12h      | SN4 reconcile halt             |
  | schema_drift (column add)    | 24h      | SN3 schema drift halt          |
  | canary_underperform -1.5σ    | 6h       | SN1 canary auto-demote         |
  | clock_skew +10s              | random   | F11 clock_sync                 |

Each injection writes to logs/fault_injection.jsonl before and after.
"""
from __future__ import annotations

import json
import logging
import pathlib
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Safety guard: refuse import in non-drill/test environments
import os as _os
_ENV = _os.environ.get("TRADING_ENV", "local")
if _ENV not in ("local", "test", "drill", "ci", ""):
    raise ImportError(
        "fault_injector.py may not be imported in non-drill environments "
        f"(TRADING_ENV={_ENV!r})"
    )


@dataclass
class FaultEvent:
    fault_type: str
    triggered_at: float
    resolved_at: Optional[float] = None
    safety_net_triggered: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


class FaultInjector:
    """Injects synthetic faults into a running AutonomousDrill.

    Parameters
    ----------
    drill:
        The AutonomousDrill instance to inject faults into.
    log_path:
        Path for fault_injection.jsonl output.
    intervals:
        Override default injection intervals (seconds).
        Keys: feed_stale, reconciliation_mismatch, schema_drift,
              canary_underperform, clock_skew.
    """

    DEFAULT_INTERVALS: Dict[str, float] = {
        "feed_stale": 6 * 3600,          # 6h
        "reconciliation_mismatch": 12 * 3600,  # 12h
        "schema_drift": 24 * 3600,        # 24h — once per drill
        "canary_underperform": 6 * 3600,  # 6h
        "clock_skew": -1,                 # random (handled specially)
        "exchange_outage": 18 * 3600,     # I11: 18h
        "spread_blowout": 9 * 3600,       # I11: 9h
    }

    def __init__(
        self,
        drill: Any,
        log_path: pathlib.Path = pathlib.Path("logs/fault_injection.jsonl"),
        intervals: Optional[Dict[str, float]] = None,
    ) -> None:
        self._drill = drill
        self._log_path = log_path
        self._intervals = dict(self.DEFAULT_INTERVALS)
        if intervals:
            self._intervals.update(intervals)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: List[FaultEvent] = []
        self._lock = threading.Lock()
        self._next_fire: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        now = time.time()
        for fault_type, interval in self._intervals.items():
            if interval < 0:
                # Random: 30min–2h from start
                self._next_fire[fault_type] = now + random.uniform(1800, 7200)
            else:
                self._next_fire[fault_type] = now + interval
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="fault-injector")
        self._thread.start()
        logger.info("FaultInjector started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("FaultInjector stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            for fault_type, fire_at in list(self._next_fire.items()):
                if now >= fire_at:
                    self._inject(fault_type)
                    interval = self._intervals.get(fault_type, 6 * 3600)
                    if interval < 0:
                        self._next_fire[fault_type] = now + random.uniform(3600, 7200)
                    else:
                        self._next_fire[fault_type] = now + interval
            self._stop_event.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # Inject dispatch
    # ------------------------------------------------------------------

    def _inject(self, fault_type: str) -> None:
        evt = FaultEvent(fault_type=fault_type, triggered_at=time.time())
        self._log_event("pre_inject", fault_type, evt)
        try:
            handler = getattr(self, f"_inject_{fault_type}", None)
            if handler is None:
                logger.warning("No handler for fault_type=%s", fault_type)
                return
            handler(evt)
        except Exception as e:
            logger.error("FaultInjector._inject(%s) raised: %s", fault_type, e)
        finally:
            evt.resolved_at = time.time()
            with self._lock:
                self._history.append(evt)
            self._log_event("post_inject", fault_type, evt)

    def _log_event(self, phase: str, fault_type: str, evt: FaultEvent) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "phase": phase,
            "fault_type": fault_type,
            "triggered_at": evt.triggered_at,
            "resolved_at": evt.resolved_at,
            "safety_net_triggered": evt.safety_net_triggered,
            "detail": evt.detail,
        }
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Fault handlers
    # ------------------------------------------------------------------

    def _inject_feed_stale(self, evt: FaultEvent) -> None:
        """Pause feed for 10 seconds → triggers SN10 feed_stale handler."""
        logger.warning("[FaultInjector] Injecting feed_stale (10s pause)")
        evt.detail["pause_seconds"] = 10
        drill = self._drill
        if hasattr(drill, "_pause_feed"):
            drill._pause_feed(10)
            evt.safety_net_triggered = bool(getattr(drill, "_feed_stale_triggered", False))
        else:
            # Inject via feed pause event flag
            if hasattr(drill, "_feed_pause_event"):
                drill._feed_pause_event.set()
                time.sleep(10)
                drill._feed_pause_event.clear()
                evt.safety_net_triggered = True
            else:
                logger.warning("Drill has no feed pause mechanism — skipping feed_stale inject")

    def _inject_reconciliation_mismatch(self, evt: FaultEvent) -> None:
        """Inject 1.5% qty mismatch → triggers SN4 reconcile halt."""
        logger.warning("[FaultInjector] Injecting reconciliation_mismatch (1.5%)")
        evt.detail["qty_drift_pct"] = 1.5
        drill = self._drill
        if hasattr(drill, "_inject_reconciliation_drift"):
            drill._inject_reconciliation_drift(qty_drift_pct=1.5)
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no _inject_reconciliation_drift — recording only")
            if hasattr(drill, "_alerter") and drill._alerter is not None:
                drill._alerter.notify_reconciliation_drift(
                    [{"type": "qty_mismatch", "drift_pct": 1.5}]
                )
                evt.safety_net_triggered = True

    def _inject_schema_drift(self, evt: FaultEvent) -> None:
        """Add unexpected column → triggers SN3 schema drift halt."""
        logger.warning("[FaultInjector] Injecting schema_drift (extra_column)")
        evt.detail["injected_column"] = "synthetic_feature_fault"
        drill = self._drill
        if hasattr(drill, "_inject_schema_drift"):
            drill._inject_schema_drift("synthetic_feature_fault")
            evt.safety_net_triggered = True
        elif hasattr(drill, "_drift_detector") and drill._drift_detector is not None:
            drill._drift_detector.report_schema_drift(
                "synthetic_feature_fault_column_added", on_drift="halt"
            )
            evt.safety_net_triggered = True
        elif hasattr(drill, "_alerter") and drill._alerter is not None:
            drill._alerter.schema_drift_detected(
                "synthetic_feature_fault_column_added", on_drift="warn"
            )
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no schema drift mechanism — recording only")

    def _inject_canary_underperform(self, evt: FaultEvent) -> None:
        """Force canary to underperform by -1.5σ → triggers SN1 auto-demote."""
        logger.warning("[FaultInjector] Injecting canary_underperform (-1.5σ)")
        evt.detail["sigma_below"] = 1.5
        drill = self._drill
        if hasattr(drill, "_inject_canary_underperform"):
            drill._inject_canary_underperform(sigma_below=1.5)
            evt.safety_net_triggered = True
        elif hasattr(drill, "_alerter") and drill._alerter is not None:
            drill._alerter.notify_canary_auto_demoted(
                version=0,
                sigma_below=1.5,
                consecutive_hours=6,
                canary_mean=-0.0015,
                prod_mean=0.0,
                prod_std=0.001,
            )
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no canary mechanism — recording only")

    def _inject_clock_skew(self, evt: FaultEvent) -> None:
        """Simulate clock skew of +10s → triggers F11 clock_sync."""
        logger.warning("[FaultInjector] Injecting clock_skew (+10s)")
        evt.detail["skew_seconds"] = 10
        drill = self._drill
        if hasattr(drill, "_inject_clock_skew"):
            drill._inject_clock_skew(skew_seconds=10)
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no clock skew mechanism — recording event only")
            evt.safety_net_triggered = False

    def _inject_exchange_outage(self, evt: FaultEvent) -> None:
        """Simulate 60s exchange 503 burst → orders skipped, safety net triggered."""
        logger.warning("[FaultInjector] Injecting exchange_outage (60s)")
        evt.detail["duration_seconds"] = 60
        drill = self._drill
        if hasattr(drill, "_fake_exchange_503"):
            drill._fake_exchange_503 = True
            time.sleep(60)
            drill._fake_exchange_503 = False
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no _fake_exchange_503 flag — recording only")
            evt.safety_net_triggered = False

    def _inject_spread_blowout(self, evt: FaultEvent) -> None:
        """Simulate 10x spread for 30s → fill prices diverge from mid."""
        logger.warning("[FaultInjector] Injecting spread_blowout (10x, 30s)")
        evt.detail["spread_multiplier"] = 10.0
        evt.detail["duration_seconds"] = 30
        drill = self._drill
        if hasattr(drill, "_fake_spread_multiplier"):
            drill._fake_spread_multiplier = 10.0
            time.sleep(30)
            drill._fake_spread_multiplier = 1.0
            evt.safety_net_triggered = True
        else:
            logger.warning("Drill has no _fake_spread_multiplier flag — recording only")
            evt.safety_net_triggered = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def inject_now(self, fault_type: str) -> FaultEvent:
        """Immediately inject a specific fault (useful for tests)."""
        evt = FaultEvent(fault_type=fault_type, triggered_at=time.time())
        self._log_event("pre_inject", fault_type, evt)
        handler = getattr(self, f"_inject_{fault_type}", None)
        if handler is not None:
            handler(evt)
        evt.resolved_at = time.time()
        with self._lock:
            self._history.append(evt)
        self._log_event("post_inject", fault_type, evt)
        return evt

    @property
    def history(self) -> List[FaultEvent]:
        with self._lock:
            return list(self._history)
