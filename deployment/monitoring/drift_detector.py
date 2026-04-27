"""
Deployment-side drift coordinator with shadow mode (Phase 7.6 I4).

During shadow_mode_hours after start:
  - Drift detected → WARNING alert only; halt_requested remains False.
After shadow period:
  - Drift detected → CRITICAL alert; halt_requested set to True.

This wraps drift-detection alerting so PaperTrader / AutonomousDrill can
query halt_requested without coupling to the training-side DriftDetector.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from deployment.monitoring.alerter import TradingAlerter

logger = logging.getLogger(__name__)

_SENTINEL_SHADOW_HOURS = 72  # default from I4 spec


class DeploymentDriftDetector:
    """Shadow-mode-aware deployment drift coordinator.

    Parameters
    ----------
    config:
        Dict containing a ``drift`` sub-dict (from config/alerts.yaml).
    alerter:
        Optional TradingAlerter to dispatch alert messages through.
    _start_time:
        Override start time (float, epoch seconds) — useful for testing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        alerter: Optional["TradingAlerter"] = None,
        _start_time: Optional[float] = None,
    ) -> None:
        drift_cfg: Dict[str, Any] = config.get("drift", {})
        shadow_hours = float(drift_cfg.get("shadow_mode_hours", _SENTINEL_SHADOW_HOURS))

        start = _start_time if _start_time is not None else time.time()
        self._shadow_mode_until: float = start + shadow_hours * 3600

        self.alerter = alerter
        self.halt_requested: bool = False
        self._n_detections: int = 0
        self._shadow_end_notified: bool = False

        # Thresholds (informational; callers may read for custom logic)
        self.reward_return_sigma_threshold: float = float(
            drift_cfg.get("reward_return_sigma_threshold", 2.0)
        )
        self.feature_psi_threshold: float = float(drift_cfg.get("feature_psi_threshold", 0.2))
        self.pnl_z_threshold: float = float(drift_cfg.get("pnl_z_threshold", 3.0))
        self.action_entropy_min: float = float(drift_cfg.get("action_entropy_min", 0.5))

        logger.info(
            "DeploymentDriftDetector init | shadow_hours=%.1f until=%.0f",
            shadow_hours,
            self._shadow_mode_until,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def in_shadow_mode(self) -> bool:
        """True while within the initial shadow observation window."""
        in_shadow = time.time() < self._shadow_mode_until
        if not in_shadow and not self._shadow_end_notified:
            self._shadow_end_notified = True
            self._on_shadow_end()
        return in_shadow

    def _on_shadow_end(self) -> None:
        """Called exactly once when shadow mode transitions to active."""
        logger.warning("Shadow mode ended — drift halt now active")
        if self.alerter is not None:
            self.alerter.send_alert(
                "Shadow mode ended — drift halt now active. "
                "Review drift_calibration_*.md before continuing.",
                level="CRITICAL",
            )

    @property
    def n_detections(self) -> int:
        return self._n_detections

    # ------------------------------------------------------------------
    # Report methods
    # ------------------------------------------------------------------

    def report_drift(
        self,
        detector: str,
        signal_name: str,
        details: Optional[str] = None,
    ) -> None:
        """Report a statistical drift event.

        During shadow mode: WARNING-level alert only, halt suppressed.
        After shadow:       CRITICAL-level alert + halt_requested = True.
        """
        self._n_detections += 1
        if self.in_shadow_mode:
            shadow_note = f"[shadow mode — no halt] {details or ''}".rstrip()
            logger.warning(
                "Drift (shadow): detector=%s signal=%s %s", detector, signal_name, shadow_note
            )
            if self.alerter is not None:
                self.alerter.notify_drift(
                    detector=detector,
                    signal_name=signal_name,
                    details=shadow_note,
                )
        else:
            logger.error("Drift halt: detector=%s signal=%s %s", detector, signal_name, details or "")
            self.halt_requested = True
            if self.alerter is not None:
                msg = f"Drift halt: {detector} detected on '{signal_name}'"
                if details:
                    msg += f" — {details}"
                self.alerter.send_alert(msg, level="CRITICAL")

    def report_schema_drift(self, drift_detail: str, on_drift: str = "halt") -> None:
        """Report schema drift.

        During shadow mode: policy downgraded to ``warn`` regardless of config.
        After shadow:       original ``on_drift`` policy applied.
        """
        self._n_detections += 1
        effective_on_drift = "warn" if self.in_shadow_mode else on_drift
        if self.alerter is not None:
            self.alerter.schema_drift_detected(drift_detail, on_drift=effective_on_drift)
        if effective_on_drift == "halt":
            self.halt_requested = True

    def reset_halt(self, *, source: Literal["auto_drill", "operator"] = "operator") -> None:
        """Clear halt flag.

        source='auto_drill' — automated 30s resume from drill (silent).
        source='operator'  — manual operator action; sends WARNING alert.
        """
        self.halt_requested = False
        logger.info("DeploymentDriftDetector: halt cleared (source=%s)", source)
        if self.alerter is not None and source == "operator":
            self.alerter.send_alert(
                "Drift halt manually cleared by operator", level="WARNING"
            )
