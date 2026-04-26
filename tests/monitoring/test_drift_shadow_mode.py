"""I4: DeploymentDriftDetector shadow mode — halt suppressed during window, enabled after."""
from __future__ import annotations

import time

import pytest

from deployment.monitoring.drift_detector import DeploymentDriftDetector


# ---------------------------------------------------------------------------
# Shadow mode active tests
# ---------------------------------------------------------------------------

def test_shadow_mode_halt_not_requested_during_window() -> None:
    """During shadow mode, report_drift must NOT set halt_requested."""
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 72}},
        _start_time=time.time(),
    )
    assert detector.in_shadow_mode
    detector.report_drift(detector="adwin", signal_name="portfolio_return")
    assert not detector.halt_requested


def test_shadow_mode_schema_drift_no_halt() -> None:
    """During shadow mode, schema drift with on_drift='halt' must NOT set halt."""
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 72}},
        _start_time=time.time(),
    )
    detector.report_schema_drift("extra_column_added", on_drift="halt")
    assert not detector.halt_requested


def test_shadow_mode_n_detections_increments() -> None:
    detector = DeploymentDriftDetector({"drift": {"shadow_mode_hours": 72}})
    assert detector.n_detections == 0
    detector.report_drift("adwin", "return")
    detector.report_drift("page_hinkley", "feature_x")
    assert detector.n_detections == 2


def test_shadow_mode_alerter_called_with_warning(tmp_path) -> None:
    """During shadow mode the alerter should receive notify_drift (level=WARNING)."""
    from deployment.monitoring.alerter import TradingAlerter

    alerter = TradingAlerter({"alert_channels": ["file"], "log_dir": str(tmp_path)})
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 72}},
        alerter=alerter,
    )
    detector.report_drift("adwin", "return", details="test")
    # alert_history should have one WARNING event
    assert len(alerter.alert_history) == 1
    assert alerter.alert_history[0].level == "WARNING"
    assert "shadow" in alerter.alert_history[0].message.lower()


# ---------------------------------------------------------------------------
# Post-shadow (expired window) tests
# ---------------------------------------------------------------------------

def test_post_shadow_halt_requested() -> None:
    """After shadow window expires, report_drift must set halt_requested."""
    # shadow_mode_hours=0 means window expired immediately
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        _start_time=past,
    )
    assert not detector.in_shadow_mode
    detector.report_drift("adwin", "return")
    assert detector.halt_requested


def test_post_shadow_schema_drift_halt() -> None:
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        _start_time=past,
    )
    detector.report_schema_drift("column_removed", on_drift="halt")
    assert detector.halt_requested


def test_post_shadow_schema_drift_warn_no_halt() -> None:
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        _start_time=past,
    )
    detector.report_schema_drift("column_removed", on_drift="warn")
    assert not detector.halt_requested


def test_post_shadow_alerter_called_with_critical(tmp_path) -> None:
    """After shadow mode, alerter.send_alert must be called with level=CRITICAL."""
    from deployment.monitoring.alerter import TradingAlerter

    alerter = TradingAlerter({"alert_channels": ["file"], "log_dir": str(tmp_path)})
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        alerter=alerter,
        _start_time=past,
    )
    detector.report_drift("adwin", "return")
    assert any(r.level == "CRITICAL" for r in alerter.alert_history)


# ---------------------------------------------------------------------------
# reset_halt
# ---------------------------------------------------------------------------

def test_reset_halt_clears_flag() -> None:
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        _start_time=past,
    )
    detector.report_drift("adwin", "return")
    assert detector.halt_requested
    detector.reset_halt()
    assert not detector.halt_requested


def test_reset_halt_operator_sends_alert() -> None:
    """I8-d: source='operator' triggers a WARNING alert."""
    from unittest.mock import MagicMock
    alerter = MagicMock()
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        alerter=alerter,
        _start_time=past,
    )
    detector.halt_requested = True
    detector.reset_halt(source="operator")
    assert not detector.halt_requested
    alerter.send_alert.assert_called_once()
    assert alerter.send_alert.call_args[1].get("level") == "WARNING"


def test_reset_halt_auto_drill_no_alert() -> None:
    """I8-d: source='auto_drill' clears halt silently."""
    from unittest.mock import MagicMock
    alerter = MagicMock()
    past = time.time() - 10
    detector = DeploymentDriftDetector(
        {"drift": {"shadow_mode_hours": 0}},
        alerter=alerter,
        _start_time=past,
    )
    detector.halt_requested = True
    detector.reset_halt(source="auto_drill")
    assert not detector.halt_requested
    alerter.send_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Default config / threshold values
# ---------------------------------------------------------------------------

def test_default_thresholds_from_config() -> None:
    cfg = {
        "drift": {
            "shadow_mode_hours": 72,
            "reward_return_sigma_threshold": 2.5,
            "feature_psi_threshold": 0.25,
            "pnl_z_threshold": 3.5,
            "action_entropy_min": 0.6,
        }
    }
    detector = DeploymentDriftDetector(cfg)
    assert detector.reward_return_sigma_threshold == 2.5
    assert detector.feature_psi_threshold == 0.25
    assert detector.pnl_z_threshold == 3.5
    assert detector.action_entropy_min == 0.6


def test_default_thresholds_fallback() -> None:
    detector = DeploymentDriftDetector({})
    assert detector.reward_return_sigma_threshold == 2.0
    assert detector.feature_psi_threshold == 0.2
    assert detector.pnl_z_threshold == 3.0
    assert detector.action_entropy_min == 0.5
