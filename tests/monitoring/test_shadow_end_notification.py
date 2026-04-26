"""I10: Shadow-end notification fires exactly once."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from deployment.monitoring.drift_detector import DeploymentDriftDetector


def test_shadow_end_notification_fires_once():
    """Notification sends exactly once when shadow mode ends."""
    alerter = MagicMock()
    dd = DeploymentDriftDetector(
        config={"drift": {"shadow_mode_hours": 0}},
        alerter=alerter,
        _start_time=time.time() - 3600,  # already past shadow
    )

    # Access in_shadow_mode multiple times
    _ = dd.in_shadow_mode
    _ = dd.in_shadow_mode
    _ = dd.in_shadow_mode

    alerter.send_alert.assert_called_once()
    call_args = alerter.send_alert.call_args
    assert "Shadow mode ended" in call_args[0][0]
    assert call_args[1].get("level") == "CRITICAL"


def test_shadow_end_not_fired_during_shadow():
    """No notification while still in shadow mode."""
    alerter = MagicMock()
    dd = DeploymentDriftDetector(
        config={"drift": {"shadow_mode_hours": 1000}},
        alerter=alerter,
    )
    _ = dd.in_shadow_mode
    alerter.send_alert.assert_not_called()


def test_shadow_end_notification_without_alerter():
    """Shadow-end transition without alerter doesn't raise."""
    dd = DeploymentDriftDetector(
        config={"drift": {"shadow_mode_hours": 0}},
        alerter=None,
        _start_time=time.time() - 3600,
    )
    assert not dd.in_shadow_mode  # should not raise
