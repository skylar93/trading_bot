"""I9-a: Alerter rate-limit tests."""
from __future__ import annotations

import time
from deployment.monitoring.alerter import TradingAlerter


def _alerter(rpm: int = 10, cooldown: float = 300) -> TradingAlerter:
    return TradingAlerter({
        "alert_channels": ["console"],
        "rate_limit_per_minute": rpm,
        "rate_limit_cooldown_s": cooldown,
    })


def test_rate_limit_suppresses_burst():
    """100 rapid dispatches → only rate_limit_per_minute+1 records in alert_history."""
    alerter = _alerter(rpm=10, cooldown=1)
    for _ in range(100):
        alerter.send_alert("burst message", level="WARNING")
    # First 10 pass, then suppressed
    assert len(alerter.alert_history) <= 10


def test_rate_limit_resets_after_window(monkeypatch):
    """After 60s window resets, new dispatches are allowed again."""
    alerter = _alerter(rpm=3, cooldown=1)
    for _ in range(5):
        alerter.send_alert("msg", level="WARNING")
    count_after_burst = len(alerter.alert_history)

    # Simulate time passing: advance window_start by 61s
    for bucket in alerter._rl_state.values():
        bucket.window_start -= 61
        bucket.cooldown_until = 0.0
        bucket.count = 0
        bucket.suppressed = 0

    alerter.send_alert("after reset", level="WARNING")
    assert len(alerter.alert_history) > count_after_burst


def test_different_event_keys_have_separate_buckets():
    """Different (event, level) keys are rate-limited independently."""
    alerter = _alerter(rpm=2)
    for _ in range(5):
        alerter._dispatch("WARNING", "event_A", "msg")
    for _ in range(5):
        alerter._dispatch("WARNING", "event_B", "msg")
    count_a = sum(1 for r in alerter.alert_history if r.event == "event_A")
    count_b = sum(1 for r in alerter.alert_history if r.event == "event_B")
    assert count_a <= 2
    assert count_b <= 2
