"""
Week 78 (H1-H5) Observability Stack tests.

H1: Prometheus integration — all snapshot fields exported
H2: Grafana dashboard JSON — well-formed, required panels present
H3: Alerter extensions — Discord channel, notify_error/kill_switch/audit_chain_break
H4: OpenTelemetry tracing — span context manager, no-op fallback
H5: Sentry scrubbing — credential and price data stripped from events
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from deployment.monitoring.alerter import TradingAlerter, AlertRecord
from deployment.monitoring.metrics_exporter import MetricsExporter, MetricSnapshot
from deployment.monitoring.sentry_init import _scrub_dict, _before_send


# ─────────────────────────────────────────────────────────────────────────────
# H1: Prometheus integration
# ─────────────────────────────────────────────────────────────────────────────

class TestPrometheusIntegration:
    """MetricsExporter Prometheus backend — all fields mapped."""

    def test_prometheus_disabled_by_default(self):
        exporter = MetricsExporter()
        assert exporter._use_prometheus is False

    def test_prometheus_enabled_with_flag(self):
        """When use_prometheus=True and prometheus_client available, server starts."""
        import prometheus_client  # noqa: F401 — assert package present

        started_ports = []

        def fake_start(port):
            started_ports.append(port)

        with patch("deployment.monitoring.metrics_exporter.start_http_server", fake_start):
            exporter = MetricsExporter({"use_prometheus": True, "prometheus_port": 19100})

        assert exporter._use_prometheus is True
        assert 19100 in started_ports

    def test_all_prom_gauges_registered(self):
        """All MetricSnapshot fields have a corresponding Prometheus entry in _prom."""
        # Patch metric classes so we don't hit the global CollectorRegistry
        with patch("deployment.monitoring.metrics_exporter.start_http_server"), \
             patch("deployment.monitoring.metrics_exporter.Gauge", MagicMock()), \
             patch("deployment.monitoring.metrics_exporter.Counter", MagicMock()), \
             patch("deployment.monitoring.metrics_exporter.Histogram", MagicMock()):
            exporter = MetricsExporter({"use_prometheus": True, "prometheus_port": 19101})

        expected_keys = {
            "portfolio_value", "cash", "position", "unrealised_pnl", "realised_pnl",
            "daily_pnl", "drawdown_pct", "current_var", "is_halted", "kill_switch_active",
            "win_rate", "sharpe_ratio", "rolling_sharpe", "rolling_sortino",
            "pnl_market_move", "pnl_slippage_cost", "pnl_fees", "pnl_net",
            "drift_detected", "current_regime",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "order_latency_histogram",
            "num_trades", "alerts_fired", "feature_drift_alarms",
        }
        assert expected_keys.issubset(set(exporter._prom.keys()))

    def test_update_pushes_all_gauges(self):
        """After update(), all gauge values reflect the snapshot."""
        set_values: Dict[str, Any] = {}

        class FakeGauge:
            def __init__(self, name, *a, **k):
                self.name = name

            def set(self, v):
                set_values[self.name] = v

        class FakeCounter:
            def __init__(self, *a, **k):
                pass

            def inc(self, v=1):
                pass

        class FakeHistogram:
            def __init__(self, *a, **k):
                pass

            def observe(self, v):
                pass

        with patch("deployment.monitoring.metrics_exporter.start_http_server"), \
             patch("deployment.monitoring.metrics_exporter.Gauge", FakeGauge), \
             patch("deployment.monitoring.metrics_exporter.Counter", FakeCounter), \
             patch("deployment.monitoring.metrics_exporter.Histogram", FakeHistogram):
            exporter = MetricsExporter({"use_prometheus": True, "prometheus_port": 19102})
            exporter.update(
                portfolio_value=12345.0,
                drawdown_pct=0.07,
                is_halted=True,
                kill_switch_active=True,
            )

        assert set_values.get("trading_portfolio_value_usd") == pytest.approx(12345.0)
        assert set_values.get("trading_drawdown_pct") == pytest.approx(0.07)
        assert set_values.get("trading_is_halted") == 1.0
        assert set_values.get("trading_kill_switch_active") == 1.0

    def test_observe_order_latency(self):
        """observe_order_latency() calls histogram.observe()."""
        observed = []

        class FakeHistogram:
            def __init__(self, *a, **k):
                pass

            def observe(self, v):
                observed.append(v)

        with patch("deployment.monitoring.metrics_exporter.start_http_server"), \
             patch("deployment.monitoring.metrics_exporter.Gauge", MagicMock), \
             patch("deployment.monitoring.metrics_exporter.Counter", MagicMock), \
             patch("deployment.monitoring.metrics_exporter.Histogram", FakeHistogram):
            exporter = MetricsExporter({"use_prometheus": True, "prometheus_port": 19103})
            exporter.observe_order_latency(42.5)

        assert observed == [42.5]

    def test_counter_delta_only(self):
        """Counter.inc() is called with delta (not absolute value)."""
        incs = []

        class FakeGauge:
            def __init__(self, *a, **k):
                pass
            def set(self, v):
                pass

        class FakeCounter:
            def __init__(self, *a, **k):
                pass
            def inc(self, v=1):
                incs.append(v)

        class FakeHistogram:
            def __init__(self, *a, **k):
                pass
            def observe(self, v):
                pass

        with patch("deployment.monitoring.metrics_exporter.start_http_server"), \
             patch("deployment.monitoring.metrics_exporter.Gauge", FakeGauge), \
             patch("deployment.monitoring.metrics_exporter.Counter", FakeCounter), \
             patch("deployment.monitoring.metrics_exporter.Histogram", FakeHistogram):
            exporter = MetricsExporter({"use_prometheus": True, "prometheus_port": 19104})
            exporter.update(num_trades=5, alerts_fired=2)
            exporter.update(num_trades=7, alerts_fired=3)

        # First update: delta 5 trades, 2 alerts
        # Second update: delta 2 trades, 1 alert
        assert 5 in incs
        assert 2 in incs
        assert 1 in incs

    def test_kill_switch_field_in_snapshot(self):
        """MetricSnapshot has kill_switch_active field."""
        exporter = MetricsExporter()
        snap = exporter.update(kill_switch_active=True)
        assert snap.kill_switch_active is True
        assert exporter.to_json()["kill_switch_active"] is True


# ─────────────────────────────────────────────────────────────────────────────
# H2: Grafana dashboard JSON
# ─────────────────────────────────────────────────────────────────────────────

class TestGrafanaDashboard:
    """grafana_dashboard.json — structural validity checks."""

    @pytest.fixture(scope="class")
    def dashboard(self) -> Dict[str, Any]:
        path = Path(__file__).parent.parent / "deployment/monitoring/grafana_dashboard.json"
        return json.loads(path.read_text())

    def test_file_is_valid_json(self, dashboard):
        assert isinstance(dashboard, dict)

    def test_required_top_level_keys(self, dashboard):
        for key in ("title", "panels", "uid", "schemaVersion", "refresh"):
            assert key in dashboard, f"Missing key: {key}"

    def test_has_prometheus_input(self, dashboard):
        inputs = dashboard.get("__inputs", [])
        assert any(i.get("pluginId") == "prometheus" for i in inputs)

    def test_panels_cover_required_topics(self, dashboard):
        all_text = json.dumps(dashboard).lower()
        required_topics = [
            "pnl",          # P&L attribution
            "latency",      # order latency
            "drawdown",     # drawdown panel
            "drift",        # drift alarms
            "kill_switch",  # kill switch status
        ]
        for topic in required_topics:
            assert topic in all_text, f"Dashboard missing panel for: {topic}"

    def test_uid_is_set(self, dashboard):
        assert dashboard["uid"] not in (None, ""), "Dashboard UID must be set"

    def test_auto_refresh(self, dashboard):
        assert dashboard.get("refresh") not in (None, ""), "Dashboard should auto-refresh"

    def test_panel_count(self, dashboard):
        panels = [p for p in dashboard["panels"] if p.get("type") != "row"]
        assert len(panels) >= 8, f"Expected at least 8 data panels, got {len(panels)}"


# ─────────────────────────────────────────────────────────────────────────────
# H3: Alerter extensions
# ─────────────────────────────────────────────────────────────────────────────

class TestAlerterExtensions:
    """New methods and Discord channel added in H3."""

    def _make_alerter(self, extra_config=None):
        cfg = {"alert_channels": ["console"]}
        if extra_config:
            cfg.update(extra_config)
        return TradingAlerter(cfg)

    # notify_error ────────────────────────────────────────────

    def test_notify_error_records_alert(self):
        alerter = self._make_alerter()
        alerter.notify_error("connection refused", context="order_submit")
        assert len(alerter.alert_history) == 1
        rec = alerter.alert_history[0]
        assert rec.event == "runtime_error"
        assert rec.level == "ERROR"
        assert "connection refused" in rec.message

    def test_notify_error_without_context(self):
        alerter = self._make_alerter()
        alerter.notify_error("timeout")
        assert alerter.alert_history[0].message == "Runtime error: timeout"

    # notify_kill_switch ──────────────────────────────────────

    def test_notify_kill_switch_records_critical(self):
        alerter = self._make_alerter()
        alerter.notify_kill_switch(reason="drawdown_exceeded")
        rec = alerter.alert_history[0]
        assert rec.event == "kill_switch_activated"
        assert rec.level == "CRITICAL"
        assert "drawdown_exceeded" in rec.message

    def test_notify_kill_switch_default_reason(self):
        alerter = self._make_alerter()
        alerter.notify_kill_switch()
        assert "manual" in alerter.alert_history[0].message

    # notify_audit_chain_break ────────────────────────────────

    def test_notify_audit_chain_break_records(self):
        alerter = self._make_alerter()
        alerter.notify_audit_chain_break(details="hash mismatch at step 42")
        rec = alerter.alert_history[0]
        assert rec.event == "audit_chain_break"
        assert rec.level == "CRITICAL"
        assert "hash mismatch" in rec.message

    # Discord channel ─────────────────────────────────────────

    def test_discord_channel_sends_embed(self):
        """Discord channel calls the webhook URL with an embed payload."""
        sent_payloads = []

        class FakeResponse:
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        def fake_urlopen(req, timeout=10):
            sent_payloads.append(json.loads(req.data.decode()))
            return FakeResponse()

        alerter = TradingAlerter({
            "alert_channels": ["discord"],
            "discord_webhook_url": "https://discord.com/api/webhooks/fake/url",
        })
        with patch("deployment.monitoring.alerter.urllib.request.urlopen", fake_urlopen):
            alerter.send_alert("test discord message", level="WARNING")

        assert len(sent_payloads) == 1
        payload = sent_payloads[0]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert "test discord message" in embed["description"]
        assert embed["title"].startswith("[WARNING]")

    def test_discord_no_url_logs_warning(self, caplog):
        """Discord channel with no URL logs a warning, no exception."""
        import logging
        alerter = TradingAlerter({"alert_channels": ["discord"]})
        with caplog.at_level(logging.WARNING):
            alerter.send_alert("test")
        assert "discord_webhook_url" in caplog.text.lower() or "discord" in caplog.text.lower()

    def test_discord_env_var_override(self, monkeypatch):
        """DISCORD_WEBHOOK_URL env var is picked up."""
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/env/url")
        alerter = TradingAlerter({"alert_channels": ["discord"]})
        assert alerter._discord_url == "https://discord.com/api/webhooks/env/url"

    def test_all_channels_record_history(self):
        """alert_history is populated regardless of channel."""
        alerter = TradingAlerter({"alert_channels": ["console"]})
        alerter.notify_error("e1")
        alerter.notify_kill_switch()
        alerter.notify_audit_chain_break()
        assert len(alerter.alert_history) == 3


# ─────────────────────────────────────────────────────────────────────────────
# H4: OpenTelemetry tracing
# ─────────────────────────────────────────────────────────────────────────────

class TestOTelTracing:
    """start_span() context manager — functional and no-op paths."""

    def test_start_span_no_exception(self):
        """start_span runs without error when OTel is available."""
        from deployment.monitoring.tracing import start_span
        with start_span("trading.test.span", {"key": "val"}) as span:
            assert span is not None

    def test_start_span_records_exception(self):
        """start_span re-raises exceptions and records them on the span."""
        from deployment.monitoring.tracing import start_span
        with pytest.raises(ValueError, match="deliberate"):
            with start_span("trading.test.error"):
                raise ValueError("deliberate")

    def test_noop_tracer_safe(self):
        """_NoopTracer is safe to use directly."""
        from deployment.monitoring.tracing import _NoopTracer
        tracer = _NoopTracer()
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("k", "v")  # no error

    def test_get_tracer_returns_tracer(self):
        """get_tracer() returns a usable tracer."""
        from deployment.monitoring.tracing import get_tracer
        tracer = get_tracer()
        assert tracer is not None

    def test_init_tracing_idempotent(self):
        """Calling init_tracing() twice does not raise."""
        from deployment.monitoring import tracing
        # Reset state
        tracing._provider = None
        tracing._tracer = None
        tracing.init_tracing(service_name="test-bot-idempotent")
        tracing.init_tracing(service_name="test-bot-idempotent")  # second call — no error

    def test_shutdown_tracing(self):
        """shutdown_tracing() does not raise."""
        from deployment.monitoring.tracing import shutdown_tracing
        shutdown_tracing()  # idempotent


# ─────────────────────────────────────────────────────────────────────────────
# H5: Sentry scrubbing
# ─────────────────────────────────────────────────────────────────────────────

class TestSentryScrubbing:
    """_scrub_dict and _before_send strip credentials and price arrays."""

    def test_api_key_scrubbed(self):
        data = {"api_key": "SUPER_SECRET_1234567890abcdef"}
        result = _scrub_dict(data)
        assert result["api_key"] == "[scrubbed]"

    def test_nested_secret_scrubbed(self):
        data = {"exchange": {"api_secret": "abc123def456abc123def456abc123de"}}
        result = _scrub_dict(data)
        assert result["exchange"]["api_secret"] == "[scrubbed]"

    def test_large_price_array_scrubbed(self):
        """A 'prices' key with a large array is scrubbed."""
        data = {"prices": list(range(500))}
        result = _scrub_dict(data)
        assert result["prices"] == "[scrubbed]"

    def test_small_price_array_kept(self):
        """A small list under 200 chars is not scrubbed."""
        data = {"prices": [1.0, 2.0, 3.0]}
        result = _scrub_dict(data)
        # Small list — not scrubbed
        assert result["prices"] == [1.0, 2.0, 3.0]

    def test_non_sensitive_keys_kept(self):
        data = {"symbol": "BTCUSDT", "portfolio_value": 10000.0}
        result = _scrub_dict(data)
        assert result["symbol"] == "BTCUSDT"
        assert result["portfolio_value"] == 10000.0

    def test_long_base64_string_scrubbed(self):
        """Bare long base64-ish strings are scrubbed at the value level."""
        long_secret = "A" * 64
        data = {"some_field": long_secret}
        result = _scrub_dict(data)
        assert result["some_field"] == "[scrubbed]"

    def test_short_string_kept(self):
        short = "hello"
        data = {"msg": short}
        result = _scrub_dict(data)
        assert result["msg"] == "hello"

    def test_before_send_scrubs_extra(self):
        event = {
            "extra": {"api_key": "SECRETKEY12345678901234567890ab"},
            "exception": {"values": []},
        }
        cleaned = _before_send(event, {})
        assert cleaned["extra"]["api_key"] == "[scrubbed]"

    def test_before_send_scrubs_frame_vars(self):
        event = {
            "exception": {
                "values": [
                    {
                        "stacktrace": {
                            "frames": [
                                {"vars": {"api_secret": "mysecret1234567890abc123def456"}}
                            ]
                        }
                    }
                ]
            }
        }
        cleaned = _before_send(event, {})
        frame_vars = cleaned["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]
        assert frame_vars["api_secret"] == "[scrubbed]"

    def test_init_sentry_no_dsn(self):
        """init_sentry() returns False and does not raise when no DSN set."""
        from deployment.monitoring.sentry_init import init_sentry
        result = init_sentry(dsn=None)
        assert result is False

    def test_init_sentry_with_dsn(self):
        """init_sentry() calls sentry_sdk.init with correct args."""
        from deployment.monitoring.sentry_init import init_sentry
        with patch("sentry_sdk.init") as mock_init:
            result = init_sentry(
                dsn="https://test@sentry.io/123",
                traces_sample_rate=0.2,
                environment="paper",
            )
        assert result is True
        mock_init.assert_called_once()
        kwargs = mock_init.call_args[1]
        assert kwargs["environment"] == "paper"
        assert kwargs["traces_sample_rate"] == pytest.approx(0.2)
        assert callable(kwargs["before_send"])

    def test_capture_exception_silent_on_error(self):
        """capture_exception() never raises even if Sentry internals fail."""
        from deployment.monitoring.sentry_init import capture_exception
        with patch("sentry_sdk.push_scope", side_effect=RuntimeError("sentry broke")):
            # Must not propagate
            capture_exception(ValueError("test"), context={"api_key": "secret"})
