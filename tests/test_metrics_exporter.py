"""Test MetricsExporter and dashboard endpoint."""
import time
import json
import threading
import pytest
from unittest.mock import MagicMock

from deployment.monitoring.metrics_exporter import MetricsExporter, MetricSnapshot


class TestMetricsExporter:
    def test_basic_update(self):
        exporter = MetricsExporter()
        snap = exporter.update(portfolio_value=10000.0, cash=5000.0, position=0.5)
        assert snap.portfolio_value == 10000.0
        assert snap.cash == 5000.0

    def test_history_accumulates(self):
        exporter = MetricsExporter()
        for i in range(10):
            exporter.update(portfolio_value=10000.0 + i * 100)
        assert len(exporter.history()) == 10
        assert exporter.history(last_n=3)[-1].portfolio_value == 10900.0

    def test_max_history_limit(self):
        exporter = MetricsExporter({"max_history": 5})
        for i in range(20):
            exporter.update(portfolio_value=float(i))
        assert len(exporter.history()) == 5
        assert exporter.history()[0].portfolio_value == 15.0

    def test_to_json(self):
        exporter = MetricsExporter()
        exporter.update(portfolio_value=10000.0, drawdown_pct=0.05)
        data = exporter.to_json()
        assert data["portfolio_value"] == 10000.0
        assert data["drawdown_pct"] == 0.05

    def test_snapshot_returns_latest(self):
        exporter = MetricsExporter()
        exporter.update(portfolio_value=100.0)
        exporter.update(portfolio_value=200.0)
        assert exporter.snapshot().portfolio_value == 200.0

    def test_thread_safety(self):
        exporter = MetricsExporter()
        def writer():
            for i in range(500):
                exporter.update(portfolio_value=float(i))
        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=writer)
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert len(exporter.history()) == 1000


class TestDashboard:
    def test_health_endpoint(self):
        """Test that dashboard /health returns 200."""
        import urllib.request
        from deployment.monitoring.dashboard import start_dashboard

        exporter = MetricsExporter()
        exporter.update(portfolio_value=10000.0)
        thread = start_dashboard(exporter, port=18080)
        time.sleep(0.3)  # let server start

        try:
            resp = urllib.request.urlopen("http://localhost:18080/health", timeout=2)
            data = json.loads(resp.read())
            assert data["status"] == "ok"
        finally:
            pass  # daemon thread auto-cleans

    def test_metrics_endpoint(self):
        import urllib.request
        from deployment.monitoring.dashboard import start_dashboard

        exporter = MetricsExporter()
        exporter.update(portfolio_value=12345.0)
        thread = start_dashboard(exporter, port=18081)
        time.sleep(0.3)

        try:
            resp = urllib.request.urlopen("http://localhost:18081/metrics", timeout=2)
            data = json.loads(resp.read())
            assert data["portfolio_value"] == 12345.0
        finally:
            pass
