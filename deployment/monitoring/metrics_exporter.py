"""
Lightweight metrics exporter — Prometheus format or JSON.

Uses prometheus_client if available; otherwise falls back to
in-memory dict that can be queried via /metrics JSON endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try prometheus_client, fallback to in-memory
try:
    from prometheus_client import Gauge, Counter, Histogram, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed; using in-memory metrics")


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of all tracked metrics."""
    timestamp: float
    portfolio_value: float
    cash: float
    position: float
    unrealised_pnl: float
    realised_pnl: float
    drawdown_pct: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    drift_detected: bool
    alerts_fired: int
    # Risk metrics
    current_var: float = 0.0
    daily_pnl: float = 0.0
    is_halted: bool = False


class MetricsExporter:
    """Collect, store, and export trading metrics.

    Supports two backends:
    - prometheus: exposes /metrics endpoint on configurable port
    - memory: stores snapshots in-memory, queryable via snapshot()/history()

    Usage:
        exporter = MetricsExporter(config)
        exporter.update(portfolio_value=10200, cash=5000, ...)
        latest = exporter.snapshot()
        all_history = exporter.history()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._lock = threading.Lock()
        self._history: List[MetricSnapshot] = []
        self._max_history = int(config.get("max_history", 10_000))
        self._latest: Optional[MetricSnapshot] = None

        # Prometheus backend
        self._use_prometheus = config.get("use_prometheus", False) and HAS_PROMETHEUS
        if self._use_prometheus:
            port = int(config.get("prometheus_port", 9090))
            self._init_prometheus(port)

    def _init_prometheus(self, port: int):
        """Register Prometheus gauges and start HTTP server."""
        self._prom = {
            "portfolio_value": Gauge("trading_portfolio_value", "Current portfolio value"),
            "cash": Gauge("trading_cash", "Available cash"),
            "position": Gauge("trading_position", "Current position size"),
            "drawdown_pct": Gauge("trading_drawdown_pct", "Current drawdown %"),
            "daily_pnl": Gauge("trading_daily_pnl", "Daily P&L"),
            "num_trades": Counter("trading_num_trades_total", "Total trades executed"),
            "alerts_fired": Counter("trading_alerts_fired_total", "Total alerts fired"),
            "drift_detected": Gauge("trading_drift_detected", "Drift detection flag"),
            "is_halted": Gauge("trading_is_halted", "Trading halt flag"),
        }
        start_http_server(port)
        logger.info("Prometheus metrics server started on port %d", port)

    def update(self, **kwargs) -> MetricSnapshot:
        """Record a new metric snapshot.

        Accepts any MetricSnapshot field as keyword argument.
        Missing fields default to previous snapshot's values.
        """
        prev = self._latest
        defaults = {
            "timestamp": time.time(),
            "portfolio_value": getattr(prev, "portfolio_value", 0.0) if prev else 0.0,
            "cash": getattr(prev, "cash", 0.0) if prev else 0.0,
            "position": getattr(prev, "position", 0.0) if prev else 0.0,
            "unrealised_pnl": getattr(prev, "unrealised_pnl", 0.0) if prev else 0.0,
            "realised_pnl": getattr(prev, "realised_pnl", 0.0) if prev else 0.0,
            "drawdown_pct": getattr(prev, "drawdown_pct", 0.0) if prev else 0.0,
            "num_trades": getattr(prev, "num_trades", 0) if prev else 0,
            "win_rate": getattr(prev, "win_rate", 0.0) if prev else 0.0,
            "sharpe_ratio": getattr(prev, "sharpe_ratio", 0.0) if prev else 0.0,
            "drift_detected": getattr(prev, "drift_detected", False) if prev else False,
            "alerts_fired": getattr(prev, "alerts_fired", 0) if prev else 0,
            "current_var": getattr(prev, "current_var", 0.0) if prev else 0.0,
            "daily_pnl": getattr(prev, "daily_pnl", 0.0) if prev else 0.0,
            "is_halted": getattr(prev, "is_halted", False) if prev else False,
        }
        defaults.update(kwargs)
        snap = MetricSnapshot(**defaults)

        with self._lock:
            self._latest = snap
            self._history.append(snap)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Update Prometheus gauges
        if self._use_prometheus:
            for key in ("portfolio_value", "cash", "position", "drawdown_pct",
                        "daily_pnl", "drift_detected", "is_halted"):
                self._prom[key].set(getattr(snap, key))

        return snap

    def snapshot(self) -> Optional[MetricSnapshot]:
        """Return latest metric snapshot."""
        return self._latest

    def history(self, last_n: Optional[int] = None) -> List[MetricSnapshot]:
        """Return metric history (optionally last N entries)."""
        with self._lock:
            if last_n:
                return list(self._history[-last_n:])
            return list(self._history)

    def to_json(self) -> Dict[str, Any]:
        """Return latest snapshot as JSON-serialisable dict."""
        snap = self._latest
        if snap is None:
            return {}
        return {
            "timestamp": snap.timestamp,
            "portfolio_value": snap.portfolio_value,
            "cash": snap.cash,
            "position": snap.position,
            "unrealised_pnl": snap.unrealised_pnl,
            "realised_pnl": snap.realised_pnl,
            "drawdown_pct": snap.drawdown_pct,
            "num_trades": snap.num_trades,
            "win_rate": snap.win_rate,
            "sharpe_ratio": snap.sharpe_ratio,
            "drift_detected": snap.drift_detected,
            "alerts_fired": snap.alerts_fired,
            "current_var": snap.current_var,
            "daily_pnl": snap.daily_pnl,
            "is_halted": snap.is_halted,
        }
