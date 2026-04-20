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

import numpy as np

logger = logging.getLogger(__name__)

# Try prometheus_client, fallback to in-memory
try:
    from prometheus_client import (
        Gauge,
        Counter,
        Histogram,
        start_http_server,
        REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed; using in-memory metrics")

# Latency histogram buckets (ms): sub-ms to 5s
_LATENCY_BUCKETS = (
    0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000
)


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
    # S52: order latency (ms)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    # S53: rolling performance
    rolling_sharpe: float = 0.0
    rolling_sortino: float = 0.0
    # S51: P&L attribution totals
    pnl_market_move: float = 0.0
    pnl_slippage_cost: float = 0.0
    pnl_fees: float = 0.0
    pnl_net: float = 0.0
    # S57: current market regime (0=low-vol, 1=medium-vol, 2=high-vol; -1=unknown)
    current_regime: int = -1
    # S56: number of feature drift alarms fired (cumulative)
    feature_drift_alarms: int = 0
    # Kill switch status (H3/H1 integration)
    kill_switch_active: bool = False


class MetricsExporter:
    """Collect, store, and export trading metrics.

    Supports two backends:
    - prometheus: exposes /metrics endpoint on configurable port (default 9100)
    - memory: stores snapshots in-memory, queryable via snapshot()/history()

    All 30 MetricSnapshot fields are mapped to Prometheus metrics:
    - Gauge  : portfolio_value, cash, position, p&l, drawdown, ratios, regime, latencies
    - Counter: num_trades, alerts_fired, feature_drift_alarms (monotonic)
    - Histogram: order latency (p50/p95/p99 observed directly via observe())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._lock = threading.Lock()
        self._history: List[MetricSnapshot] = []
        self._max_history = int(config.get("max_history", 10_000))
        self._latest: Optional[MetricSnapshot] = None

        # Prometheus backend
        self._use_prometheus = config.get("use_prometheus", False) and HAS_PROMETHEUS
        self._prom: Dict[str, Any] = {}
        if self._use_prometheus:
            port = int(config.get("prometheus_port", 9100))
            self._init_prometheus(port)

    def _init_prometheus(self, port: int) -> None:
        """Register Prometheus gauges/counters/histograms and start HTTP server."""
        # Gauges — portfolio state
        self._prom["portfolio_value"] = Gauge(
            "trading_portfolio_value_usd", "Current portfolio value (USD)"
        )
        self._prom["cash"] = Gauge(
            "trading_cash_usd", "Available cash (USD)"
        )
        self._prom["position"] = Gauge(
            "trading_position_size", "Current position size (base asset)"
        )
        self._prom["unrealised_pnl"] = Gauge(
            "trading_unrealised_pnl_usd", "Mark-to-market unrealised P&L (USD)"
        )
        self._prom["realised_pnl"] = Gauge(
            "trading_realised_pnl_usd", "Realised P&L (USD)"
        )
        self._prom["daily_pnl"] = Gauge(
            "trading_daily_pnl_usd", "Daily P&L (USD)"
        )

        # Gauges — risk
        self._prom["drawdown_pct"] = Gauge(
            "trading_drawdown_pct", "Current drawdown fraction (0-1)"
        )
        self._prom["current_var"] = Gauge(
            "trading_var_usd", "Current Value-at-Risk (USD)"
        )
        self._prom["is_halted"] = Gauge(
            "trading_is_halted", "Trading halt flag (1=halted, 0=running)"
        )
        self._prom["kill_switch_active"] = Gauge(
            "trading_kill_switch_active", "Kill switch status (1=active)"
        )

        # Gauges — performance
        self._prom["win_rate"] = Gauge(
            "trading_win_rate", "Win rate fraction (0-1)"
        )
        self._prom["sharpe_ratio"] = Gauge(
            "trading_sharpe_ratio", "Sharpe ratio (annualised)"
        )
        self._prom["rolling_sharpe"] = Gauge(
            "trading_rolling_sharpe_ratio", "Rolling Sharpe ratio (20-step window)"
        )
        self._prom["rolling_sortino"] = Gauge(
            "trading_rolling_sortino_ratio", "Rolling Sortino ratio (20-step window)"
        )

        # Gauges — P&L attribution
        self._prom["pnl_market_move"] = Gauge(
            "trading_pnl_market_move_usd", "P&L from market direction (USD)"
        )
        self._prom["pnl_slippage_cost"] = Gauge(
            "trading_pnl_slippage_cost_usd", "P&L cost from slippage (USD)"
        )
        self._prom["pnl_fees"] = Gauge(
            "trading_pnl_fees_usd", "P&L cost from trading fees (USD)"
        )
        self._prom["pnl_net"] = Gauge(
            "trading_pnl_net_usd", "Net P&L (market_move - slippage - fees, USD)"
        )

        # Gauges — drift / regime
        self._prom["drift_detected"] = Gauge(
            "trading_drift_detected", "Concept drift detected flag (1=yes)"
        )
        self._prom["current_regime"] = Gauge(
            "trading_current_regime",
            "Market regime (0=low-vol, 1=medium-vol, 2=high-vol, -1=unknown)",
        )

        # Gauges — latency percentiles (ms)
        self._prom["latency_p50_ms"] = Gauge(
            "trading_order_latency_p50_ms", "Order latency p50 (ms)"
        )
        self._prom["latency_p95_ms"] = Gauge(
            "trading_order_latency_p95_ms", "Order latency p95 (ms)"
        )
        self._prom["latency_p99_ms"] = Gauge(
            "trading_order_latency_p99_ms", "Order latency p99 (ms)"
        )

        # Histogram — raw latency observations (each order submission calls observe())
        self._prom["order_latency_histogram"] = Histogram(
            "trading_order_latency_ms",
            "Order round-trip latency (ms)",
            buckets=_LATENCY_BUCKETS,
        )

        # Counters — monotonically increasing
        self._prom["num_trades"] = Counter(
            "trading_trades_total", "Total number of trades executed"
        )
        self._prom["alerts_fired"] = Counter(
            "trading_alerts_total", "Total number of alerts dispatched"
        )
        self._prom["feature_drift_alarms"] = Counter(
            "trading_feature_drift_alarms_total",
            "Cumulative feature drift alarms fired",
        )

        start_http_server(port)
        logger.info("Prometheus metrics server started on :%d/metrics", port)

    # ------------------------------------------------------------------
    # Counter tracking (needed for monotonic-only counters)
    # ------------------------------------------------------------------

    _prev_trades: int = 0
    _prev_alerts: int = 0
    _prev_drift_alarms: int = 0

    def _update_prometheus(self, snap: MetricSnapshot) -> None:
        """Push all snapshot fields to registered Prometheus metrics."""
        g = self._prom

        # Portfolio state
        g["portfolio_value"].set(snap.portfolio_value)
        g["cash"].set(snap.cash)
        g["position"].set(snap.position)
        g["unrealised_pnl"].set(snap.unrealised_pnl)
        g["realised_pnl"].set(snap.realised_pnl)
        g["daily_pnl"].set(snap.daily_pnl)

        # Risk
        g["drawdown_pct"].set(snap.drawdown_pct)
        g["current_var"].set(snap.current_var)
        g["is_halted"].set(1.0 if snap.is_halted else 0.0)
        g["kill_switch_active"].set(1.0 if snap.kill_switch_active else 0.0)

        # Performance
        g["win_rate"].set(snap.win_rate)
        g["sharpe_ratio"].set(snap.sharpe_ratio)
        g["rolling_sharpe"].set(snap.rolling_sharpe)
        g["rolling_sortino"].set(snap.rolling_sortino)

        # P&L attribution
        g["pnl_market_move"].set(snap.pnl_market_move)
        g["pnl_slippage_cost"].set(snap.pnl_slippage_cost)
        g["pnl_fees"].set(snap.pnl_fees)
        g["pnl_net"].set(snap.pnl_net)

        # Drift / regime
        g["drift_detected"].set(1.0 if snap.drift_detected else 0.0)
        g["current_regime"].set(snap.current_regime)

        # Latency percentile gauges
        g["latency_p50_ms"].set(snap.latency_p50_ms)
        g["latency_p95_ms"].set(snap.latency_p95_ms)
        g["latency_p99_ms"].set(snap.latency_p99_ms)

        # Monotonic counters — only inc() by delta
        delta_trades = max(0, snap.num_trades - self._prev_trades)
        if delta_trades:
            g["num_trades"].inc(delta_trades)
            self._prev_trades = snap.num_trades

        delta_alerts = max(0, snap.alerts_fired - self._prev_alerts)
        if delta_alerts:
            g["alerts_fired"].inc(delta_alerts)
            self._prev_alerts = snap.alerts_fired

        delta_drift = max(0, snap.feature_drift_alarms - self._prev_drift_alarms)
        if delta_drift:
            g["feature_drift_alarms"].inc(delta_drift)
            self._prev_drift_alarms = snap.feature_drift_alarms

    def observe_order_latency(self, latency_ms: float) -> None:
        """Record a raw order latency observation in the Prometheus histogram."""
        if self._use_prometheus and "order_latency_histogram" in self._prom:
            self._prom["order_latency_histogram"].observe(latency_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        if self._use_prometheus:
            self._update_prometheus(snap)

        return snap

    def update_latency(self, p50: float, p95: float, p99: float) -> MetricSnapshot:
        """Record order latency percentiles (ms) as a metric update."""
        return self.update(
            latency_p50_ms=float(p50),
            latency_p95_ms=float(p95),
            latency_p99_ms=float(p99),
        )

    def rolling_sharpe(self, window: int = 20, annualize: int = 252) -> float:
        """Compute rolling Sharpe ratio from the last *window* portfolio-value snapshots."""
        snaps = self.history(last_n=window + 1)
        if len(snaps) < 2:
            return 0.0
        values = np.array([s.portfolio_value for s in snaps], dtype=float)
        prev = np.where(values[:-1] != 0, values[:-1], 1e-8)
        returns = np.diff(values) / prev
        std = np.std(returns)
        if std < 1e-10:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(annualize))

    def rolling_sortino(
        self,
        window: int = 20,
        annualize: int = 252,
        mar: float = 0.0,
    ) -> float:
        """Compute rolling Sortino ratio from the last *window* portfolio-value snapshots."""
        snaps = self.history(last_n=window + 1)
        if len(snaps) < 2:
            return 0.0
        values = np.array([s.portfolio_value for s in snaps], dtype=float)
        prev = np.where(values[:-1] != 0, values[:-1], 1e-8)
        returns = np.diff(values) / prev
        downside = returns[returns < mar]
        if len(downside) == 0:
            return float(np.mean(returns) * np.sqrt(annualize))
        downside_std = float(np.std(downside))
        if downside_std < 1e-10:
            return 0.0
        return float(np.mean(returns) / downside_std * np.sqrt(annualize))

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
            # S52
            "latency_p50_ms": snap.latency_p50_ms,
            "latency_p95_ms": snap.latency_p95_ms,
            "latency_p99_ms": snap.latency_p99_ms,
            # S53
            "rolling_sharpe": snap.rolling_sharpe,
            "rolling_sortino": snap.rolling_sortino,
            # S51
            "pnl_market_move": snap.pnl_market_move,
            "pnl_slippage_cost": snap.pnl_slippage_cost,
            "pnl_fees": snap.pnl_fees,
            "pnl_net": snap.pnl_net,
            # S57
            "current_regime": snap.current_regime,
            # S56
            "feature_drift_alarms": snap.feature_drift_alarms,
            # H1/H3
            "kill_switch_active": snap.kill_switch_active,
        }
