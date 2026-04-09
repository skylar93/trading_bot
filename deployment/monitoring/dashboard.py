"""
Minimal HTTP dashboard for real-time trading metrics.

Serves:
  GET /metrics          → JSON snapshot of current trading state
  GET /metrics/history  → JSON array of recent snapshots (last 100)
  GET /health           → {"status": "ok"}

Usage:
    from deployment.monitoring.dashboard import start_dashboard
    start_dashboard(metrics_exporter, port=8080)
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)


def start_dashboard(metrics_exporter, port: int = 8080) -> threading.Thread:
    """Start dashboard HTTP server in a daemon thread.

    Args:
        metrics_exporter: MetricsExporter instance
        port: HTTP port (default 8080)

    Returns:
        threading.Thread: The daemon thread running the server
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                data = metrics_exporter.to_json()
                self._json_response(200, data)
            elif self.path == "/metrics/history":
                history = metrics_exporter.history(last_n=100)
                data = [
                    {
                        "timestamp": s.timestamp,
                        "portfolio_value": s.portfolio_value,
                        "drawdown_pct": s.drawdown_pct,
                        "position": s.position,
                        "num_trades": s.num_trades,
                    }
                    for s in history
                ]
                self._json_response(200, data)
            elif self.path == "/health":
                self._json_response(200, {"status": "ok"})
            else:
                self._json_response(404, {"error": "not found"})

        def _json_response(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format, *args):
            logger.debug(format, *args)

    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Dashboard server started on port %d", port)
    return thread
