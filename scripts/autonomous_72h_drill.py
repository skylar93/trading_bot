#!/usr/bin/env python3
"""
Autonomous 72h Paper Drill (Phase 7.6 I2)

Runs a paper trading simulation with:
  - Live Binance public WS feed (auth-free) with CSV/GBM fallback
  - Synthetic fault injection every N hours (FaultInjector)
  - 15-min observer snapshots
  - Auto-resume after safety net halts (30s max downtime)
  - Final report auto-written to docs/phase7/week85_72h_{start_date}.md

Feed priority:
  1. Binance public WS wss://stream.binance.com:9443/ws/btcusdt@ticker
  2. Fallback: test_data.csv replay at 1 tick/s
  3. Tertiary: synthetic GBM μ=0 σ=0.01

Usage:
    python scripts/autonomous_72h_drill.py                        # 72h
    python scripts/autonomous_72h_drill.py --duration-hours 0.083 # 5-min test
    python scripts/autonomous_72h_drill.py --feed csv             # force CSV
    python scripts/autonomous_72h_drill.py --feed gbm             # force GBM
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

# Allow running as script from project root
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deployment.monitoring.alerter import TradingAlerter
from deployment.monitoring.drift_detector import DeploymentDriftDetector
from deployment.testing.fault_injector import FaultInjector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("autonomous_72h_drill")

# ---------------------------------------------------------------------------
# Price feed implementations
# ---------------------------------------------------------------------------

_BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
_CSV_DEFAULT = pathlib.Path("test_data.csv")
_WS_FAIL_THRESHOLD = 5       # consecutive failures before fallback
_WS_HEARTBEAT_TIMEOUT = 600  # 10 min without tick → fallback


def _gbm_feed(s0: float = 30_000.0) -> Generator[float, None, None]:
    """Synthetic GBM price generator (μ=0, σ=0.01 daily, 1 tick/s)."""
    mu = 0.0
    sigma = 0.01
    dt = 1.0 / 86400
    s = s0
    while True:
        s *= math.exp(
            (mu - 0.5 * sigma ** 2) * dt
            + sigma * math.sqrt(dt) * random.gauss(0, 1)
        )
        yield s


def _csv_replay_feed(
    csv_path: pathlib.Path, tick_interval: float = 1.0
) -> Generator[float, None, None]:
    """Replay $close column from CSV at tick_interval seconds, looping forever."""
    prices: List[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next(
            (c for c in (reader.fieldnames or []) if c.lower() in ("$close", "close")),
            None,
        )
        for row in reader:
            if col and row.get(col):
                try:
                    prices.append(float(row[col]))
                except ValueError:
                    pass
    if not prices:
        logger.warning("CSV has no price data — falling back to GBM")
        yield from _gbm_feed()
        return
    logger.info("CSV replay: %d ticks loaded from %s", len(prices), csv_path)
    while True:
        for p in prices:
            yield p
            time.sleep(tick_interval)


def _binance_ws_feed(
    price_queue: "queue.Queue[Optional[float]]",
    stop_event: threading.Event,
    fail_counter: "list[int]",
    last_tick_time: "list[float]",
) -> None:
    """Feed prices from Binance public WS into price_queue.  Runs in a thread."""
    try:
        import websocket  # type: ignore
    except ImportError:
        logger.warning("websocket-client not installed; skipping Binance WS")
        fail_counter[0] = _WS_FAIL_THRESHOLD + 1
        return

    def _on_message(ws: Any, message: str) -> None:
        try:
            data = json.loads(message)
            bid = float(data.get("b", 0))
            ask = float(data.get("a", 0))
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                price_queue.put(mid)
                last_tick_time[0] = time.time()
                fail_counter[0] = 0
        except Exception:
            pass

    def _on_error(ws: Any, error: Any) -> None:
        logger.warning("Binance WS error: %s", error)
        fail_counter[0] += 1

    def _on_close(ws: Any, *args: Any) -> None:
        logger.info("Binance WS closed")

    while not stop_event.is_set() and fail_counter[0] < _WS_FAIL_THRESHOLD:
        try:
            ws = websocket.WebSocketApp(
                _BINANCE_WS_URL,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.warning("Binance WS run_forever raised: %s", e)
            fail_counter[0] += 1
        if not stop_event.is_set():
            time.sleep(5)


# ---------------------------------------------------------------------------
# Drill statistics
# ---------------------------------------------------------------------------

@dataclass
class DrillStats:
    start_time: float = field(default_factory=time.time)
    tick_count: int = 0
    halt_count: int = 0
    resume_count: int = 0
    fault_count: int = 0
    max_halt_duration_s: float = 0.0
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)

    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "elapsed_hours": self.elapsed_hours(),
            "tick_count": self.tick_count,
            "halt_count": self.halt_count,
            "resume_count": self.resume_count,
            "fault_count": self.fault_count,
            "max_halt_duration_s": self.max_halt_duration_s,
        }


# ---------------------------------------------------------------------------
# AutonomousDrill
# ---------------------------------------------------------------------------

class AutonomousDrill:
    """Orchestrates the autonomous 72h paper drill.

    Parameters
    ----------
    config:
        Dict with keys: duration_hours, log_dir, feed, tick_interval.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._duration_hours: float = float(config.get("duration_hours", 72))
        self._log_dir = pathlib.Path(config.get("log_dir", "logs"))
        self._docs_dir = pathlib.Path(config.get("docs_dir", "docs/phase7"))
        self._force_feed: Optional[str] = config.get("feed", None)  # "ws"|"csv"|"gbm"
        self._tick_interval: float = float(config.get("tick_interval", 1.0))

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._fault_log = self._log_dir / "fault_injection.jsonl"
        self._incidents_dir = self._log_dir / "incidents"

        # Alerter + drift detector
        alerter_cfg: Dict[str, Any] = dict(config.get("monitoring", {}))
        alerter_cfg["log_dir"] = str(self._log_dir)
        self._alerter = TradingAlerter(alerter_cfg)
        self._drift_detector = DeploymentDriftDetector(
            config.get("alerts", {}),
            alerter=self._alerter,
        )

        # Fault injector (I2-b)
        self._fault_injector = FaultInjector(
            drill=self,
            log_path=self._fault_log,
            intervals=config.get("fault_intervals", None),
        )

        # Thread coordination
        self._stop_event = threading.Event()
        self._halt_event = threading.Event()      # set when trader should pause
        self._feed_pause_event = threading.Event()  # set by feed_stale fault
        self._price_queue: "queue.Queue[Optional[float]]" = queue.Queue(maxsize=100_000)

        # Feed tracking
        self._ws_fail_counter: list[int] = [0]
        self._ws_last_tick: list[float] = [time.time()]

        # State
        self._stats = DrillStats()
        self._portfolio_value: float = float(config.get("initial_capital", 10_000.0))
        self._peak_portfolio: float = self._portfolio_value
        self._current_price: float = 30_000.0
        self._position: float = 0.0  # units of BTC held

        # For fault injection hooks
        self._feed_stale_triggered: bool = False
        self._reconcile_drift_pct: float = 0.0

        # I11: fault injection flags
        self._fake_exchange_503: bool = False   # exchange_outage: submit raises 503
        self._fake_spread_multiplier: float = 1.0  # spread_blowout: fill price offset

        # I9-b: active feed type for snapshot reporting
        self._active_feed: str = "unknown"

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------

    def start_feed(self) -> None:
        """Launch price feed (WS → CSV → GBM priority)."""
        self._feed_thread = threading.Thread(
            target=self._run_feed, daemon=True, name="drill-feed"
        )
        self._feed_thread.start()

    def start_trader(self) -> None:
        """Launch simulated paper trader."""
        self._trader_thread = threading.Thread(
            target=self._run_trader, daemon=True, name="drill-trader"
        )
        self._trader_thread.start()

    def start_fuzzer(self) -> None:
        """Launch fault injector."""
        self._fault_injector.start()

    def start_observer(self) -> None:
        """Launch 15-min snapshot writer."""
        self._observer_thread = threading.Thread(
            target=self._run_observer, daemon=True, name="drill-observer"
        )
        self._observer_thread.start()

    def run(self, duration_hours: Optional[float] = None) -> DrillStats:
        """Block until duration_hours elapsed or stop() called."""
        if duration_hours is not None:
            self._duration_hours = duration_hours

        deadline = time.time() + self._duration_hours * 3600
        logger.info("AutonomousDrill starting | duration=%.2fh", self._duration_hours)

        self.start_feed()
        self.start_trader()
        self.start_fuzzer()
        self.start_observer()

        try:
            while not self._stop_event.is_set():
                if time.time() >= deadline:
                    logger.info("Drill duration reached — stopping")
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping drill")
        finally:
            self._stop_event.set()
            self._fault_injector.stop()

        return self._stats

    def finalize(self) -> pathlib.Path:
        """Write final report to docs/phase7/week85_72h_{start_date}.md."""
        start_date = datetime.utcfromtimestamp(self._stats.start_time).strftime("%Y%m%d")
        report_path = self._docs_dir / f"week85_72h_{start_date}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self._render_final_report(), encoding="utf-8")
        logger.info("Final report written to %s", report_path)
        return report_path

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Feed thread
    # ------------------------------------------------------------------

    def _run_feed(self) -> None:
        feed = self._select_feed()
        for price in feed:
            if self._stop_event.is_set():
                break
            # Feed pause (fault injection: feed_stale)
            if self._feed_pause_event.is_set():
                self._feed_stale_triggered = True
                logger.warning("Feed paused by fault injector")
                self._feed_pause_event.wait(timeout=15)
                self._feed_pause_event.clear()
                self._feed_stale_triggered = False
            self._price_queue.put(price)
        self._price_queue.put(None)  # sentinel

    def _select_feed(self) -> Generator[float, None, None]:
        if self._force_feed == "csv":
            self._active_feed = "csv"
            return self._fallback_feed()
        if self._force_feed == "gbm":
            self._active_feed = "gbm"
            return self._fallback_feed()
        if self._force_feed == "ws":
            self._active_feed = "ws"
            return self._ws_to_generator()

        # Auto: try WS, fall back after failures
        try:
            import websocket  # type: ignore  # noqa: F401
            self._active_feed = "ws"
            return self._ws_to_generator()
        except ImportError:
            logger.info("websocket-client not available — using fallback feed")
            self._active_feed = "csv"
            return self._fallback_feed()

    def _ws_to_generator(self) -> Generator[float, None, None]:
        ws_thread = threading.Thread(
            target=_binance_ws_feed,
            args=(self._price_queue, self._stop_event, self._ws_fail_counter, self._ws_last_tick),
            daemon=True,
            name="drill-ws",
        )
        ws_thread.start()

        # Wait for first tick or give up and switch to fallback
        deadline = time.time() + 30
        while time.time() < deadline and self._ws_fail_counter[0] < _WS_FAIL_THRESHOLD:
            try:
                price = self._price_queue.get(timeout=5)
                if price is not None:
                    yield price
                    break
            except queue.Empty:
                pass
        else:
            logger.warning("Binance WS failed — switching to fallback")
            self._on_feed_fallback("ws_initial_fail")
            yield from self._fallback_feed()
            return

        # Consume from WS queue until stop or heartbeat timeout
        while not self._stop_event.is_set():
            elapsed_since_tick = time.time() - self._ws_last_tick[0]
            if (
                self._ws_fail_counter[0] >= _WS_FAIL_THRESHOLD
                or elapsed_since_tick > _WS_HEARTBEAT_TIMEOUT
            ):
                logger.warning("WS fallback triggered: fails=%d stale=%.0fs",
                               self._ws_fail_counter[0], elapsed_since_tick)
                self._on_feed_fallback("ws_mid_run_fail")
                yield from self._fallback_feed()
                return
            try:
                price = self._price_queue.get(timeout=1)
                if price is None:
                    break
                yield price
            except queue.Empty:
                pass

    def _on_feed_fallback(self, reason: str) -> None:
        """Record WS→fallback transition: update active_feed and log incident."""
        csv_path = _CSV_DEFAULT
        new_feed = "csv" if csv_path.exists() else "gbm"
        self._active_feed = new_feed
        incident = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "feed_fallback",
            "detail": {"reason": reason, "new_feed": new_feed},
        }
        self._stats.incidents.append(incident)
        self._alerter.send_alert(
            f"Feed fallback: {reason} → switched to {new_feed}", level="WARNING"
        )

    def _fallback_feed(self) -> Generator[float, None, None]:
        if self._force_feed == "gbm":
            self._active_feed = "gbm"
            logger.info("Using GBM feed")
            yield from _gbm_feed()
            return
        csv_path = _CSV_DEFAULT
        if csv_path.exists():
            self._active_feed = "csv"
            logger.info("Using CSV replay feed: %s", csv_path)
            yield from _csv_replay_feed(csv_path, tick_interval=self._tick_interval)
        else:
            self._active_feed = "gbm"
            logger.info("CSV not found — using GBM synthetic feed")
            yield from _gbm_feed()

    # ------------------------------------------------------------------
    # Trader thread (simplified paper simulation)
    # ------------------------------------------------------------------

    def _run_trader(self) -> None:
        """Minimal paper simulation; checks safety nets each tick."""
        while not self._stop_event.is_set():
            try:
                price = self._price_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if price is None:
                break

            self._current_price = price
            self._stats.tick_count += 1

            # Simple random action: hold/buy/sell with equal prob
            action = random.choice([-1, 0, 1])
            self._execute_action(action, price)

            # Safety net checks
            self._check_drawdown()
            self._check_drift_halt()

            # Handle halt
            if self._halt_event.is_set():
                halt_start = time.time()
                logger.warning("Trader halted — auto-resume in 30s")
                self._alerter.send_alert("Drill trader halted — auto-resume in 30s", level="WARNING")
                self._stats.halt_count += 1
                resumed = self._halt_event.wait(timeout=30)
                duration = time.time() - halt_start
                self._stats.max_halt_duration_s = max(self._stats.max_halt_duration_s, duration)
                self._halt_event.clear()
                self._drift_detector.reset_halt(source="auto_drill")
                self._stats.resume_count += 1
                logger.info("Trader auto-resumed after %.1fs", duration)

    def _execute_action(self, action: int, price: float) -> None:
        trade_size = 0.001  # 0.001 BTC per trade

        # I11: exchange_outage — simulated 503; skip this tick's order submission
        if self._fake_exchange_503:
            logger.warning("Exchange 503 simulated — order skipped this tick")
            return

        # I11: spread_blowout — widen effective fill price
        effective_price = price
        if self._fake_spread_multiplier != 1.0:
            base_spread = price * 0.0005  # 0.05% default half-spread
            slippage = base_spread * self._fake_spread_multiplier
            if action == 1:
                effective_price = price + slippage
            elif action == -1:
                effective_price = price - slippage

        if action == 1 and self._portfolio_value > effective_price * trade_size:
            self._position += trade_size
            self._portfolio_value -= effective_price * trade_size
        elif action == -1 and self._position >= trade_size:
            self._position -= trade_size
            self._portfolio_value += effective_price * trade_size
        total_value = self._portfolio_value + self._position * price
        if total_value > self._peak_portfolio:
            self._peak_portfolio = total_value

    def _check_drawdown(self) -> None:
        total = self._portfolio_value + self._position * self._current_price
        if self._peak_portfolio > 0:
            dd = (self._peak_portfolio - total) / self._peak_portfolio
            if dd >= 0.15:  # 15% drawdown → alert
                self._alerter.check_drawdown(current=total, peak=self._peak_portfolio)

    def _check_drift_halt(self) -> None:
        if self._drift_detector.halt_requested:
            self._halt_event.set()

    # ------------------------------------------------------------------
    # Fault injection hooks (called by FaultInjector)
    # ------------------------------------------------------------------

    def _pause_feed(self, seconds: float) -> None:
        self._feed_pause_event.set()
        time.sleep(seconds)
        self._feed_pause_event.clear()

    def _inject_reconciliation_drift(self, qty_drift_pct: float) -> None:
        self._alerter.notify_reconciliation_drift(
            [{"type": "qty_mismatch", "drift_pct": qty_drift_pct}]
        )
        self._stats.fault_count += 1
        self._record_incident("reconciliation_mismatch", {"qty_drift_pct": qty_drift_pct})

    def _inject_schema_drift(self, column_name: str) -> None:
        self._drift_detector.report_schema_drift(column_name, on_drift="halt")
        self._stats.fault_count += 1
        self._record_incident("schema_drift", {"column": column_name})

    def _inject_canary_underperform(self, sigma_below: float) -> None:
        self._alerter.notify_canary_auto_demoted(
            version=0,
            sigma_below=sigma_below,
            consecutive_hours=6,
            canary_mean=-sigma_below * 0.001,
            prod_mean=0.0,
            prod_std=0.001,
        )
        self._stats.fault_count += 1
        self._record_incident("canary_underperform", {"sigma_below": sigma_below})

    def _inject_clock_skew(self, skew_seconds: float) -> None:
        logger.warning("Clock skew +%.0fs injected (no action taken — event logged)", skew_seconds)
        self._stats.fault_count += 1
        self._record_incident("clock_skew", {"skew_seconds": skew_seconds})

    def _record_incident(self, incident_type: str, detail: Dict[str, Any]) -> None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        incident = {
            "ts": ts,
            "type": incident_type,
            "detail": detail,
            "elapsed_hours": self._stats.elapsed_hours(),
        }
        self._stats.incidents.append(incident)
        self._incidents_dir.mkdir(parents=True, exist_ok=True)
        inc_path = self._incidents_dir / f"{ts}_{incident_type}.md"
        inc_path.write_text(
            f"# Incident: {incident_type}\n\n"
            f"**Time**: {ts}  \n"
            f"**Elapsed hours**: {incident['elapsed_hours']:.2f}  \n\n"
            f"## Detail\n\n```json\n{json.dumps(detail, indent=2)}\n```\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Observer thread (15-min snapshots)
    # ------------------------------------------------------------------

    def _run_observer(self) -> None:
        snapshot_interval = 15 * 60  # 15 minutes
        last_snapshot = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now - last_snapshot >= snapshot_interval:
                self._write_snapshot()
                last_snapshot = now
            time.sleep(5)
        # Final snapshot on stop
        self._write_snapshot()

    def _write_snapshot(self) -> None:
        total_value = self._portfolio_value + self._position * self._current_price
        snapshot = {
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_hours": self._stats.elapsed_hours(),
            "tick_count": self._stats.tick_count,
            "portfolio_value": round(total_value, 4),
            "position_btc": round(self._position, 6),
            "current_price": round(self._current_price, 2),
            "halt_count": self._stats.halt_count,
            "fault_count": self._stats.fault_count,
            "in_shadow_mode": self._drift_detector.in_shadow_mode,
            "active_feed": self._active_feed,
        }
        self._stats.snapshots.append(snapshot)
        snap_path = self._log_dir / "drill_snapshots.jsonl"
        with snap_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snapshot) + "\n")
        logger.info(
            "Snapshot: elapsed=%.2fh ticks=%d portfolio=%.2f halts=%d faults=%d",
            snapshot["elapsed_hours"],
            snapshot["tick_count"],
            snapshot["portfolio_value"],
            snapshot["halt_count"],
            snapshot["fault_count"],
        )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    def _render_final_report(self) -> str:
        stats = self._stats.to_dict()
        fault_history = self._fault_injector.history
        fault_summary = {}
        for evt in fault_history:
            fault_summary.setdefault(evt.fault_type, 0)
            fault_summary[evt.fault_type] += 1

        lines = [
            f"# 72h Autonomous Drill — Final Report",
            "",
            f"**Start**: {datetime.utcfromtimestamp(stats['start_time']).strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Duration**: {stats['elapsed_hours']:.2f}h  ",
            f"**Feed**: {self._force_feed or 'auto'}  ",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total ticks | {stats['tick_count']} |",
            f"| Safety net halts | {stats['halt_count']} |",
            f"| Auto-resumes | {stats['resume_count']} |",
            f"| Faults injected | {stats['fault_count']} |",
            f"| Max halt duration | {stats['max_halt_duration_s']:.1f}s |",
            "",
            "## Fault Injection Summary",
            "",
            "| Fault Type | Count |",
            "|------------|-------|",
        ]
        if fault_summary:
            for fault_type, count in fault_summary.items():
                lines.append(f"| {fault_type} | {count} |")
        else:
            lines.append("| _no faults injected during run_ | — |")

        if self._stats.snapshots:
            lines += [
                "",
                "## Observation Table (15-min snapshots)",
                "",
                "| Time (elapsed h) | Portfolio Value | Price | Halts | Faults |",
                "|-----------------|----------------|-------|-------|--------|",
            ]
            for s in self._stats.snapshots[-20:]:  # last 20 snapshots
                lines.append(
                    f"| {s['elapsed_hours']:.2f} | {s['portfolio_value']:.2f} | "
                    f"{s['current_price']:.2f} | {s['halt_count']} | {s['fault_count']} |"
                )

        if self._stats.incidents:
            lines += [
                "",
                "## Incidents",
                "",
            ]
            for inc in self._stats.incidents:
                lines.append(f"- **{inc['ts']}** `{inc['type']}` — {json.dumps(inc['detail'])}")

        lines += [
            "",
            "---",
            "",
            "## Sign-off",
            "",
            "- [ ] Operator reviewed all incidents",
            "- [ ] Drift calibration run: `python scripts/analyze_drift_baseline.py`",
            "- [ ] No kill-switch fires during shadow mode",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Autonomous 72h paper drill (Phase 7.6 I2)")
    parser.add_argument(
        "--duration-hours", type=float, default=72.0,
        help="Drill duration in hours (default: 72)"
    )
    parser.add_argument(
        "--feed", choices=["ws", "csv", "gbm"], default=None,
        help="Force a specific feed (default: auto WS→CSV→GBM)"
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Log output directory (default: logs)"
    )
    parser.add_argument(
        "--docs-dir", default="docs/phase7",
        help="Docs output directory for final report (default: docs/phase7)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Initial paper capital in USD (default: 10000)"
    )
    parser.add_argument(
        "--tick-interval", type=float, default=1.0,
        help="Tick interval in seconds for CSV/GBM feed (default: 1.0)"
    )
    args = parser.parse_args(argv)

    config: Dict[str, Any] = {
        "duration_hours": args.duration_hours,
        "feed": args.feed,
        "log_dir": args.log_dir,
        "docs_dir": args.docs_dir,
        "initial_capital": args.capital,
        "tick_interval": args.tick_interval,
        "monitoring": {"alert_channels": ["console", "file"]},
        "alerts": {},
    }

    drill = AutonomousDrill(config)
    stats = drill.run()
    report_path = drill.finalize()

    print(f"\n--- Drill complete ---")
    print(f"Duration:  {stats.elapsed_hours():.2f}h")
    print(f"Ticks:     {stats.tick_count}")
    print(f"Halts:     {stats.halt_count} (max {stats.max_halt_duration_s:.1f}s)")
    print(f"Faults:    {stats.fault_count}")
    print(f"Report:    {report_path}")


if __name__ == "__main__":
    main()
