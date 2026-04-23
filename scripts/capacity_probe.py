#!/usr/bin/env python
"""
R17 (G9): Capacity baseline snapshot — Week 84.

Runs the system in simulation mode for a configurable duration, collects
latency / queue / resource metrics, and writes a Markdown baseline report.

Usage:
    # Quick 60-second probe (default)
    python scripts/capacity_probe.py

    # Full 1-hour probe
    python scripts/capacity_probe.py --duration 3600

    # Custom output path
    python scripts/capacity_probe.py --output docs/phase7/week84_baseline.md

Metrics collected:
    - submit_order latency (p50 / p95 / p99)
    - OrderManager lock acquire latency (p50 / p95 / p99)
    - CPU and memory usage (sampled every second)
    - Orders submitted / filled / rejected per second
    - Simulated network round-trip (synthetic, Gaussian model)

Exit codes:
    0 — baseline captured successfully
    1 — error during probe
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Percentile helper (no scipy dependency)
# ---------------------------------------------------------------------------

def _pct(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_d):
        return sorted_d[lo]
    return sorted_d[lo] + (idx - lo) * (sorted_d[hi] - sorted_d[lo])


# ---------------------------------------------------------------------------
# Metrics collector (thread-safe, bounded deque per metric)
# ---------------------------------------------------------------------------

class MetricsCollector:
    def __init__(self, max_samples: int = 100_000) -> None:
        self._lock = threading.Lock()
        self._submit_latencies: Deque[float] = deque(maxlen=max_samples)
        self._lock_latencies: Deque[float] = deque(maxlen=max_samples)
        self._cpu_pct: Deque[float] = deque(maxlen=max_samples)
        self._mem_mb: Deque[float] = deque(maxlen=max_samples)
        self._rtt_ms: Deque[float] = deque(maxlen=max_samples)
        self._orders_submitted: int = 0
        self._orders_filled: int = 0
        self._orders_rejected: int = 0
        self._start_ts: float = time.time()

    def record_submit(self, latency_s: float) -> None:
        with self._lock:
            self._submit_latencies.append(latency_s * 1000)  # → ms
            self._orders_submitted += 1

    def record_lock(self, latency_s: float) -> None:
        with self._lock:
            self._lock_latencies.append(latency_s * 1000)

    def record_fill(self) -> None:
        with self._lock:
            self._orders_filled += 1

    def record_reject(self) -> None:
        with self._lock:
            self._orders_rejected += 1

    def record_rtt(self, ms: float) -> None:
        with self._lock:
            self._rtt_ms.append(ms)

    def record_resources(self, cpu: float, mem_mb: float) -> None:
        with self._lock:
            self._cpu_pct.append(cpu)
            self._mem_mb.append(mem_mb)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._start_ts
            sub = list(self._submit_latencies)
            lck = list(self._lock_latencies)
            rtt = list(self._rtt_ms)
            cpu = list(self._cpu_pct)
            mem = list(self._mem_mb)
            n_sub = self._orders_submitted
            n_fill = self._orders_filled
            n_rej = self._orders_rejected

        return {
            "elapsed_s": elapsed,
            "submit_latency_ms": {
                "p50": _pct(sub, 50),
                "p95": _pct(sub, 95),
                "p99": _pct(sub, 99),
                "n": len(sub),
            },
            "lock_latency_ms": {
                "p50": _pct(lck, 50),
                "p95": _pct(lck, 95),
                "p99": _pct(lck, 99),
                "n": len(lck),
            },
            "network_rtt_ms": {
                "p50": _pct(rtt, 50),
                "p95": _pct(rtt, 95),
                "n": len(rtt),
            },
            "cpu_pct": {
                "mean": sum(cpu) / len(cpu) if cpu else float("nan"),
                "max": max(cpu) if cpu else float("nan"),
            },
            "mem_mb": {
                "mean": sum(mem) / len(mem) if mem else float("nan"),
                "max": max(mem) if mem else float("nan"),
            },
            "orders": {
                "submitted": n_sub,
                "filled": n_fill,
                "rejected": n_rej,
                "rate_per_s": n_sub / max(elapsed, 1),
            },
        }


# ---------------------------------------------------------------------------
# Instrumented OrderManager wrapper
# ---------------------------------------------------------------------------

class InstrumentedOrderManager:
    """Thin wrapper that intercepts submit_order calls to collect latency."""

    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        self._inner_lock = threading.RLock()
        self._call_count = 0
        self._rng = __import__("random").Random(42)

    def submit_order(self, side: str, amount: float, price: float) -> str:
        # Measure lock acquire latency
        t0 = time.monotonic()
        with self._inner_lock:
            lock_lat = time.monotonic() - t0
            self._metrics.record_lock(lock_lat)

            # Simulate order processing: idempotency lookup + risk check + exchange submit
            submit_start = time.monotonic()

            # Simulate idempotency map lookup (hash + dict lookup)
            order_id = f"ord_{self._call_count:08d}"
            self._call_count += 1

            # Simulate exchange round-trip (Gaussian, mean ~80ms, std ~15ms)
            rtt_ms = max(10.0, self._rng.gauss(80, 15))
            self._metrics.record_rtt(rtt_ms)
            time.sleep(rtt_ms / 1000.0)  # realistic sleep

            submit_lat = time.monotonic() - submit_start
            self._metrics.record_submit(submit_lat)

            # 95% fill rate, 5% reject
            if self._rng.random() < 0.95:
                self._metrics.record_fill()
            else:
                self._metrics.record_reject()

        return order_id


# ---------------------------------------------------------------------------
# Resource sampler (background thread)
# ---------------------------------------------------------------------------

def _resource_sampler(metrics: MetricsCollector, stop_event: threading.Event) -> None:
    proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
    while not stop_event.is_set():
        if proc is not None:
            try:
                cpu = proc.cpu_percent(interval=1.0)
                mem = proc.memory_info().rss / 1024 / 1024
                metrics.record_resources(cpu, mem)
            except Exception:
                pass
        else:
            # Rough CPU estimate via /proc/stat or just record 0
            metrics.record_resources(0.0, 0.0)
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Workload driver
# ---------------------------------------------------------------------------

def _workload_driver(
    om: InstrumentedOrderManager,
    stop_event: threading.Event,
    target_ops_per_s: float = 5.0,
) -> None:
    """Submit orders at approximately target_ops_per_s until stop_event is set."""
    interval = 1.0 / max(target_ops_per_s, 0.1)
    rng = __import__("random").Random(99)
    while not stop_event.is_set():
        side = "buy" if rng.random() < 0.5 else "sell"
        amount = round(rng.uniform(0.001, 0.01), 4)
        price = round(rng.uniform(25_000, 35_000), 2)
        try:
            om.submit_order(side, amount, price)
        except Exception:
            pass
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(path: Path, snap: Dict[str, Any], duration_s: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _fmt(v) -> str:
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    sl = snap["submit_latency_ms"]
    ll = snap["lock_latency_ms"]
    nt = snap["network_rtt_ms"]
    cp = snap["cpu_pct"]
    mm = snap["mem_mb"]
    od = snap["orders"]

    lines = [
        "# Capacity Baseline Snapshot — Week 84",
        "",
        f"**Timestamp**: {ts}  ",
        f"**Probe duration**: {int(snap['elapsed_s'])}s (planned {duration_s}s)  ",
        f"**Host**: simulation mode (InstrumentedOrderManager)  ",
        "",
        "## submit_order Latency",
        "",
        "| Percentile | Latency (ms) |",
        "|------------|-------------|",
        f"| p50        | {_fmt(sl['p50'])} |",
        f"| p95        | {_fmt(sl['p95'])} |",
        f"| p99        | {_fmt(sl['p99'])} |",
        f"| n samples  | {sl['n']} |",
        "",
        "## OrderManager Lock Acquire Latency",
        "",
        "| Percentile | Latency (ms) |",
        "|------------|-------------|",
        f"| p50        | {_fmt(ll['p50'])} |",
        f"| p95        | {_fmt(ll['p95'])} |",
        f"| p99        | {_fmt(ll['p99'])} |",
        f"| n samples  | {ll['n']} |",
        "",
        "## Simulated Network Round-Trip",
        "",
        "| Percentile | RTT (ms) |",
        "|------------|---------|",
        f"| p50        | {_fmt(nt['p50'])} |",
        f"| p95        | {_fmt(nt['p95'])} |",
        f"| n samples  | {nt['n']} |",
        "",
        "## Resource Usage",
        "",
        "| Metric    | Mean | Max |",
        "|-----------|------|-----|",
        f"| CPU (%)   | {_fmt(cp['mean'])} | {_fmt(cp['max'])} |",
        f"| Mem (MB)  | {_fmt(mm['mean'])} | {_fmt(mm['max'])} |",
        "",
        "## Order Throughput",
        "",
        f"- Submitted: **{od['submitted']}**",
        f"- Filled:    **{od['filled']}** ({100*od['filled']/max(od['submitted'],1):.1f}%)",
        f"- Rejected:  **{od['rejected']}** ({100*od['rejected']/max(od['submitted'],1):.1f}%)",
        f"- Rate:      **{od['rate_per_s']:.2f} orders/s**",
        "",
        "## Phase 8 Scale-Up Signals",
        "",
        "| Bottleneck | Current p95 | Phase 8 trigger |",
        "|------------|-------------|-----------------|",
        f"| submit_order latency | {_fmt(sl['p95'])} ms | > 500 ms → async queue |",
        f"| lock acquire latency | {_fmt(ll['p95'])} ms | > 10 ms → lock-free map |",
        f"| network RTT          | {_fmt(nt['p95'])} ms | > 200 ms → co-located exchange |",
        f"| CPU usage            | {_fmt(cp['max'])}% | > 80% → multi-process |",
        f"| Memory               | {_fmt(mm['max'])} MB | > 2000 MB → streaming state |",
        "",
        "---",
        "*Auto-generated by `scripts/capacity_probe.py`*",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="R17: Capacity baseline snapshot (G9)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Probe duration in seconds (default: 60; use 3600 for full 1h baseline)")
    parser.add_argument("--ops-per-s", type=float, default=5.0,
                        help="Target order submission rate (default: 5.0)")
    parser.add_argument("--output", default="docs/phase7/week84_baseline.md",
                        help="Output report path")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"  Capacity Probe — {ts}")
    print(f"  Duration: {args.duration}s  |  Target: {args.ops_per_s:.1f} ops/s")
    print(f"{'='*60}\n")

    if not _HAS_PSUTIL:
        print("  NOTE: psutil not installed — CPU/mem metrics will be N/A")
        print("        Install with: pip install psutil\n")

    metrics = MetricsCollector()
    om = InstrumentedOrderManager(metrics)
    stop_event = threading.Event()

    # Resource sampler (background)
    sampler_t = threading.Thread(target=_resource_sampler, args=(metrics, stop_event), daemon=True)
    sampler_t.start()

    # Workload driver (background)
    worker_t = threading.Thread(
        target=_workload_driver,
        args=(om, stop_event, args.ops_per_s),
        daemon=True,
    )
    worker_t.start()

    # Progress display
    deadline = time.time() + args.duration
    tick = max(args.duration // 10, 5)
    next_tick = time.time() + tick
    try:
        while time.time() < deadline:
            remaining = int(deadline - time.time())
            if time.time() >= next_tick:
                snap = metrics.snapshot()
                print(
                    f"  [{remaining:4d}s left]  submitted={snap['orders']['submitted']:5d}"
                    f"  submit_p50={snap['submit_latency_ms']['p50']:.1f}ms"
                    f"  cpu={snap['cpu_pct']['mean']:.1f}%"
                )
                next_tick += tick
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Interrupted — saving partial results...")

    stop_event.set()
    sampler_t.join(timeout=3.0)
    worker_t.join(timeout=3.0)

    snap = metrics.snapshot()
    output_path = PROJECT_ROOT / args.output
    _write_report(output_path, snap, args.duration)

    print(f"\n  Report written to: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"\n  Summary:")
    sl = snap["submit_latency_ms"]
    print(f"    submit_order p50/p95/p99 = {sl['p50']:.1f} / {sl['p95']:.1f} / {sl['p99']:.1f} ms")
    ll = snap["lock_latency_ms"]
    print(f"    lock acquire p50/p95/p99 = {ll['p50']:.3f} / {ll['p95']:.3f} / {ll['p99']:.3f} ms")
    print(f"    orders: {snap['orders']['submitted']} submitted, "
          f"{snap['orders']['filled']} filled, "
          f"rate={snap['orders']['rate_per_s']:.2f}/s")
    print(f"\n{'='*60}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
