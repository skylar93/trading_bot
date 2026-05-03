#!/usr/bin/env python
"""
Thin wrapper around deployment/paper_trader.PaperTrader with:
  - PID file management (state/paper_trader.pid by default)
  - SIGTERM / SIGINT → graceful shutdown via _trigger_shutdown()
  - --duration-hours and --exchange-mode CLI flags
  - Final report written to logs/paper_trader_{start_ts}.json

Usage:
    python scripts/run_paper_trader.py \\
        --config config/local_3060ti.yaml \\
        --exchange-mode sandbox \\
        --duration-hours 72

Exit codes:
    0 — clean shutdown (duration elapsed or SIGTERM)
    1 — startup error (bad config / missing checkpoint / import error)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger("run_paper_trader")


# ---------------------------------------------------------------------------
# PID helpers (public for unit-testing)
# ---------------------------------------------------------------------------

def _write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    logger.info("PID %d written to %s", os.getpid(), pid_file)


def _remove_pid(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaperTrader runner with PID management and duration-hours support"
    )
    parser.add_argument("--config", required=True,
                        help="Path to paper_trading YAML config")
    parser.add_argument(
        "--exchange-mode",
        choices=["paper", "sandbox", "live"],
        default="paper",
        help="Override config execution.exchange_mode (default: paper)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        metavar="HOURS",
        help="Run duration in hours (omit for indefinite)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "logs"),
        help="Directory for run logs and final report (default: logs/)",
    )
    parser.add_argument(
        "--pid-file",
        default=str(PROJECT_ROOT / "state" / "paper_trader.pid"),
        help="PID file path (default: state/paper_trader.pid)",
    )
    args = parser.parse_args()

    # ── Config loading ───────────────────────────────────────────────────────
    try:
        import yaml  # type: ignore
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to load config %s: %s", args.config, exc)
        sys.exit(1)

    # Apply exchange-mode override
    config.setdefault("execution", {})["exchange_mode"] = args.exchange_mode
    config.setdefault("paper_trading", {})["exchange_mode"] = args.exchange_mode

    # ── [A0.5] Live signal gate — block live mode until evidence passes ──────
    if args.exchange_mode == "live":
        try:
            from deployment.governance.live_signal_gate import LiveSignalGate
            gate_cfg = config.get("live_signal_gate", {})
            evidence_pack = Path(
                gate_cfg.get("evidence_pack", "docs/phase8/strategy_evidence_v1.md")
            )
            max_age = float(gate_cfg.get("max_evidence_age_days", 30))
            thresholds = gate_cfg.get("thresholds", {})
            gate = LiveSignalGate(
                evidence_pack=evidence_pack,
                thresholds=thresholds,
                max_evidence_age_days=max_age,
            )
            result = gate.check()
            if not result.passed:
                logger.error(
                    "❌ Live signal gate FAILED — live mode blocked:\n%s",
                    "\n".join(f"  • {f}" for f in result.failures),
                )
                sys.exit(2)
            logger.info(
                "✅ Live signal gate PASSED (evidence age %.1f days)",
                result.evidence_pack_age_days,
            )
        except ImportError as exc:
            logger.error("Failed to import live_signal_gate: %s", exc)
            sys.exit(1)

    # ── [E2/E3] Warmup / live-ramp guard ────────────────────────────────────
    warmup_cfg = config.setdefault("warmup", {})
    if warmup_cfg.get("enabled", True):
        warmup_cfg["enabled"] = True
        warmup_cfg.setdefault("warmup_minutes", 30)
        warmup_cfg.setdefault("max_qps", 1.0)
        if args.exchange_mode == "live":
            # E3: tighter size cap + 1-minute progress alerts
            warmup_cfg.setdefault("size_fraction", 0.3)
            warmup_cfg.setdefault("progress_alerts", True)
        else:
            # E2: half-size, no progress spam
            warmup_cfg.setdefault("size_fraction", 0.5)
            warmup_cfg.setdefault("progress_alerts", False)

    duration_seconds: Optional[float] = (
        args.duration_hours * 3600.0 if args.duration_hours is not None else None
    )

    # ── Logging to file ──────────────────────────────────────────────────────
    pid_file = Path(args.pid_file)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    start_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"paper_trader_{start_ts}.log"

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(fh)

    # ── Agent loading ────────────────────────────────────────────────────────
    agent_cfg = config.get("agent", {})
    checkpoint = agent_cfg.get("checkpoint")
    algo = agent_cfg.get("algo", "PPO").upper()

    if not checkpoint:
        logger.error("agent.checkpoint must be set in config")
        sys.exit(1)

    try:
        from stable_baselines3 import PPO, SAC, TD3  # type: ignore
        algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
        agent = algo_map[algo].load(checkpoint)
    except Exception as exc:
        logger.error("Failed to load agent checkpoint %s: %s", checkpoint, exc)
        sys.exit(1)

    # ── PaperTrader setup ────────────────────────────────────────────────────
    from deployment.paper_trader import PaperTrader

    simulation_mode = args.exchange_mode == "paper"
    trader = PaperTrader(agent, config, simulation_mode=simulation_mode)

    # ── PID file + signal handlers ───────────────────────────────────────────
    _write_pid(pid_file)

    def _handle_signal(sig, frame):  # type: ignore[no-untyped-def]
        logger.info("Signal %s received — requesting graceful shutdown", sig)
        trader._trigger_shutdown(f"signal {sig}")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "Starting PaperTrader | exchange_mode=%s duration=%s",
        args.exchange_mode,
        f"{args.duration_hours}h" if args.duration_hours is not None else "indefinite",
    )

    # ── Run ──────────────────────────────────────────────────────────────────
    report: dict = {}
    try:
        report = trader.run(duration_seconds=duration_seconds)
    except Exception as exc:
        logger.error("PaperTrader run error: %s", exc, exc_info=True)
    finally:
        _remove_pid(pid_file)
        report_path = log_dir / f"paper_trader_{start_ts}.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Final report written to %s", report_path)
        fh.close()
        logging.getLogger().removeHandler(fh)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
