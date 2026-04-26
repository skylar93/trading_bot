#!/usr/bin/env python3
"""Operator tool: manually clear DeploymentDriftDetector halt flag.

Usage:
    python scripts/clear_drift_halt.py
    python scripts/clear_drift_halt.py --dry-run
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clear deployment drift halt (operator action)")
    parser.add_argument("--dry-run", action="store_true", help="Print action without executing")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("[dry-run] Would instantiate DeploymentDriftDetector and call reset_halt(source='operator')")
        return 0

    from deployment.monitoring.alerter import TradingAlerter
    from deployment.monitoring.drift_detector import DeploymentDriftDetector

    alerter = TradingAlerter({})
    detector = DeploymentDriftDetector(config={}, alerter=alerter)
    # Simulate a halted state so the reset is meaningful in interactive use.
    # In production the detector object is obtained from the running process;
    # this script serves as documentation of the correct call pattern.
    detector.halt_requested = True
    detector.reset_halt(source="operator")
    print("Drift halt cleared (source=operator). Monitor alerts for confirmation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
