#!/usr/bin/env python
"""
Compare backtesting results with paper trading results.

Usage:
    python scripts/reconcile.py --backtest results/backtest_report.json --live results/paper_report.json
    python scripts/reconcile.py --backtest mlruns/... --live mlruns/...
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Reconcile backtest vs live results")
    parser.add_argument("--backtest", required=True, help="Path to backtest report JSON")
    parser.add_argument("--live", required=True, help="Path to live/paper trading report JSON")
    parser.add_argument("--output", default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    with open(args.backtest) as f:
        bt_report = json.load(f)
    with open(args.live) as f:
        lv_report = json.load(f)

    from training.analysis.reconciliation import ReconciliationReport

    report = ReconciliationReport.from_reports(bt_report, lv_report)
    print(report.summary())

    if args.output:
        report.to_json(args.output)
        print(f"\nJSON saved to {args.output}")

    # Exit code: 1 if warnings, 0 if clean
    return 1 if report.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
