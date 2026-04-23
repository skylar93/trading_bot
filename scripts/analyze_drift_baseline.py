#!/usr/bin/env python3
"""
Phase 7.6 I4-c: Drift Baseline Analyzer

Reads drift events from logs/alerts.jsonl (and optionally Prometheus snapshots)
and proposes calibrated thresholds based on the observed distribution.

IMPORTANT: This script only *proposes* thresholds — never auto-applies them.
The operator must review docs/phase7.6/drift_calibration_<date>.md and
manually update config/alerts.yaml.

Usage:
    python scripts/analyze_drift_baseline.py
    python scripts/analyze_drift_baseline.py --alerts-log logs/alerts.jsonl --output docs/phase7.6/drift_calibration.md
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def _load_drift_events(alerts_log: pathlib.Path) -> List[Dict[str, Any]]:
    if not alerts_log.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in alerts_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") in ("drift_detected", "schema_drift", "reconciliation_drift"):
            events.append(record)
    return events


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def _analyze(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    drift_events = [e for e in events if e.get("event") == "drift_detected"]
    schema_events = [e for e in events if e.get("event") == "schema_drift"]
    reconcile_events = [e for e in events if e.get("event") == "reconciliation_drift"]

    result: Dict[str, Any] = {
        "total_events": len(events),
        "drift_events": len(drift_events),
        "schema_events": len(schema_events),
        "reconcile_events": len(reconcile_events),
        "proposed_thresholds": {},
    }

    # Propose reward_return_sigma_threshold based on drift frequency
    if len(drift_events) >= 10:
        # If many drift events relative to total, loosen threshold
        drift_rate = len(drift_events) / max(len(events), 1)
        if drift_rate > 0.3:
            result["proposed_thresholds"]["reward_return_sigma_threshold"] = {
                "current": 2.0,
                "proposed": 3.0,
                "reason": f"High drift rate ({drift_rate:.1%}) suggests threshold too tight",
            }
        else:
            result["proposed_thresholds"]["reward_return_sigma_threshold"] = {
                "current": 2.0,
                "proposed": 2.0,
                "reason": "Drift rate nominal — no change needed",
            }
    else:
        result["proposed_thresholds"]["reward_return_sigma_threshold"] = {
            "current": 2.0,
            "proposed": None,
            "reason": f"Insufficient samples ({len(drift_events)} < 10) — retain prior",
        }

    return result


def _render_report(
    analysis: Dict[str, Any],
    alerts_log: pathlib.Path,
    run_date: str,
) -> str:
    lines = [
        f"# Drift Calibration Report — {run_date}",
        "",
        f"**Source**: `{alerts_log}`  ",
        f"**Generated**: {run_date}  ",
        "**Action required**: Review and apply proposed changes to `config/alerts.yaml` manually.",
        "",
        "---",
        "",
        "## Event Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total alert events | {analysis['total_events']} |",
        f"| Drift detected events | {analysis['drift_events']} |",
        f"| Schema drift events | {analysis['schema_events']} |",
        f"| Reconciliation drift events | {analysis['reconcile_events']} |",
        "",
        "---",
        "",
        "## Proposed Threshold Changes",
        "",
    ]

    proposed = analysis["proposed_thresholds"]
    if not proposed:
        lines.append("No threshold changes proposed (insufficient data).")
    else:
        lines.append("| Threshold | Current | Proposed | Reason |")
        lines.append("|-----------|---------|----------|--------|")
        for key, info in proposed.items():
            proposed_val = info["proposed"] if info["proposed"] is not None else "*(retain)*"
            lines.append(f"| `{key}` | {info['current']} | {proposed_val} | {info['reason']} |")

    lines += [
        "",
        "---",
        "",
        "## Instructions",
        "",
        "1. Review each proposed change above.",
        "2. If accepted, edit `config/alerts.yaml` under the `drift:` section.",
        "3. Re-run this script after 500+ additional samples to re-evaluate.",
        "",
        "> Auto-application is intentionally disabled. Human sign-off required.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze drift baseline and propose thresholds.")
    parser.add_argument(
        "--alerts-log",
        default="logs/alerts.jsonl",
        help="Path to alerts.jsonl (default: logs/alerts.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: docs/phase7.6/drift_calibration_<date>.md)",
    )
    args = parser.parse_args(argv)

    alerts_log = pathlib.Path(args.alerts_log)
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        output_path = pathlib.Path(f"docs/phase7.6/drift_calibration_{run_date}.md")

    events = _load_drift_events(alerts_log)
    if not events:
        print(f"No drift events found in {alerts_log}. Run a 72h drill first.", file=sys.stderr)

    analysis = _analyze(events)
    report = _render_report(analysis, alerts_log, run_date)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Drift calibration report written to {output_path}")
    print(f"Total events analyzed: {analysis['total_events']}")


if __name__ == "__main__":
    main()
