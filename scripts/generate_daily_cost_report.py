#!/usr/bin/env python3
"""
A6 Daily Cost Report generator (Week 90).

Reads fills from audit_log/audit.jsonl and writes a 4-axis P&L breakdown
to docs/reports/cost_breakdown_{date}.md.

Usage:
    python scripts/generate_daily_cost_report.py
    python scripts/generate_daily_cost_report.py --date 2026-04-27
    python scripts/generate_daily_cost_report.py --audit-log audit_log/audit.jsonl \\
        --output-dir docs/reports --date 2026-04-28

Scheduled via launchd (00:30 UTC daily).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Resolve repo root so this script works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deployment.analysis.cost_decomposition import CostDecomposer, DailyCostSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _bar(value: float, total: float, width: int = 20) -> str:
    """ASCII bar proportional to |value / total|."""
    if abs(total) < 1e-10:
        return " " * width
    frac = min(abs(value / total), 1.0)
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def render_daily_report(
    summary: DailyCostSummary,
    cumulative_summary,
    target_date: date,
    generated_at: datetime,
) -> str:
    """Render a markdown cost breakdown report."""
    lines: list[str] = []

    lines.append(f"# Daily Cost Report — {target_date.isoformat()}")
    lines.append("")
    lines.append(f"*Generated: {generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}*")
    lines.append("")

    # ── Today ──────────────────────────────────────────────────────────────
    lines.append("## Today's 4-Axis P&L Breakdown")
    lines.append("")
    lines.append(f"| Axis | Amount ($) | Share |")
    lines.append(f"|------|-----------|-------|")

    total_abs = abs(summary.total_pnl) if abs(summary.total_pnl) > 1e-10 else 1.0
    rows = [
        ("Signal P&L", summary.total_signal_pnl, "strategy alpha"),
        ("Slippage P&L", summary.total_slippage_pnl, "execution friction"),
        ("Fee P&L", summary.total_fee_pnl, "transaction costs"),
        ("Funding P&L", summary.total_funding_pnl, "funding / perps"),
    ]
    for label, val, note in rows:
        pct = val / total_abs * 100 if total_abs > 1e-10 else 0.0
        lines.append(f"| {label} | `{val:+.4f}` | {pct:+.1f}% — {note} |")

    lines.append(f"| **Total P&L** | **`{summary.total_pnl:+.4f}`** | 100% |")
    lines.append("")

    # ASCII stacked bar
    lines.append("### Visual breakdown")
    lines.append("")
    lines.append("```")
    for label, val, _ in rows:
        bar = _bar(val, total_abs)
        sign = "+" if val >= 0 else "-"
        lines.append(f"  {label:<14} [{bar}] {sign}${abs(val):.4f}")
    lines.append("```")
    lines.append("")

    # Stats
    lines.append("### Statistics")
    lines.append("")
    lines.append(f"- **Fills today**: {summary.num_fills}")
    lines.append(f"- **Closing trades**: {summary.num_sells}")
    lines.append(f"- **Avg slippage / fill**: ${summary.avg_slippage_per_fill:.4f}")
    if summary.num_fills > 0 and abs(summary.total_signal_pnl) > 1e-10:
        slip_pct = abs(summary.total_slippage_pnl) / abs(summary.total_signal_pnl) * 100
        lines.append(f"- **Slippage as % of signal**: {slip_pct:.2f}%")
    lines.append("")

    # ── Cumulative ─────────────────────────────────────────────────────────
    if cumulative_summary is not None:
        lines.append("## Cumulative (since stage start)")
        lines.append("")
        lines.append(f"| Axis | Cumulative ($) |")
        lines.append(f"|------|---------------|")
        lines.append(f"| Signal P&L | `{cumulative_summary.total_signal_pnl:+.4f}` |")
        lines.append(f"| Slippage P&L | `{cumulative_summary.total_slippage_pnl:+.4f}` |")
        lines.append(f"| Fee P&L | `{cumulative_summary.total_fee_pnl:+.4f}` |")
        lines.append(f"| Funding P&L | `{cumulative_summary.total_funding_pnl:+.4f}` |")
        lines.append(f"| **Total P&L** | **`{cumulative_summary.total_pnl:+.4f}`** |")
        lines.append("")
        lines.append(f"*Total fills: {cumulative_summary.num_fills}*")
        lines.append("")

    # ── Regression check ───────────────────────────────────────────────────
    lines.append("## Regression Check (A6.5)")
    lines.append("")
    computed = (
        summary.total_signal_pnl
        + summary.total_slippage_pnl
        + summary.total_fee_pnl
        + summary.total_funding_pnl
    )
    delta = abs(computed - summary.total_pnl)
    status = "✅ PASS" if delta < 0.01 else "❌ FAIL"
    lines.append(f"- 4-axis sum: `{computed:+.6f}`")
    lines.append(f"- Stored total: `{summary.total_pnl:+.6f}`")
    lines.append(f"- Δ: `{delta:.8f}` — **{status}** (threshold: $0.01)")
    lines.append("")

    lines.append("---")
    lines.append("*A6 Cost Decomposition — Phase 8 Week 90*")

    return "\n".join(lines)


def render_empty_report(target_date: date, generated_at: datetime) -> str:
    return (
        f"# Daily Cost Report — {target_date.isoformat()}\n\n"
        f"*Generated: {generated_at.strftime('%Y-%m-%dT%H:%M:%SZ')}*\n\n"
        f"No fills found for {target_date.isoformat()}.\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily A6 cost breakdown report")
    p.add_argument(
        "--date",
        default=None,
        help="Target date YYYY-MM-DD (default: yesterday UTC)",
    )
    p.add_argument(
        "--audit-log",
        default=str(_REPO_ROOT / "audit_log" / "audit.jsonl"),
        help="Path to audit.jsonl",
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "docs" / "reports"),
        help="Directory to write cost_breakdown_{date}.md",
    )
    p.add_argument(
        "--enable-funding",
        action="store_true",
        default=False,
        help="Parse funding_pnl from audit log (perps only)",
    )
    p.add_argument(
        "--all-time",
        action="store_true",
        default=False,
        help="Write reports for every date that has fills (not just target date)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    now_utc = datetime.now(timezone.utc)

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        # Default: yesterday UTC (report runs at 00:30 UTC for the completed day)
        target_date = (now_utc - timedelta(days=1)).date()

    audit_log_path = Path(args.audit_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading fills from %s", audit_log_path)
    decomposer = CostDecomposer.from_audit_log(
        audit_log_path,
        enable_funding=args.enable_funding,
    )

    cumulative = decomposer.cumulative_summary() if decomposer.fills() else None

    dates_to_report: list[date] = []
    if args.all_time:
        dates_to_report = [s.date for s in decomposer.all_daily_summaries()]
        if not dates_to_report:
            dates_to_report = [target_date]
    else:
        dates_to_report = [target_date]

    reports_written = 0
    for d in dates_to_report:
        summary = decomposer.daily_summary(d)
        if summary is not None:
            content = render_daily_report(summary, cumulative, d, now_utc)
        else:
            content = render_empty_report(d, now_utc)

        out_path = output_dir / f"cost_breakdown_{d.isoformat()}.md"
        out_path.write_text(content, encoding="utf-8")
        logger.info("Written: %s", out_path)
        reports_written += 1

    logger.info("Done — %d report(s) written to %s", reports_written, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
