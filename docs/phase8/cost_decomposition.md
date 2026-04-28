# A6 Cost Decomposition Dashboard

**Phase 8 Week 90 — P&L 4-axis breakdown + daily report**

## Overview

Real P&L = signal P&L + slippage P&L + fee P&L + funding P&L.

When live P&L goes negative, this decomposition identifies *which axis* is the cause — essential for debugging before the issue compounds.

| Axis | Formula | Notes |
|------|---------|-------|
| **Signal P&L** | `(mid_at_submit − entry_price) × qty` | Pure strategy alpha; what you'd earn filling at mid |
| **Slippage P&L** | `(fill_price − mid_at_submit) × qty` | Execution quality vs mid (negative = worse) |
| **Fee P&L** | `−fee_paid` | Always ≤ 0 |
| **Funding P&L** | `−funding_accrued` | Perps only; 0 for spot (config flag) |
| **Total P&L** | Sum of above | Algebraic identity guaranteed |

**Regression guarantee (A6.5)**: `|4-axis sum − realized P&L| < $0.01` per fill.

## Files

| File | Purpose |
|------|---------|
| [`deployment/analysis/cost_decomposition.py`](../../deployment/analysis/cost_decomposition.py) | Core module: `FillRecord`, `FillDecomposition`, `decompose()`, `CostDecomposer` |
| [`scripts/generate_daily_cost_report.py`](../../scripts/generate_daily_cost_report.py) | CLI: reads `audit_log/audit.jsonl` → `docs/reports/cost_breakdown_{date}.md` |
| [`deployment/launchd/com.tradingbot.daily-cost-report.plist`](../../deployment/launchd/com.tradingbot.daily-cost-report.plist) | macOS launchd: 00:30 UTC daily |
| [`tests/deployment/analysis/test_cost_decomposition.py`](../../tests/deployment/analysis/test_cost_decomposition.py) | 25 unit + regression tests |

## Dashboard

`GET /cost-breakdown` on the live dashboard (port 8080) returns cumulative and per-day JSON:

```json
{
  "cumulative": {
    "num_fills": 42,
    "total_signal_pnl": 128.50,
    "total_slippage_pnl": -3.20,
    "total_fee_pnl": -12.60,
    "total_funding_pnl": 0.0,
    "total_pnl": 112.70
  },
  "daily": [...]
}
```

Wire it in by passing `cost_decomposer=<CostDecomposer instance>` to `start_dashboard()`.

## Audit Log Fill Format

`paper_trader.py` now writes fill records to `audit_log/audit.jsonl` on every buy/sell:

```json
{
  "ts": "2026-04-28T12:00:01.234Z",
  "type": "fill",
  "payload": {
    "fill_id": "sell_20260428T120001234000",
    "side": "sell",
    "price": 65000.0,
    "mid_price": 65000.0,
    "quantity": 0.001,
    "fee": 6.5,
    "pnl": 20.0,
    "entry_price": 45000.0
  },
  "hash": "..."
}
```

## Running the Report Manually

```bash
# Yesterday's report (default)
python scripts/generate_daily_cost_report.py

# Specific date
python scripts/generate_daily_cost_report.py --date 2026-04-27

# All dates with fills
python scripts/generate_daily_cost_report.py --all-time

# With funding (perps)
python scripts/generate_daily_cost_report.py --enable-funding
```

## Config

`config/deployment.yaml`:

```yaml
cost_decomposition:
  audit_log: "audit_log/audit.jsonl"
  output_dir: "docs/reports"
  enable_funding_tracking: false   # set true for perps
  report_schedule_utc: "00:30"
```

## launchd Setup

```bash
cp deployment/launchd/com.tradingbot.daily-cost-report.plist \
    ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradingbot.daily-cost-report.plist
```

**Note**: Hour/Minute in the plist is set for PDT (UTC-7). Adjust for your timezone.
