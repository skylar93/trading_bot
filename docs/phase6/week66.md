# Week 66 Retrospective: P&L Attribution & Latency SLO (S51-S55)

**Date**: 2026-04-11  
**Branch**: claude/wonderful-burnell  
**Track**: C — Production Safety (final week)

---

## What Was Done

### S51 — PnLAttributor (`deployment/analysis/pnl_attribution.py`)

New module that decomposes each realised trade's P&L into four additive components:

```
market_move = net_pnl + slippage_cost + fees
```

- **`market_move`**: `(exit_price - entry_price) × quantity` — pure price direction PnL
- **`slippage_cost`**: execution quality cost (slip_frac × qty × exit_price)
- **`fees`**: actual transaction costs paid
- **`net_pnl`**: bottom-line contribution = `market_move - slippage_cost - fees`

Key design choices:
- `entry_price` is inferred from `trade.pnl` when available (reverses `apply_sell` formula),
  falling back to the preceding buy trade price. This avoids needing a separate buy-sell
  pairing data structure.
- `PnLAttributor.to_exporter_fields()` returns a dict that can be splatted directly into
  `MetricsExporter.update(**fields)` — zero coupling.
- `AttributionSummary` exposes `slippage_pct_of_gross` and `fees_pct_of_gross` for quick
  quality assessment.

### S52 — Latency Tracking (`deployment/execution/order_manager.py`)

Added to `Order` dataclass:
- `submitted_at`: timestamp when `submit_order()` is called
- `acked_at`: exchange ack (paper = immediate)
- `filled_at`: fill completion

`OrderManager._latency_samples` accumulates submit-to-fill durations (ms).  
`compute_latency_percentiles()` returns `{p50, p95, p99, count}` via `numpy.percentile`.

Paper mode: all three timestamps collapse to the same instant (sub-millisecond); samples
are still collected and percentiles remain valid (near-zero).

### S52 — `MetricsExporter.update_latency(p50, p95, p99)`

Convenience wrapper that stores latency percentiles as `MetricSnapshot` fields:
`latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`.

### S53 — Rolling Sharpe/Sortino (`deployment/monitoring/metrics_exporter.py`)

- `MetricsExporter.rolling_sharpe(window=20)`: rolling Sharpe from last N portfolio-value
  snapshots (annualised ×√252).
- `MetricsExporter.rolling_sortino(window=20)`: uses downside std below `mar=0.0`.
- Both added as `rolling_sharpe`, `rolling_sortino` fields in `MetricSnapshot`.
- `PaperTrader._log_step_metrics()` now computes and pushes these every step.

### S54 — `ReconciliationReport.by_order` (`training/analysis/reconciliation.py`)

New `OrderDivergence` dataclass:

```python
@dataclass
class OrderDivergence:
    order_id: str
    expected_price: float
    fill_price: float
    quantity: float
    slippage: float          # |fill - expected| / expected
    slippage_cost: float     # directional cost (positive = adverse)
    side: str
```

`ReconciliationReport.from_reports()` now accepts optional `orders` and `expected_prices`
arguments. When present, `by_order` is populated and an `order_avg_slippage` delta is
added. Warning fires if avg order-level slippage > 0.2%.

Backward compat: callers passing no `orders` get `by_order=[]` as before.

### Wiring into PaperTrader

`_log_step_metrics()` now orchestrates:
1. Attribution via `PnLAttributor` on all current trades
2. Latency via `order_manager.compute_latency_percentiles()`
3. Rolling Sharpe/Sortino via `metrics_exporter.rolling_sharpe/sortino()`

All fields flow into `MetricsExporter` and are exposed via `to_json()` / dashboard.

---

## Test Coverage (S55)

`tests/deployment/test_pnl_attribution.py` — 33 tests, 0 failures:

| Class | Tests |
|---|---|
| `TestPnLAttributor` | decomposition sum check, slippage, edge cases, summary |
| `TestMetricsExporterWeek66` | defaults, latency, rolling metrics, backward compat, thread-safety |
| `TestOrderManagerLatency` | empty, fill recording, accumulation, `filled_at`, rejected orders |
| `TestReconciliationByOrder` | empty, populated, sign, JSON, warning, backward compat |
| `TestPnLAttributionIntegration` | sum matches trade PnL, net = market_move - costs |

---

## Gotchas

1. **`VolatilityCircuitBreaker` validation**: `vol_threshold > 0` and `window >= 2` are
   hard requirements. Test must feed `window+1` prices to produce ddof=1 std (need ≥2 returns).

2. **`entry_price` inference**: In paper mode `trade.pnl` is the gross price-change PnL
   (before fees, from `PositionTracker.apply_sell`). So `entry_price = exit_price - pnl/qty`.
   This breaks for partial fills where multiple buys exist — production data should store
   VWAP entry explicitly.

3. **Rolling metrics in `_log_step_metrics`**: the method is only called when
   `mlflow_manager` is set. If you run PaperTrader without MLflow, rolling metrics
   still work via `metrics_exporter.rolling_sharpe()` as a query method — they just
   won't be automatically pushed each step.

4. **Latency in paper mode**: all latencies are < 1 ms. percentiles are technically
   correct but not meaningful for SLO enforcement. Enforcement is a live-mode concern;
   this implementation is the measurement foundation.

---

## Phase 7 Candidates (out of scope here)

- SLO alerting: `p99 > N ms → audit event + halt`
- Live-mode `acked_at` tracking (currently set immediately for paper)
- Attribution against a passive benchmark (buy-and-hold baseline)
- `PnLAttributor` integration into `ReconciliationReport.from_reports()`
