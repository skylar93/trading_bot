# Week 78: Observability Stack

**Track**: H — Production Ready (H1–H5)  
**Commit**: bc8c7b6  
**Tests**: 41 new, all passing

---

## Files Added / Modified

| File | Change | Purpose |
|------|--------|---------|
| `deployment/monitoring/metrics_exporter.py` | Modified (+243 lines) | Prometheus backend for 30+ MetricSnapshot fields |
| `deployment/monitoring/grafana_dashboard.json` | New (684 lines) | 12-panel import-ready Grafana dashboard |
| `deployment/monitoring/alerter.py` | Modified (+134 lines) | Discord channel + 3 new notification methods |
| `deployment/monitoring/tracing.py` | New (188 lines) | OpenTelemetry span context manager |
| `deployment/monitoring/sentry_init.py` | New (163 lines) | Sentry init + before_send scrubber |
| `config/monitoring.yaml` | Modified | alert_channels, discord_webhook_url, sentry config |
| `docker-compose.yml` | Modified | Prometheus :9090 + Grafana :3000 services |
| `deployment/monitoring/prometheus.yml` | New (32 lines) | Prometheus scrape config (15s interval, port 9100) |
| `tests/test_week78_observability.py` | New (487 lines) | H1–H5 test suite |

---

## Design Decisions

**MetricsExporter: 30+ Prometheus metrics (H1)**  
Port changed from 9090 → 9100 (Prometheus itself runs on 9090). All 30 `MetricSnapshot` fields mapped:
- **Gauges**: portfolio_value, cash, position, unrealised_pnl, realised_pnl, drawdown_pct, daily_pnl, is_halted, kill_switch_active, win_rate, sharpe_ratio, rolling_sharpe, rolling_sortino, current_var, P&L attribution (market_move, slippage, fees, net), current_regime, latencies (p50/p95/p99)
- **Counters**: num_trades, alerts_fired, feature_drift_alarms (monotonic)
- **Histogram**: order latency with buckets [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms

New `MetricSnapshot` field: `kill_switch_active`. `observe_order_latency(span, latency_ms)` for direct histogram recording.

**Grafana Dashboard: 12 panels (H2)**  
Auto-provisioned via `datasources/prometheus.yaml` (no manual setup). Panels:
1. Trading Status (halt/running gauge)
2. Kill Switch (active flag)
3. Drift Detected (boolean)
4. Portfolio Value (time series)
5. Drawdown % (gauge with thresholds)
6. Trades / Alerts Total (counter stats)
7. P&L Attribution (stacked area: market_move, slippage, fees)
8. P&L Attribution Breakdown (table)
9. Risk-Adjusted Performance (Sharpe/Sortino/VaR)
10. Order Latency (distribution)
11. Order Latency Percentiles (p50/p95/p99 gauges)
12. Drift Alarms & Alerts (cumulative)

**Discord as 4th alert channel (H3)**  
`DISCORD_WEBHOOK_URL` env var. Colour-coded embeds: WARNING=orange (0xFFA500), CRITICAL=red (0xFF0000), INFO=blue (0x00B0F4). New methods: `notify_error(exception, context)`, `notify_kill_switch()`, `notify_audit_chain_break()`. Total channels: console, telegram, webhook, Discord.

**OTel tracing: init only, span instrumentation deferred (H4)**  
`tracing.py` provides `start_span(name, attributes)` context manager and named span conventions:
- `trading.order.submit` (full round-trip)
- `trading.order.risk_check`
- `trading.order.compliance`
- `trading.agent.decide`
- `trading.data.feed_tick`

Auto-selects exporter via `OTEL_EXPORTER_OTLP_ENDPOINT` (OTLP if set, ConsoleSpanExporter otherwise). `_NoopTracer` fallback if opentelemetry-sdk not installed. **Note**: Span instrumentation callsites in OrderManager are deferred to Phase 7.5 W83.

**Sentry scrubbing (H5)**  
`before_send` hook in `sentry_init.py`:
- Credential fields masked: api_key, api_secret, secret, token, password, passphrase, etc.
- Exchange env vars: BINANCE_API_KEY, COINBASE_API_KEY, etc.
- Price data arrays: stripped if > 200 chars (quota protection)
- Regex: 32+ char base64-ish strings (API key signature patterns)
- Recursive through exception stack frames (vars section)
- `capture_exception()` never raises (safe wrapper)

---

## Deviation from Plan

OTel span instrumentation (H4) — `tracing.py` provides the infrastructure but callsites in `OrderManager.submit_order` are not wired. This was intentional: the tracing init was the W78 deliverable; instrumentation is deferred to Phase 7.5 W83 (R12) when Grafana Tempo integration is also available.

---

## Test Coverage (41/41 pass)

- MetricsExporter: all 30+ fields exported, histogram buckets, counter monotonicity
- Grafana JSON: valid JSON, 12 panels, datasource name
- Alerter: Discord webhook format, colour codes, new notification methods
- Tracing: start_span context manager, noop fallback, attribute recording
- Sentry: before_send scrubbing (each credential pattern), price data stripping, safe capture
