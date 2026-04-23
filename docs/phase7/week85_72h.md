# Week 85 — Sandbox 72h Continuous Run (R19)

**Date**: 2026-04-23
**Plan ref**: Phase 7.5 R19
**Status**: PENDING — requires Binance testnet credentials + 72h real-time window

---

## Pre-Run Validation (Completed)

All automated pre-flight checks PASS before starting the 72h run.

```
python scripts/first_dollar_drill.py --check-only
  → 15/15 PASS (2026-04-23T01:03:13Z)
```

| Check | Result |
|-------|--------|
| pytest.ini ignore ≤ 5 | ✅ 4 ignores |
| Deprecated risk API callers → 0 | ✅ |
| Postmortem template exists | ✅ |
| Go-live checklist exists | ✅ |
| kill_switch.py exists | ✅ |
| max_drawdown_threshold set | ✅ 0.20 |
| daily_loss_limit set | ✅ -500.0 |
| per_symbol_notional_max set | ✅ 10,000 |
| portfolio_notional_max set | ✅ 50,000 |
| leverage_max set | ✅ 1.0 |
| API key scope probe (dry-run) | ✅ |
| pre-commit detect-secrets | ✅ |
| Runbook drills ≥ 2 | ✅ feed_stale + kill_switch |

---

## Run Configuration

```bash
# Environment
export EXCHANGE_BINANCE_TESTNET_KEY="..."
export EXCHANGE_BINANCE_TESTNET_SECRET="..."

# Start sandbox smoke (manual step, verify ticks arrive)
python scripts/sandbox_smoke.py --exchange binance --symbol BTC/USDT --duration 300

# Start PaperTrader in sandbox mode (72h)
python scripts/run_paper_trader.py \
    --config config/local_3060ti.yaml \
    --exchange-mode sandbox \
    --duration-hours 72

# Monitor via Grafana (http://localhost:3000)
docker-compose -f docker-compose.monitoring.yml up -d
```

**Exchange**: Binance Spot Testnet  
**Symbol**: BTC/USDT  
**exchange_mode**: sandbox  
**Initial balance**: 10,000 USDT (testnet)

---

## Observation Items (Fill After Run)

| Metric | Expected | Observed | Status |
|--------|----------|----------|--------|
| Runtime duration | 72h continuous | | |
| Kill switch self-trigger | 0 | | |
| Max drawdown during run | < 20% | | |
| Reconciliation drift events | 0–5 (minor) | | |
| Feed stale events | ≥ 0 (auto-recover) | | |
| Slippage model refresh count | ≥ 3 | | |
| Fee API sync success count | ≥ 3 (24h × 3) | | |
| Schema drift events | 0 | | |
| Canary auto-demotion triggers | 0 (if no canary running) | | |
| OTel traces in Tempo | ≥ 100 | | |

---

## 72h Run Results (Fill After Run)

**Run start**: _______________  
**Run end**:   _______________  
**Total uptime**: ___ h ___ m

### Prometheus Snapshot (final hour)

| Metric | p50 | p95 | p99 |
|--------|-----|-----|-----|
| submit_order latency (ms) | | | |
| risk_check latency (ms) | | | |
| compliance_check latency (ms) | | | |
| feed tick → decision (ms) | | | |

### Incident Log

| Timestamp | Type | Detail | Resolution |
|-----------|------|--------|------------|
| (none) | | | |

### Grafana Screenshots

Attach screenshots for:
- [ ] Drawdown chart (72h)
- [ ] Order flow (orders/hour)
- [ ] Slippage vs. expected chart
- [ ] Alert history

---

## Completion Condition (R19)

- [ ] 72h elapsed without kill switch self-trigger
- [ ] Slippage model refresh ≥ 3 times
- [ ] Fee sync ≥ 3 times
- [ ] Max drawdown within limits
- [ ] All metrics in expected range

**Sign-off**: _______________ (date) — 72h run complete, no critical incidents.
