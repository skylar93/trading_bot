# Week 85 — $100 First Dollar Drill Report (R21)

**Date**: 2026-04-23
**Plan ref**: Phase 7.5 R21
**Script**: `scripts/first_dollar_drill.py`

---

## Phase A — Simulation Drill (Completed)

Simulation drill with synthetic GBM price stream. All checks and kill switch test passed.

**Command**:
```bash
python scripts/first_dollar_drill.py --capital 100 \
    --report /tmp/week85_full_drill_report.json
```

**Timestamp**: 2026-04-23T01:03:41Z

### Results: 17/17 PASS

| Check | Result | Detail |
|-------|--------|--------|
| pytest.ini ignore ≤ 5 | ✅ | 4 ignores |
| Deprecated risk API callers → 0 | ✅ | |
| postmortem_template.md exists | ✅ | |
| go_live_checklist.md exists | ✅ | |
| kill_switch.py exists | ✅ | |
| StateStore checkpoint fresh | ✅ | first run |
| audit chain integrity | ✅ | first run |
| max_drawdown_threshold set | ✅ | 0.20 |
| daily_loss_limit set | ✅ | -500.0 |
| per_symbol_notional_max set | ✅ | 10,000 |
| portfolio_notional_max set | ✅ | 50,000 |
| leverage_max set | ✅ | 1.0 |
| API key scope probe (dry-run) | ✅ | |
| pre-commit detect-secrets | ✅ | no secrets found |
| Runbook drills ≥ 2 | ✅ | feed_stale + kill_switch |
| Kill switch < 5s | ✅ | **0.01s** |
| $100 drill completed | ✅ | steps=200, pnl=−$4.16, elapsed=0.0s |

**Kill switch timing**: 0.01s (limit: 5s) ✅  
**Final PnL (simulation)**: −$4.16 (random agent, expected noise)  
**Shutdown triggered**: False (clean run, no guards fired)

---

## Phase B — Real $100 Live Drill (PENDING)

**Pre-condition**: R19 72h sandbox run must be completed first.

### Protocol

The real drill is intentionally small and self-cancelling:

1. **Submit** a BTC/USDT limit buy order, $50 notional, 2% below mid-price
   - Reason: far enough from market to avoid immediate fill, gives cancel window
2. **Wait** 3 minutes
   - If unfilled: cancel → record
   - If filled: step 3
3. **Immediately submit** opposing sell order to flatten
4. Entire cycle must complete in **< 10 minutes**
5. Verify **audit chain** complete after cycle

### Commands

```bash
# Ensure credentials are loaded
export EXCHANGE_BINANCE_KEY="..."
export EXCHANGE_BINANCE_SECRET="..."

# Connectivity check
python scripts/sandbox_smoke.py --exchange binance --symbol BTC/USDT --duration 30

# Live drill — real $100, BTC/USDT, full audit cycle
python scripts/first_dollar_drill.py --live --capital 100

# Audit chain verification (auto-run inside drill, but can re-verify manually)
python scripts/verify_audit_log.py audit_log/audit.jsonl
```

### Results (Fill After Live Drill)

| Item | Expected | Actual |
|------|----------|--------|
| Order submission timestamp | | |
| Symbol | BTC/USDT | |
| Side | buy | |
| Notional | $50 | |
| Limit price | mid × 0.98 | |
| Fill status (3 min) | unfilled/filled | |
| Cancel/flat timestamp | | |
| Fill price | N/A (cancelled) | |
| Slippage vs. expected | N/A | |
| Exchange fee paid | | |
| Final position | flat | |
| Audit log complete | yes | |
| Total cycle time | < 10 min | |

**Exchange account**: _______________  
**Drill conducted by**: _______________  
**Date/time**: _______________  
**Outcome**: [ ] PASS — position flat, audit intact  /  [ ] FAIL — see incident log

### Audit Chain Verification

```bash
python scripts/verify_audit_log.py audit_log/audit.jsonl
# Expected: all entries chained, 0 gaps
```

---

## Completion Condition (R21)

- [ ] $100 real drill executed (or simulation + manual testnet order)
- [ ] Position confirmed flat after drill
- [ ] Audit log chain verified intact
- [ ] Total cycle time < 10 minutes
- [ ] Results recorded in this document

**Sign-off**: _______________ (date) — $100 drill complete, position flat, audit intact.
