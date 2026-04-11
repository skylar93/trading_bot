# Week 64 — Live Risk Enforcement (S41-S46)

**Track**: C — Production Safety  
**Date**: 2026-04-11  
**PR**: Week 64: Live Risk Enforcement (S41-S46)

---

## What was done

Six safeguards wired into the live order submission path:

| Section | Feature | File(s) |
|---------|---------|---------|
| S41 | Correlation limit enforcement | `order_manager.py` |
| S42 | Fat-finger guard | `fat_finger_guard.py` |
| S43 | Volatility circuit breaker | `circuit_breaker.py` |
| S44 | Order idempotency key | `order_manager.py` |
| S45 | RateLimiter (extracted), ClockSync | `rate_limiter.py`, `clock_sync.py` |
| S46 | Tests (50 cases) | `tests/deployment/test_live_risk_enforcement.py` |

---

## Design decisions

### S41 — Correlation limit
`OrderManager.set_correlation(value)` stores the latest computed correlation externally (e.g., from PaperTrader or a live feed). At submit time, if a value is stored, `check_correlation()` is called on the risk_manager if it has that method, otherwise falls back to an inline `abs(corr) > threshold` check. This keeps the order manager decoupled from the specific risk manager type.

Rejected orders are logged to the audit trail as `type: "correlation_limit"`.

### S42 — FatFingerGuard
Two independent rules:
1. **Hard cap**: absolute maximum order size in base currency. No history needed. Immediate reject.
2. **Size multiplier**: order > mean(recent N fills) × multiplier. Only activates once fills are recorded — first orders always pass multiplier check (safe bootstrap).

History is tracked as a `deque(maxlen=lookback)`, updated only on filled orders so cancelled/rejected orders don't skew the baseline.

### S43 — VolatilityCircuitBreaker
Uses std(returns, ddof=1) over a rolling price window. Prices are fed via `OrderManager.update_paper_price()` (already the existing mechanism for paper mode price updates). In production, callers should feed prices at every tick.

After `cooldown` seconds, `is_tripped()` re-evaluates current vol and auto-resets if vol has dropped — no external reset call needed. Existing positions are unaffected; only new order submission is blocked.

### S44 — Idempotency key
`submit_order(..., idempotency_key=str)` stores a `key → order_id` mapping under the existing `RLock`. If the same key arrives again (duplicate submission, retry), the existing `order_id` is returned immediately without creating a new order. In live mode, the key is forwarded to the exchange as `clientOrderId` (standard CCXT field).

Concurrent threads with the same key are safe: the lock ensures only the first thread creates the order; all others read the existing mapping.

### S45 — RateLimiter / ClockSync
`RateLimiter` was already implemented inside `order_manager.py`. Extracted to `rate_limiter.py` and re-imported — no behavioral change, backward compatible.

`ClockSync` accepts an optional `time_fn` for unit testing without a live exchange. In production, `set_exchange(ccxt_obj)` is called in `OrderManager.__init__` and `check()` can be called periodically (e.g., before each batch of orders). With `halt_on_skew=False` (default), it only warns; set `halt_on_skew=True` in production config for hard enforcement.

---

## Gotchas

- **Correlation check fires before drawdown check**: correlation is checked first because it reflects market-structure risk that persists regardless of current P&L. Drawdown is portfolio-level, correlation is position-level.
- **FatFingerGuard bootstrap**: multiplier check is silently bypassed until the first fill is recorded. This is intentional — rejecting the very first order is unhelpful.
- **Circuit breaker cooldown=0**: valid for tests but ill-advised in production (constant re-evaluation thrash). Config default is 300s.
- **ClockSync returns 0.0 when no server time available**: callers should not treat 0.0 as "synchronized" — it means "unknown". For hard enforcement, pair with `halt_on_skew=True` and ensure `time_fn` or exchange is always set.
- **`update_paper_price` feeds circuit breaker**: in live mode, callers must call a price-update path to keep vol tracking active. In production, wire this to the market data feed.

---

## Test results

```
50 passed in 2.20s        (Week 64 new tests)
1634 passed, 0 failed     (full regression)
```

Baseline before Week 64: 1584 passed. Week 64 adds 50 tests.

---

## Week 65 preview

Data Pipeline Safety (S47-S50): feed staleness halt, NaN/inf feature guard, survivorship bias warning.
