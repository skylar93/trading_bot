# Week 76: Pre-Trade Compliance

**Track**: G — Governance & Go-Live Gate (G6–G10)  
**Commit**: 119b894  
**Tests**: 39 new, 2336 total passed, 42 skipped

---

## Files Added / Modified

| File | Change | Purpose |
|------|--------|---------|
| `risk_management/limits.py` | New (278 lines) | PreTradeComplianceChecker module |
| `deployment/execution/order_manager.py` | Modified (+58 lines) | compliance_checker integration |
| `tests/deployment/test_pre_trade_compliance.py` | New (455 lines) | G6–G10 test suite |

---

## Design Decisions

**PreTradeComplianceChecker with `check_all()` (G6–G9)**  
Single class exposing `check_all(symbol, side, amount, price, portfolio_snapshot)` → returns `(bool, str)` (allowed, rejection_reason). Individual guards:

**Position Limits (G6)**
- Per-symbol notional max: `per_symbol_notional_max`
- Portfolio notional max: `portfolio_notional_max`
- Leverage cap: `leverage_max`
- Projected notional computed from (amount × price) before acceptance

**Self-Trade Prevention (G7)**
- `register_open_order(symbol, price, side)` tracks resting limit orders
- `deregister_open_order(symbol, price)` on fill/cancel
- Rejects new order if opposite-side resting order at same price exists
- Toggle: `self_trade_prevention: bool` (default True)

**Notional Cap Per Unit Time (G8)**
- Sliding-window deques: `_hourly_window`, `_daily_window` store `(timestamp, notional)` pairs
- `check_notional_cap()` sums rolling 60-min and 24-hour windows
- Rejects if sum + new_order_notional > `hourly_notional_cap` or `daily_notional_cap`
- Old entries auto-evicted from deque heads

**Wash Trade Guard (G9)**
- Per (symbol, side) cooldown dict `_wash_guard` stores last acceptance timestamp
- `check_wash_trade()` rejects same-direction order on same symbol within `wash_trade_cooldown_sec`
- Configurable: `wash_trade_cooldown_sec: 0` disables (default)

**OrderManager Integration**  
`compliance_checker` is an optional constructor param (backward compatible: default None). `check_all()` called between fat-finger guard and max-order-size clamp. `record_order()` called on acceptance to update notional windows. `register/deregister_open_order()` wired to limit-order lifecycle (submit + fill/cancel callbacks). Rejection reasons audit-logged as `risk_events`.

**Thread Safety**  
All deque operations and dict mutations in `PreTradeComplianceChecker` protected by `threading.Lock`.

---

## Test Coverage (39/39 pass)

- Each guard: unit scenario (pass + reject boundary cases)
- check_all() integration: multiple guards active simultaneously
- Notional cap: hourly window eviction, daily window eviction
- Wash trade: cooldown expiry allows re-entry
- Thread-safety: concurrent check_all() calls (no double-counting)
- OrderManager E2E: compliance_checker rejection audited and returned as error
