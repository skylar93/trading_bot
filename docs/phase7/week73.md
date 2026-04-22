# Week 73: Position Reconciliation on Startup

**Track**: F — Real Connectivity (F7–F11)  
**Commit**: 41c66d4  
**Tests**: 2202 total passed, 42 skipped, 0 failed

---

## Files Added / Modified

| File | Change | Purpose |
|------|--------|---------|
| `deployment/exchange/snapshot.py` | New (139 lines) | CCXT REST wrapper for exchange state |
| `deployment/paper_trader.py` | Modified (+151 lines) | reconcile_on_boot, periodic reconcile |
| `deployment/execution/order_manager.py` | Modified | ClockSync wired into submit_order (F11) |
| `data/sources/ccxt_live.py` | New (137 lines) | F2 deliverable missed in W72 |
| `tests/exchange/test_snapshot.py` | New (599 lines) | Full reconcile scenario coverage |
| `tests/exchange/test_ccxt_live_source.py` | New (210 lines) | ccxt_live_source unit tests |

---

## Design Decisions

**ExchangeSnapshot best-effort (F7)**  
`get_positions()`, `get_open_orders()`, `get_balance()` each catch all exceptions and return empty/zero on failure. Normalised output format: position dict `{symbol, qty, entry_price, side, unrealised_pnl}`. `snapshot()` combines all three.

**Mismatch detection at boot (F8)**  
`PaperTrader.reconcile_on_boot()` called during `restore()` after StateStore load. Three mismatch types detected:
- `qty_mismatch`: `abs(local_qty - exchange_qty) > qty_threshold`
- `price_drift`: `abs(local_entry - exchange_entry) / exchange_entry > price_threshold`
- `open_orders_mismatch`: local pending count vs exchange open order count

**Three mismatch policies**  
Config `reconciliation.on_mismatch`:
- `halt`: PaperTrader sets `is_halted=True`, alerter notified
- `warn`: alerter notified, bot continues
- `ignore`: diff logged only

All diffs written to audit log as structured dicts.

**Periodic reconcile throttled (F9)**  
`_periodic_reconcile()` fires every `interval_sec` (default 60s) inside the `run()` loop. Throttled to avoid per-step exchange RTT overhead.

**ClockSync wired before submit (F11)**  
`clock_sync.check()` called with 30s throttle in `submit_order`. Reduces exchange round-trip overhead vs per-order sync.

---

## Deviation from Plan

`data/sources/ccxt_live.py` (F2) was delivered in this week, not W72. The commit note documents this as a missed W72 deliverable caught in W73 review.

---

## Test Coverage

- ExchangeSnapshot: happy path (positions, orders, balance), API error graceful return
- Boot reconcile: no-mismatch, qty_mismatch, price_drift, open_orders_mismatch
- Each mismatch with halt|warn|ignore policy
- Periodic throttle: fires after interval, does not fire before
- Bootstrap integration: restore() → reconcile-on-boot with forced mismatch injection
