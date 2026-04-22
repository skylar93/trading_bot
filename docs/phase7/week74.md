# Week 74: Execution Realism

**Track**: F — Real Connectivity (F12–F16)  
**Commit**: a518e75  
**Tests**: 58 new (test_execution_realism.py), all passing

---

## Files Added / Modified

| File | Change | Purpose |
|------|--------|---------|
| `deployment/execution/order_manager.py` | Modified | 4 order types, partial fills, TTL, cancel-replace |
| `deployment/analysis/slippage_model.py` | New (207 lines) | OLS linear regression slippage calibration |
| `deployment/exchange/fee_model.py` | Modified (261 lines) | VIP fee tiers + BNB discount |
| `tests/deployment/test_execution_realism.py` | New (664 lines) | F12–F16 test suite |
| `docs/phase7/slippage_calibration.md` | New (120 lines) | Slippage methodology documentation |

---

## Design Decisions

**Four order types (F12)**  
Paper and live mode both support: `market`, `limit`, `stop_loss_limit`, `take_profit`. Order dataclass extended with `stop_price` field. Paper mode: `_resolve_paper_fill_price()` dispatches per type. Live mode: dispatches to CCXT `create_order()` with `stopPrice` params. `update_paper_price()` processes pending limit/stop orders on each tick.

**Partial fill simulation (F13)**  
Config: `partial_fill_sim: bool`, `partial_fill_min_ratio: float` (default 0.3). `_draw_partial_fill_ratio()` returns random fill % in `[min_ratio, 1.0]`. Order status `"partial"` when partially filled; each fill appended to `order.fills` list. Live mode: status `"partial"` when CCXT returns `open` + `filled > 0`. Every fill event audit-logged with `{timestamp, filled_qty, fill_price, fee}`.

**Per-order TTL + cancel-replace (F14)**  
`expires_at` timestamp per order; background `_order_expiry_worker` thread auto-cancels expired orders. `cancel_replace_order(order_id, new_amount, new_price)` is atomic cancel + re-submit. Alerter notified on cancel failure.

**SlippageModel OLS (F15)**  
Features: `log_volume` (log(1 + bar_volume)), `realized_vol`, `side_enc` (0=buy, 1=sell), `size_frac` (order_size / bar_volume). Target: `slippage_frac = |fill_price - expected_price| / expected_price`. OLS with ridge regularisation 1e-6. Prediction clipped to `[0, max_slippage_frac]` (default 2%). Minimum 10 observations before model is used. Provides R² and coefficient summary for calibration.

**FeeModel VIP tiers (F16)**  
Binance VIP schedule (VIP 0–5):
- VIP 0: 0.10% maker / 0.10% taker
- VIP 1: 0.09% / 0.10%
- VIP 2: 0.08% / 0.10%
- VIP 3: 0.07% / 0.08%
- VIP 4–5: further reductions

BNB (native token) discount: 25% off fee when `use_bnb_discount=True`. Daily refresh via `fetch_trading_fees()` (default `refresh_interval_sec=86400`). FeeModel integrated into OrderManager `_compute_paper_fee()`.

---

## Test Coverage (310/310 pass)

- 4 order types paper-mode fill simulation
- Partial fill: random ratio distribution, fill list completeness
- TTL expiry: auto-cancel timing, alerter notification
- Cancel-replace: atomicity under concurrent workers
- SlippageModel: fit with synthetic data, R² > 0.3, prediction bounds
- FeeModel: each VIP tier, BNB discount, refresh cycle
