# Reward Function Audit (A0.7)

**Audited**: 2026-04-27  
**File audited**: `envs/single_asset_rl_env.py`  
**Auditor**: Claude Sonnet 4.6 (Phase 8 P0-a)

---

## Question

> Is the training reward **net-of-cost** (fees + slippage deducted) or **gross**?
> A gross reward causes train-vs-deploy mismatch: the agent is rewarded as if friction doesn't
> exist, but deployment deducts it.

---

## Finding: **PASS — Reward is net-of-cost**

### Execution path (envs/single_asset_rl_env.py)

```
step()
  ├─ slippage applied to executed_price          (lines ~464-482)
  │    executed_price = mid_price × (1 ± slippage)
  ├─ fee deducted from current_capital           (lines ~490-514)
  │    trade_cost = |actual_change × executed_price| × fee_rate
  │    capital_change = -trade_cost (always negative)
  │    if buy:  capital_change -= trade_value
  │    if sell: capital_change += trade_value
  │    self.current_capital += capital_change
  └─ portfolio_value = current_capital + position × close_price
       → already includes slippage + fee deduction

reward = log(portfolio_value_now / portfolio_value_prev)
       = log-return of net portfolio (after fees and slippage)
```

**Both slippage and fees are subtracted from `current_capital` before `portfolio_value` is
computed. The log-return reward therefore reflects what the agent actually earned after
transaction costs — there is no train-vs-deploy mismatch on the cost side.**

---

## Reward Components

| Component | Net-of-cost? | Notes |
|-----------|--------------|-------|
| Log portfolio return (basic) | ✅ Yes | `current_capital` has fees+slippage deducted |
| Sharpe proxy (risk-adjusted mode) | ✅ Yes | computed from the same net returns buffer |
| Drawdown penalty | ✅ Yes | `peak_portfolio_value` tracks net portfolio |
| Bankruptcy penalty (`-100`) | N/A | terminal signal, not a P&L term |

---

## Fee Model

- **Default fee**: `trading_fee` parameter (set per config, typically 0.001 = 10 bps)
- **Dynamic fee**: `_calculate_dynamic_fee(trade_value)` — applies tier-based rate for larger
  trades. Config: `config/trading.yaml` `trading_fee` field.
- **Fee tier sync**: active (Phase 7.5 SN6 — daily sync from exchange fee schedule).

---

## Slippage Model

- **Default**: `apply_slippage=True`. Market-impact formula:
  `base_slippage × (1 + volume_slippage_factor × |trade_size| / volume)`
- **Slippage model calibration**: R² > 0.3 confirmed (Phase 7.5 SN5).
- Train slippage uses the **same formula** as deployment
  (`deployment/execution/slippage_model.py`). Config values should be kept in sync via
  `config/trading.yaml`.

---

## Train-vs-Deploy Gap Risk (residual)

| Risk | Status | Mitigation |
|------|--------|-----------|
| Fee model divergence | Low | Fee tier sync (SN6) keeps rates aligned |
| Slippage model divergence | Moderate | R²~0.3 (realistic but noisy); A0.6 will track realized vs predicted |
| Partial-fill divergence | Low | `partial_fills=True` in env; deployment uses same fill-rate model |
| Funding rate (perp) | N/A (spot only) | `funding_pnl=0` until perp introduced (A6 config flag) |

**No action required** on reward structure. Residual gap is in slippage noise (expected for
retail-level market impact model) — addressed in A0.6 reality gap section.

---

## Verdict

> **Reward is net-of-cost. No train-vs-deploy mismatch on the reward definition.**
> The agent is trained on log-returns of a portfolio that already deducts slippage and fees.
> This is the correct design.

*Audit complete. No code changes needed.*
