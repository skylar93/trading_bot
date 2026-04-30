# Diagnostic: design lever isolation

**Date**: 2026-04-30
**Setting**: 50k × 1 fold (n_splits=2) × 20 eval episodes, random_start_eval=True
**Data**: data/BTCUSDT_1h.csv, OOS slice ~8760 bars (last 50% of 2024-04 → 2026-04 range)
**Driver**: run_diagnostic.py

## Variants

| Variant | trading_fee | slippage | sharpe_weight | risk_adj_reward | action_space |
|---------|-------------|----------|---------------|-----------------|--------------|
| baseline   | 0.001 | on  | 0.1 | True  | [-1, 1] |
| no_fee     | 0.0   | off | 0.1 | True  | [-1, 1] |
| no_sharpe  | 0.001 | on  | 0.0 | False | [-1, 1] |
| long_only  | 0.001 | on  | 0.1 | True  | [0, 1]  |
| all_off    | 0.0   | off | 0.0 | False | [0, 1]  |

## Results

| Variant | OOS ret (fixed) | OOS ret (random) | OOS DD (fixed) |
|---------|-----------------|------------------|----------------|
| baseline   | -10.89% | -3.94% | 89.3% |
| no_fee     | **+0.61%** | **-0.19%** | **6.3%** |
| no_sharpe  | -8.00% | -3.91% | 79.5% |
| long_only  | -7.39% | -3.57% | 76.6% |
| all_off    | -3.46% | -0.24% | 48.8% |
| BTC buy-hold (same OOS slice) | -33.0% | — | — |

> OOS slice is a bear-market period (BTC -33% over same window). Fixed-start result
> picks a deterministic start at the OOS window open; random-start averages over all
> start positions (tends to shorten effective episode length → smaller loss magnitude).

## Interpretation

### Case C: `no_fee` single lever dominates

**Verdict**: Fee + slippage (0.1% + 0.05%) is the dominant failure lever. `no_fee` alone
achieves near-breakeven (+0.61% fixed) while all other single-lever variants remain in the
-7% to -8% range. The `all_off` combination is _worse_ than `no_fee` alone (−3.46% vs
+0.61%), indicating that long_only + no_sharpe partially cancel each other out on this
bear-market OOS slice.

**Why fee dominates**: The policy trades frequently (ep_len ≈ 33 steps at 1h bars).
Each round-trip costs 2 × 0.1% fee + slippage ≈ 0.25%. Over ~33 steps this compounds
to significant drag. With fee=0 the policy can break even; with fee=0.1% it cannot.

**Caveats**:
1. OOS slice is a 2025–2026 bear period for BTC (-33% buy-hold). The no_fee advantage
   partly reflects the policy learning to short effectively without fee drag.
2. The A0 overnight run covered a 2024-04 → 2026-04 bull period (+200% BTC) — different
   regime. The diagnostic may not fully generalize; both bull and bear diagnostics needed
   before concluding fee alone is sufficient.
3. 50k steps is too short for stable policy convergence; results have high variance.
   All conclusions are directional, not definitive.

## Next steps

- Hand to operator + Opus for Phase 8-Alpha plan design
- Do NOT auto-proceed (per phase8-restructured.md NO-GO branch policy)
- Recommended follow-up experiment (operator decision): re-run `no_fee` variant at
  1M steps × 12 folds on the full dataset to confirm whether fee elimination alone
  recovers positive OOS returns across bull and bear periods
- Also consider: reward shaping for lower trade frequency (penalty on position changes)
  as an alternative to zero-fee assumption

## Raw log

`logs/diagnostic_long_only_no_fee.log` — 257 KB, 2026-04-30
