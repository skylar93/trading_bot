# Phase 8-Beta §0 — Empirical reward stream dump

**Date**: 2026-05-05
**Script**: `scripts/diagnostic_reward_stream.py`
**Purpose**: empirically confirm the §0 diagnosis of the Phase 8-Beta plan before handing to Sonnet for code work.

## Setup

- Env: `SingleAssetRLTradingEnv` with futures_maker config (trading_fee=0.018%, no slippage, cost_model=futures_maker, funding_rate_per_8h=0.0001, hourly data, sharpe_lookback=60, sharpe_weight=0.1, drawdown_penalty_threshold=0.1, min_episode_steps=30).
- Data: `data/raw/BTCUSDT_1h.csv` (8760 rows, 1h bars). 200-bar slices.
- Three deterministic policies + a "worst drawdown slice" scan.

## Results

| Policy | Steps | Trades | Total reward | basic (sum) | sharpe_proxy raw (sum) | drawdown_penalty (sum) |
|---|---|---|---|---|---|---|
| P1 hold-only (bull slice) | 200 | 0 | **0.000** | 0.000 | 0.0 | 0.000 |
| P2 buy-and-hold (bull, terminated) | 31 | 1 | **−28.00** | −0.0005 | −280.0 (≈ −9.03/step) | 0.000 |
| P3 buy-and-hold (worst-drawdown slice) | 31 | 1 | **−28.00** | −0.0004 | −280.0 (≈ −9.03/step) | 0.000 |
| P4 active flip every 10 bars | 42 | 9 | **−39.00** | −0.0015 | −390.0 (≈ −9.29/step) | 0.000 |

(Sharpe column shows raw `sharpe_proxy` summed; final-reward contribution = `0.1 × sharpe_proxy`. The clip is ±10, weight 0.1, so per-step contribution is in ±1.0.)

## Key findings

1. **Hold-only confirms reward floor = 0**. 200 steps, all components 0. Returns buffer stays at 0, sharpe stays at 0, drawdown stays at 0.

2. **Sharpe component is the dominant negative force when in position**. Per-step sharpe_proxy averages −9.0 to −9.3 — i.e. saturated at the −10 clip almost every step. Weighted contribution ≈ −0.9 to −1.0 per step, accumulating to −28 over 31 steps and −39 over 42 steps.

3. **Drawdown penalty is effectively 0 in typical training episodes**. Across P2/P3/P4, drawdown never reached the 10% threshold within the 31-42 forced-termination window, so the penalty contributed exactly 0. v1 plan's "softer drawdown" variant would have had **zero observable effect** at this episode length.

4. **Sharpe penalty is invariant to trade frequency**. P2 (1 trade) and P4 (9 trades) accumulate the same per-step sharpe drag. Trade count itself does not modulate this lever.

5. **Episodes terminate early when in position** (31-42 steps vs P1's full 200). Likely stop-loss or capital-exhaustion check tripping under the saturated negative reward + small adverse moves. The agent observes: "stay flat = 200 steps × 0 reward; take position = 30-40 steps × −1 reward and forced exit." Optimal policy is to stay flat.

## Implications for plan §2 variants

v1 had three primary variants (inactivity, softer-drawdown, combined). v2 corrects this:

- **Drop softer-drawdown variant entirely** — empirical contribution is 0 in training-length episodes.
- **Add sharpe-weight reduction (0.1 → 0.02)** as the dominant uniform-reduction lever.
- **Add sharpe-clip narrowing (±10 → ±2)** as the dominant saturation-only lever.
- **Keep inactivity penalty** as a clean 0-floor-breaking lever (caveat: alone it is 100× smaller than sharpe drag, so unlikely to be sufficient in isolation).

## Reproducibility

```bash
PYTHONPATH=. python scripts/diagnostic_reward_stream.py
```

Deterministic given the same data file. Outputs vary by ≤ float tolerance.
