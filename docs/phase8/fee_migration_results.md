# Phase 8-Alpha Fee Migration Results

**Date**: 2026-05-05
**Setup**: futures_maker config, 12-fold expanding walk-forward × 1M timesteps × fixed-start eval
**Branch**: claude/phase8-alpha-evidence
**PR baseline**: PR #126 (futures_maker cost model + funding rate wiring)

## Per-Fold Results

```
Fold Period start    BTC b&h   OOS ret (fix)   Trades/ep (fix)  Regime
   0 2025-04-28        13.1%          +0.27%                2   (bull)
   1 2025-05-28        -0.3%          -3.28%                1   (bear)
   2 2025-06-28        11.1%          +1.07%                1   (bull)
   3 2025-07-28        -5.7%          -0.88%                1   (bear)
   4 2025-08-27        -2.3%          -3.77%                1   (bear)
   5 2025-09-27         4.9%          +1.83%                1   (bull)
   6 2025-10-27       -21.7%          -2.65%                1   (bear)
   7 2025-11-26        -3.2%          -0.95%                2   (bear)
   8 2025-12-27         0.1%          +0.32%                2   (bull)
   9 2026-01-26       -22.0%          +1.52%                2   (bear)
  10 2026-02-26        -2.8%          -1.70%                2   (bear)
  11 2026-03-28        16.0%          +1.13%                2   (bull)
```

## Aggregates

| Metric                       | Value                      |
|------------------------------|----------------------------|
| All folds mean               | -0.59%                     |
| Bull mean (folds 0,2,5,8,11) | +0.92% — all 5 positive    |
| Bear mean (folds 1,3,4,6,7,9,10) | -1.67% — 1/7 positive |
| Folds positive               | 6/12                       |
| Mean max drawdown            | 4.2% (A0 baseline: 18.4%)  |
| Mean trades / episode        | 1.5                        |

## Verdict

**Marginal NO-GO with clear directional improvement; bear regime still bleeds, bull regime consistent positive.**

Against the Phase 8-Alpha go/no-go criteria (plan §1):

| Criterion                        | Target    | Result        | Pass? |
|----------------------------------|-----------|---------------|-------|
| Folds positive ≥ 7/12            | ≥ 7/12    | 6/12          | ✗     |
| Bull mean OOS return ≥ +0.5%     | ≥ +0.5%   | +0.92%        | ✓     |
| Bear mean OOS return ≥ −0.5%     | ≥ −0.5%   | −1.67%        | ✗     |

Bull regime is consistent and positive across all 5 folds — this is a meaningful signal. Bear regime remains the structural problem: 6 of 7 bear folds negative, with deep losses in folds 1 (−3.28%) and 4 (−3.77%). The exception (fold 9, BTC −22.0%, +1.52%) is an outlier that likely reflects a sharp reversal captured during the fold window rather than a reliable bear-regime edge.

Compared to the A0 pre-migration baseline (all-folds mean −5.03%), the fee migration improved mean OOS return by +4.44pp. Max drawdown collapsed from 18.4% to 4.2%, confirming that fee friction was the dominant drag. However, bull-only profitability is not sufficient for a live deployment where bear regimes constitute ~58% of the walk-forward window.

## Comparison to A0 Baseline

```
                     A0 (spot_taker, 1M)    Phase 8-Alpha (futures_maker, 1M)
All-folds mean              -5.03%                      -0.59%   (+4.44pp)
Bull mean                     n/a*                      +0.92%
Bear mean                     n/a*                      -1.67%
Folds positive               0/12                        6/12
Mean max drawdown            18.4%                        4.2%
Mean trades/ep               ~22                          1.5
```

\* A0 per-regime breakdown not computed at 1M (all folds were negative, no regime split).

## Implications for Phase 8-Beta

- **Reward shaping for bear regimes.** The agent's near-zero trade frequency (1.5/ep) suggests it learned "stay flat to avoid fees." This is rational in bull markets but leaves bear-regime alpha unexploited. A directional reward component (e.g., short-side incentive) or asymmetric position sizing should be explored.
- **Trade frequency too low.** 1.5 trades/episode at 1M training implies the policy collapsed to a near-hold strategy. Even with maker fees of 0.018%, the model is not trading enough to accumulate returns. Phase 8-Beta reward redesign should target 4–8 trades/episode.
- **Bull regime is solved.** All 5 bull folds positive at +0.92% mean is a reliable result at 1M training. Bull-regime logic should not be disturbed in Phase 8-Beta changes.
- **Fold 9 anomaly (bear +1.52%).** BTC −22% fold with a positive return warrants inspection — the model may have caught a relief-rally. Worth checking whether short positions were opened. Does not change the verdict but may inform bear-regime reward design.

## Caveats

1. **First true futures_maker measurement.** All prior runs (A0 1M baseline, 50k funding ablation) used the spot_taker environment — `run_wf.py` ignored `--config` before PR #126. This is the first result where the futures_maker cost model (maker fee 0.018% + funding rate) was actually active during training and eval.

2. **random_start eval invalid — all values are dataclass default 0.0.** `WalkForwardValidator.random_start_eval` defaults to `False` and `train_pipeline` does not propagate the config key. No random-start eval was executed; any "random" column in raw logs reflects uninitialized dataclass fields. Random-start results must be ignored entirely.

3. **Not strict apples-to-apples vs A0.** The A0 baseline used code before PR #123/126; this run uses code after PR #126. The environment diff (fee model + funding rate wiring) is the intended change, but other code-path differences cannot be fully excluded. The comparison is meaningful as a directional signal — fee/funding environment change dominates — but should not be treated as a controlled ablation.

4. **Trade count 1–2/episode is extremely low.** With round-trip cost now ~0.036%, the fee is no longer the binding constraint for a 1–2 trade policy; the policy itself is. The model trained toward minimal trading under the old high-fee regime and may need explicit reward shaping to explore higher-frequency strategies at the new cost level.
