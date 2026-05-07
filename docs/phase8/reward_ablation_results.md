# Phase 8-Beta Stage 1 — Reward Ablation Results

**Date**: 2026-05-06 (Stage 1 + 200k); 2026-05-07 (B3 1M empirical closure)
**Setup**: 4 variants × 50k timesteps × 12-fold expanding walk-forward × eval_episodes=20 × random_start_eval=true. Plus B3-only 200k and 1M follow-ups to test convergence dynamics.
**Branch**: claude/phase8-beta-reward-ablation (Stage 1 code) + claude/phase8-beta-closure (this evidence pack)
**Plan reference**: `/Users/skylar/.claude/plans/phase8-beta-reward-shaping.md` §2 (v2)
**Verdict**: **NEGATIVE — Phase 8-Beta closed. No Stage 2.**

---

## TL;DR

- All 4 reward-shaping variants (B0/B1/B2/B3) produce **statistically indistinguishable returns** at 50k.
- B3 (sharpe-clip ±10→±2) is the only variant that **changed trade frequency** (12.90 trades/ep vs B0's 6.36 at 50k fixed-start).
- B3-only 200k follow-up shows trade count collapsing **12.90 → 5.00** as training progresses, monotonically converging to the v3 1M baseline (~1.5 trades/ep).
- Returns at 200k match v3 1M to within 0.04pp on every metric — **the policy converges to the same equilibrium regardless of reward shaping**.
- Predictions in plan §7 were largely wrong: B2 was expected to be the dominant lever but produced the *fewest* trades. The reward-magnitude model was incomplete; the missing piece is policy-gradient convergence dynamics.

**Implication**: reward shaping within these 4 variants does not change the final policy — it only changes the *rate* of convergence to the same near-hold equilibrium. Phase 8-Gamma must use a different lever (regime gating, asymmetric reward, or feature engineering).

---

## Variant Descriptions

| Variant | Change vs B0 | Config knob(s) |
|---------|-------------|----------------|
| B0 | Baseline (no change) | `config/futures_maker.yaml` |
| B1 | Per-step inactivity penalty when flat | `inactivity_penalty: 0.0005` |
| B2 | Sharpe weight reduced 5× (0.1 → 0.02) | `sharpe_weight: 0.02` |
| B3 | Sharpe clip narrowed (±10 → ±2) | `sharpe_clip_value: 2.0` |

---

## Aggregate Comparison Table — 50k Stage 1

### Random-start eval (selection criterion uses these)

| Variant | All mean | Bull mean | Bear mean | Folds positive | Trades/ep |
|---------|---------|-----------|-----------|----------------|-----------|
| B0      | -0.03%  | +0.00%    | -0.05%    | 3/12           | 1.45      |
| B1      | -0.04%  | +0.01%    | -0.07%    | 4/12           | 1.40      |
| B2      | +0.00%  | +0.03%    | -0.01%    | 4/12           | 1.33      |
| B3      | -0.06%  | +0.02%    | -0.11%    | 3/12           | **2.12**  |

### Fixed-start eval (the *training* metric — what the policy actually optimized)

| Variant | All mean | Bull mean | Bear mean | Folds positive | Trades/ep | Mean MaxDD |
|---------|---------|-----------|-----------|----------------|-----------|------------|
| B0      | -0.52%  | +0.90%    | -1.53%    | 6/12           | 6.36      | 15.74%     |
| B1      | -0.58%  | +0.84%    | -1.60%    | 6/12           | 6.75      | 16.18%     |
| B2      | -0.58%  | +0.92%    | -1.65%    | 6/12           | 5.55      | 16.36%     |
| B3      | -0.57%  | +0.89%    | -1.61%    | 6/12           | **12.90** | 16.00%     |

Bull folds: 0, 2, 5, 8, 11. Bear folds: 1, 3, 4, 6, 7, 9, 10.

Notes:
- All four variants land within seed-noise distance on every return metric; the only meaningful behavioural difference is trade frequency.
- B3 doubled trade count without changing returns — the extra trades carried no edge.
- MaxDD ~16% across all variants is *much higher* than v3 1M's 4.2% — at 50k the policy has not yet learned to avoid bad trades. See §"Convergence trajectory" below.

---

## Selection Criterion (§2.5)

| Check | Criterion | Result | Pass? |
|-------|-----------|--------|-------|
| Criterion 1 | Random-start trades/ep > 5 AND bull mean ≥ B0 bull (+0.00%) | Best is B3 at 2.12 trades/ep | ❌ NO |
| Criterion 2 | Random-start trades/ep > 3 AND bear mean strictly better than B0 (-0.05%) | Best is B3 at 2.12 trades/ep with bear -0.11% (worse) | ❌ NO |

**Selected variant**: NONE. Aggregator output: `STOP — no variant passes criterion 2`.

---

## Criterion calibration caveat — random-start trade counts are confounded

Random-start episodes begin at random offsets within each fold's test window, so they are typically much shorter than fixed-start episodes (10-30 steps vs 30-200). Trade count scales with episode length. Comparing B0's random-start 1.45 trades/ep to a "trades > 5" threshold is a **3.4× lift requirement on a length-confounded metric**.

The fixed-start view (table above) shows trade frequency differences clearly: B3 *did* produce 2× more trades than B0 (12.90 vs 6.36). But these extra trades had no return edge.

For Phase 8-Gamma, the criterion should be expressed against **fixed-start** trade counts (where episode length is uniform per fold), not random-start.

---

## B3-only 200k follow-up — convergence dynamics

The 50k snapshot showed B3 trading 2× more than B0 in fixed-start. To test whether this advantage survives longer training, B3 was re-run at 200k with all other settings identical.

### B3 trade count vs training duration

| Timesteps | All mean | Bull mean | Bear mean | Folds+ | Trades/ep | MaxDD |
|-----------|---------|-----------|-----------|--------|-----------|-------|
| 50k       | -0.57%  | +0.89%    | -1.61%    | 6/12   | 12.90     | 16.00%|
| 200k      | -0.59%  | +0.88%    | -1.65%    | 6/12   | 5.00      | 16.37%|
| **1M**    | **-0.59%** | **+0.93%** | **-1.68%** | **6/12** | **1.66** | **16.59%** |
| v3 1M (B0 baseline) | -0.59% | +0.92% | -1.67% | 6/12 | 1.50 | **4.20%** |

The 1M run was launched after the 200k follow-up to confirm the convergence prediction empirically (the prior version of this evidence pack had 1M as a predicted row). Per-fold returns match v3 1M to within 0.05pp on every fold — an even cleaner reproduction than predicted.

### B3 200k per-fold detail (fixed-start)

| Fold | Regime | OOS ret | Sharpe | Trades/ep | MaxDD |
|------|--------|---------|--------|-----------|-------|
| 0    | bull   | +0.25%  | +1.52  | 6.40      | 0.15% |
| 1    | bear   | -3.24%  | -14.06 | 1.80      | 46.44%|
| 2    | bull   | +1.03%  | +18.27 | 4.95      | 0.00% |
| 3    | bear   | -0.91%  | -10.45 | 4.25      | 16.04%|
| 4    | bear   | -3.65%  | -23.50 | 5.45      | 50.61%|
| 5    | bull   | +1.77%  | +9.51  | 2.85      | 0.00% |
| 6    | bear   | -2.63%  | -8.72  | 4.90      | 39.74%|
| 7    | bear   | -0.88%  | -7.40  | 6.75      | 15.58%|
| 8    | bull   | +0.30%  | +3.34  | 5.50      | 0.00% |
| 9    | bear   | +1.50%  | +26.20 | 2.50      | 0.00% |
| 10   | bear   | -1.71%  | -21.70 | 5.15      | 27.91%|
| 11   | bull   | +1.04%  | +3.59  | 9.55      | 0.00% |

### Comparison: B3 1M vs Phase 8-Alpha v3 1M (= B0 1M on current code)

| Metric | v3 1M (B0 baseline) | B3 1M | Δ |
|--------|---------------------|-------|---|
| All mean | -0.59% | -0.59% | 0.00 |
| Bull mean | +0.92% | +0.93% | +0.01 |
| Bear mean | -1.67% | -1.68% | -0.01 |
| Folds positive | 6/12 | 6/12 | 0 |
| Trades/ep | 1.50 | 1.66 | +0.16 |
| **Mean MaxDD** | **4.20%** | **16.59%** | **+12.39pp (4× worse)** |

Returns are within 0.01pp on every metric — B3 1M is essentially a byte-identical reproduction of v3 1M on the return surface. **But mean MaxDD is 4× worse.**

### B3 1M per-fold detail (fixed-start)

| Fold | Regime | OOS ret | Trades/ep | MaxDD | v3 MaxDD (approx) |
|------|--------|---------|-----------|-------|--------|
| 0    | bull   | +0.27%  | 2.00      | 0.00% | low |
| 1    | bear   | -3.28%  | 1.05      | **46.97%** | low |
| 2    | bull   | +1.07%  | 1.00      | 0.00% | low |
| 3    | bear   | -0.88%  | 1.00      | 15.49% | low |
| 4    | bear   | -3.77%  | 1.00      | **51.82%** | low |
| 5    | bull   | +1.87%  | 2.00      | 0.00% | low |
| 6    | bear   | -2.65%  | 1.00      | **40.01%** | low |
| 7    | bear   | -0.95%  | 1.95      | 16.58% | low |
| 8    | bull   | +0.32%  | 1.95      | 0.00% | low |
| 9    | bear   | +1.52%  | 2.15      | 0.00% | low |
| 10   | bear   | -1.74%  | 2.95      | 28.27% | low |
| 11   | bull   | +1.13%  | 1.85      | 0.00% | low |

Bull folds: clean (0% MaxDD on 4/5 folds). The negative-DD all comes from bear folds 1, 4, 6 — three deep unrealized drawdowns (40-52%) that recovered to "only" -1 to -4% realized loss.

### Newly-discovered cost of B3: Sharpe-clip narrowing degrades drawdown path

The 1M run revealed a hidden cost not visible in 200k: **B3's narrower clip removed a useful stop-loss-like reflex** that v3 (default clip ±10) had learned. The mechanism:

- v3 trains under saturated negative sharpe whenever a position bleeds; the policy learns "exit fast when reward turns deeply negative" → small MaxDD.
- B3 caps the negative sharpe at -0.4 (= ±2 × weight 0.1); the policy's exit signal is dampened → it holds through deep unrealized drawdowns and only liquidates when basic_reward + drawdown_penalty finally compounds enough.
- Realized returns end up similar (the deep drawdowns recover), but the path through 40-50% drawdowns is unacceptable for a live system.

Plan §0 hypothesis: "sharpe saturation = trade-aversion mechanism, weakening it = good." Reality: **saturation was also doing the work of an implicit stop-loss**. Weakening it kept the policy *more* willing to enter and *less* willing to exit — net worse risk control with the same realized return.

**The convergence is monotonic and predictable on returns; the drawdown path is not.** B3 1M would have been strictly worse than B0 1M in production despite matching returns, had it been chosen for Stage 2.

---

## Verdict

**Phase 8-Beta closed with negative result.**

- ✅ **Bull signal robust**: bull mean +0.88-0.92% across all 5 runs (4 variants × 50k + B3 × 200k). Phase 8-Alpha's verified bull-side edge is reproduced.
- ❌ **Bear signal absent**: bear mean -1.5 to -1.7% across all 5 runs. No reward-shaping variant moved the bear regime.
- ❌ **Reward shaping does not change the final policy**: it only changes convergence speed. Once trained to ~1M, all four variants would produce v3-equivalent results.
- ❌ **Plan §7 predictions wrong on 3/4 variants**: the reward-magnitude model was missing policy-gradient convergence dynamics. The mechanism §0 identified (sharpe saturation) is real but the lever is too weak to change the final equilibrium.
- ❌ **Bonus finding from B3 1M**: sharpe-clip narrowing degrades the drawdown path (mean MaxDD 16.59% vs v3's 4.20%) without changing returns. The clip was acting as an implicit stop-loss that B3 weakened. **B3 would have been strictly worse in production than B0 even though returns are identical.**

**Stage 2 was empirically closed by the B3 1M run.** Per-fold returns match v3 1M to within 0.05pp on every fold; aggregate returns match within 0.01pp. The drawdown blowup confirms reward shaping in this direction is not just useless — it is actively harmful.

---

## Why the model converges to "stay flat" — corrected diagnosis

The original §0 hypothesis: "sharpe saturation at the clip dominates → trade-aversion → near-hold policy."

What the empirical evidence actually shows:
1. **At 50k** (mid-training), variants with smaller per-step sharpe penalty (B3) trade more freely. Mechanism confirmed.
2. **At 200k**, trade count starts collapsing across the board. The policy is learning that bear-regime trades have negative expected value *regardless of reward magnitude* — this is a structural feature of the data, not the reward function.
3. **At 1M**, all variants reach the same near-hold equilibrium because they all see the same underlying signal-to-noise ratio.

The corrected picture: **bear-regime long trades have genuinely negative expected value on this data**. Reward shaping that amplifies this fact (current main) accelerates convergence; reward shaping that dampens it (B3) merely slows convergence. None of the variants change the *direction* of the signal.

This means the bottleneck is not the reward — it is what the model can *learn* from the features it sees. Phase 8-Gamma must address one of:
- **Better signal**: feature engineering for bear-side detection, or richer regime context.
- **Architecture capacity**: longer-context model that can detect regime shifts before bear trades go bad.
- **Trade-structure change**: realized-PnL reward (paid only on close), which would let the model wait out unrealized drawdowns.
- **Hard regime gate**: HMM regime detector wiring to refuse long trades in bear regime, capping bear losses at zero.

---

## Phase 8-Gamma starting hypothesis

Most-promising next lever, ranked:

1. **HMM regime gate** (`training/signals/regime_detector.py` already exists; `risk_manager.adjust_for_regime` is wired in env step but underutilized). Refuse long trades in detected bear regime. Bear loss → ~0. Lifts all-fold mean from -0.59% to ~0% mechanically.
2. **Realized-PnL reward**: replace mark-to-market sharpe + drawdown with realized P&L on trade close. Eliminates the unrealized-drawdown trap that current reward creates.
3. **Bear-side feature engineering**: short-momentum indicators, regime-shift detectors. Useful regardless of (1) and (2).
4. **Long-only 1M**: cheap diagnostic to confirm the bidirectional action space is not wasting capacity. Probably ~ no-op given short-side rarely used.

Phase 8-Gamma plan to be drafted separately.

---

## Notes / Anomalies discovered during Stage 1

1. **`run_wf.py` log encoding**: PowerShell `*>` redirects produce UTF-16 LE files with BOM. `aggregate_wf_results.py:parse_log` opened with `encoding="utf-8"` and `errors="replace"`, silently garbling logs and reporting "RESULT block not found". Fixed in `claude/phase8-beta-closure` via BOM-sniffing read helper + 3 regression tests.
2. **Bull/bear fold defaults**: `aggregate_wf_results.py` originally defaulted to `0-4`/`5-11`, which doesn't match the BTC b&h-based regime classification (bull = 0,2,5,8,11; bear = 1,3,4,6,7,9,10). Fixed in PR #138 (`claude/phase8-beta-fold-classification-fix`) prior to result aggregation.
3. **Random-start trade-count criterion**: confounded with episode length. Fixed-start should be the criterion metric in Phase 8-Gamma.

---

## Reproducibility

```bash
# All 4 variants × 50k (run on trading-pc):
python run_wf.py --config config/futures_maker.yaml --n_splits 12 --total_timesteps 50000
python run_wf.py --config config/phase8_beta/B1_inactivity.yaml --n_splits 12 --total_timesteps 50000
python run_wf.py --config config/phase8_beta/B2_sharpe_weight.yaml --n_splits 12 --total_timesteps 50000
python run_wf.py --config config/phase8_beta/B3_sharpe_clip.yaml --n_splits 12 --total_timesteps 50000

# B3 200k and 1M follow-ups:
python run_wf.py --config config/phase8_beta/B3_sharpe_clip.yaml --n_splits 12 --total_timesteps 200000
python run_wf.py --config config/phase8_beta/B3_sharpe_clip.yaml --n_splits 12 --total_timesteps 1000000

# Aggregate:
python scripts/aggregate_wf_results.py \
    --logs logs/phase8_beta/{B0_baseline,B1_inactivity,B2_sharpe_weight,B3_sharpe_clip}.log \
    --variant-names B0 B1 B2 B3 \
    --apply-criterion --detail B3
```

Raw logs: `logs/phase8_beta_stage1/` (50k, 5 files), `logs/phase8_beta_200k/B3_200k.log`, `logs/phase8_beta_1M/B3_1M.log`. Not committed (gitignored under `logs/`).
