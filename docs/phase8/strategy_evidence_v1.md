---
generated_at: 2026-04-29T12:00:00+00:00
walk_forward_period: "2024-04 to 2026-04 (real, BTCUSDT_1h, 12 expanding folds × 1M timesteps)"
data_source: data/BTCUSDT_1h.csv (17520 rows)
agent: CVaRPPO (sb3cvarppo, default config/base.yaml)
go_no_go_status: "NO-GO (preliminary, baseline-failure)"
metrics:
  oos_total_return_per_episode_mean: -0.054
  oos_total_return_per_episode_min: -0.084
  oos_total_return_per_episode_max: -0.022
  mean_max_drawdown: 0.184
  n_folds: 12
  baseline_btc_buy_hold_period_return: 2.0  # ~+200% over period
  oos_sharpe_mean_raw: -3060.18  # unstable, see notes
caveats:
  - "oos_total_return in original code was np.sum of 5 eval episodes; per-episode = sum / 5"
  - "Sharpe metric unstable: 6/12 folds had std≈0 (deterministic policy → identical episodes); fixed in claude/a0-no-go-evidence via random_start_eval but does not change NO-GO conclusion"
  - "Buy-and-hold BTC over the same period returned ~+200%; bot lost on every fold"
---

# Strategy Evidence Pack v1

**Generated**: 2026-04-29T12:00:00+00:00
**Walk-forward period**: 2024-04 to 2026-04 (real, BTCUSDT_1h, 12 expanding folds × 1M timesteps)
**Folds**: 12
**Data source**: `data/BTCUSDT_1h.csv` (17520 rows, 1h bars)
**Agent**: CVaRPPO (sb3cvarppo, default `config/base.yaml`)

> This document is the authoritative evidence record required before `exchange_mode: live`.
> Operator GO/NO-GO decision is made after reviewing all sections.
> **Status: NO-GO (preliminary, baseline-failure)** — see Final Verdict below.

---

## A0.1 Walk-Forward Results

### Per-Fold Metrics

| Fold | Train rows | Test rows | IS Sharpe | OOS Sharpe (raw) | OOS Max DD | OOS return /episode |
|------|-----------|-----------|-----------|------------------|------------|---------------------|
| 0  | 8760  | 729 | -725.0  | -2348.6         | 16.7% | -4.5% |
| 1  | 9489  | 729 | -745.3  | 0.0 (std≈0)     | 28.1% | -7.9% |
| 2  | 10218 | 729 | -898.7  | 0.0 (std≈0)     | 14.2% | -3.8% |
| 3  | 10947 | 729 | -884.7  | 0.0 (std≈0)     | 20.7% | -5.6% |
| 4  | 11676 | 729 | -1106.1 | 0.0 (std≈0)     | 29.5% | -8.4% |
| 5  | 12405 | 729 | -4101.1 | 0.0 (std≈0)     | 11.7% | -3.0% |
| 6  | 13134 | 729 | -1437.7 | 0.0 (std≈0)     | 26.2% | -7.3% |
| 7  | 13863 | 729 | -735.4  | -3370.7         | 20.4% | -5.5% |
| 8  | 14592 | 729 | -631.3  | -14996.5        | 15.3% | -4.1% |
| 9  | 15321 | 729 | -1155.2 | -6625.9         | 11.1% | -2.9% |
| 10 | 16050 | 729 | -1175.9 | -6277.1         | 18.8% | -5.1% |
| 11 | 16779 | 729 | -1147.3 | -3103.3         |  8.6% | -2.2% |
| Mean | — | — | -1228.6 | (unstable)      | 18.4% | **-5.4%** |

> OOS Sharpe raw: 6/12 folds show std≈0 because deterministic policy produces identical
> episodes from the same fixed start. Fixed via `random_start_eval=True` (this branch),
> but does not change the NO-GO conclusion — all variants remain negative.

### Aggregate

| Metric | Value |
|--------|-------|
| Mean per-episode OOS return | -5.4% |
| Range | -2.2% to -8.4% |
| Mean max drawdown | 18.4% |
| Folds with positive return | 0 / 12 |
| BTC buy-hold (same period) | ~+200% |

---

## A0.2 Statistical Confidence

| Test | Value | Threshold | Pass? |
|------|-------|-----------|-------|
| Net Sharpe (per-episode return / std) | NEGATIVE (mean -5.4%, std small) | > 0.5 | ❌ |
| Stability (OOS / IS Sharpe ratio) | 2.49 | > 0.5 | technically pass but both negative — meaningless |
| All 12 folds positive return | 0/12 | ≥ 8/12 | ❌ |
| Outperform BTC buy-hold | NO | yes | ❌ |
| Mean max drawdown | 18.4% | < 30% | ✅ (only "pass") |

> Stability ratio 2.49 is a red herring: OOS Sharpe is less negative than IS Sharpe,
> but both are deeply negative. The ratio is meaningless in this regime.

---

## A0.3 Regime-Conditional Breakdown

> **Skipped**: HMM regime labelling was not run on the 1M overnight experiment.
> Pending Track C re-run with redesigned strategy.

---

## A0.4 Baseline Comparisons

| Strategy | Period return | Bot mean OOS return | Beats bot? |
|----------|--------------|---------------------|-----------|
| BTC buy-and-hold | ~+200% (2024-04 → 2026-04) | -5.4% per episode | ❌ bot loses decisively |

> The simplest possible baseline — buy BTC and hold — returned approximately +200% over the
> same period during which the bot lost money on every single fold. This is the primary
> NO-GO signal, independent of Sharpe metric instability.

---

## A0.5 Agent Contribution Decomposition

> **Deferred to Track C strategy redesign.**
> Agent ablation not meaningful when all agents fail to beat buy-and-hold.

---

## A0.6 Reality Gap

> Shadow-mode 72h drill 2026-04-26 → 2026-04-29 completed separately
> (see `docs/phase7/week85_72h_20260426.md`): 0 halts / 65 chaos faults survived.
> Infrastructure is sound. P&L not meaningful (synthetic chaos feed, not live market).
> This section will be relevant again after Track C produces a viable strategy.

---

## A0.7 Reward / Cost Function Audit

> **Deferred to Track C.**
> Current: `log_return + 0.1 × sharpe_proxy + DD_penalty`, fee=0.1%, slippage=0.05%.
> Re-audit required as part of strategy redesign — reward shaping is a candidate root
> cause (see Task B diagnostic in `docs/phase8/diagnostic_long_only_no_fee.md`).

---

## Final Verdict: **NO-GO (preliminary, baseline-failure)**

**Decision date**: 2026-04-29
**Decision basis**:
1. All 12 OOS folds returned negative mean per-episode return (-2.2% ~ -8.4%, mean -5.4%)
2. Same period BTC buy-and-hold ~+200% — bot decisively underperforms simplest baseline
3. Net Sharpe threshold (> 0.5) impossible to meet given negative mean returns
4. Random-start sanity experiment (50k × 1 fold, 2026-04-29) confirmed result is not metric artefact

**Implication for plan**: Phase 8 Track B-F all blocked. Trigger "A0 NO-GO 분기" in
`/Users/skylar/.claude/plans/phase8-restructured.md` — Phase 8-Alpha new plan
required (operator + Opus collaboration).

**Next concrete step**: Quick diagnostic experiment (Task B in `a0-no-go-handoff.md`) —
long-only + sharpe_weight=0 + fee=0, 50k × 1 fold, to identify which lever is the
strongest contributor to the failure. Results in `docs/phase8/diagnostic_long_only_no_fee.md`.

---

*Real data run: overnight 2026-04-28 → 2026-04-29, branch `claude/wf-metric-units`, `run_wf.py` (n_splits=12, total_timesteps=1M).*
*Review `docs/phase8/README.md` for Phase 8 GO/NO-GO branch criteria.*
