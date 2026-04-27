---
generated_at: 2026-04-27T17:02:36.844817+00:00
walk_forward_period: "2025-04 to 2026-04 (simulated — replace with real walk-forward output after running scripts/run_full_pipeline.py)"
metrics:
  net_sharpe: 0.3492
  gross_sharpe: 0.3492
  dsr: 1.0000
  bootstrap_ci_lower: -0.5255
  bootstrap_ci_upper: 1.2308
  permutation_p: 0.2164
  max_regime_dd:
    trend: 0.2141
    range: 0.1967
    crisis: 0.2215
  n_folds: 5
  n_hyperopt_trials: 100
---

# Strategy Evidence Pack v1

**Generated**: 2026-04-27T17:02:36.844817+00:00
**Walk-forward period**: 2025-04 to 2026-04 (simulated — replace with real walk-forward output after running scripts/run_full_pipeline.py)
**Folds**: 5
**Hyperopt trials (N)**: 100

> This document is the authoritative evidence record required before `exchange_mode: live`.
> Operator GO/NO-GO decision is made after reviewing all sections.
> Automated gate thresholds: `deployment/governance/live_signal_gate.py` (A0.5).

---

## A0.1 Walk-Forward Results

### Per-Fold Metrics

| Fold | Period | Gross Sharpe | Net Sharpe | Sortino | Calmar | Max DD | Hit Rate | Avg Trade | Turnover |
|------|--------|-------------|-----------|---------|--------|--------|----------|-----------|----------|
| 0 | 2025-01-01 → 2025-02-01 | 0.961 | 0.752 | 1.277 | 1.022 | 17.8% | 51.2% | 0.00072 | 16.5% |
| 1 | 2025-02-01 → 2025-03-01 | 0.473 | 0.261 | 0.409 | 0.317 | 19.5% | 50.0% | 0.00025 | 34.4% |
| 2 | 2025-03-01 → 2025-04-01 | -1.262 | -1.502 | -2.342 | -0.764 | 41.3% | 45.2% | -0.00125 | 30.9% |
| 3 | 2025-04-01 → 2025-05-01 | 1.828 | 1.626 | 2.854 | 2.036 | 19.9% | 53.6% | 0.00161 | 36.1% |
| 4 | 2025-05-01 → 2025-06-01 | 0.530 | 0.330 | 0.552 | 0.258 | 32.2% | 52.4% | 0.00033 | 37.6% |

### Aggregate (OOS, all folds concatenated)

| Metric | Gross | Net-of-cost |
|--------|-------|------------|
| Sharpe | 0.349 | 0.349 |
| Bootstrap 95% CI | — | [-0.526, 1.231] |

---

## A0.2 Statistical Confidence

| Test | Value | Threshold | Pass? |
|------|-------|-----------|-------|
| Net Sharpe (OOS agg) | 0.3492 | > 0.5 | ❌ |
| DSR (N=100 trials) | 1.0000 | > 0 | ✅ |
| Bootstrap 95% CI lower | -0.5255 | > 0 | ❌ |
| Bootstrap 95% CI upper | 1.2308 | — | — |
| Permutation p-value | 0.2164 | < 0.05 | ❌ |

> Bootstrap: 10,000 resamples. Permutation: 10,000 sign-randomizations.
> DSR uses Bailey & López de Prado (2014) formula. N = 100 hyperopt trials.

---

## A0.3 Regime-Conditional Breakdown

HMM re-fit **per fold** (no label leakage). 3 states: Trend / Range / Crisis.

### Per-Fold Regime Table

| Fold | Regime | Sharpe | Max DD | N samples |
|------|--------|--------|--------|-----------|
| 0 | trend | -0.476 | 20.0% | 87 |
| 0 | range | 0.930 | 19.7% | 84 |
| 0 | crisis | 2.182 | 7.2% | 81 |
| 1 | trend | 1.367 | 7.6% | 76 |
| 1 | range | -0.365 | 9.8% | 79 |
| 1 | crisis | -0.104 | 14.1% | 97 |
| 2 | trend | -0.988 | 21.4% | 82 |
| 2 | range | -1.030 | 16.6% | 83 |
| 2 | crisis | -2.662 | 22.2% | 87 |
| 3 | trend | 0.857 | 12.8% | 85 |
| 3 | range | 2.614 | 15.7% | 76 |
| 3 | crisis | 1.382 | 14.1% | 91 |
| 4 | trend | 0.645 | 21.3% | 104 |
| 4 | range | 0.159 | 10.3% | 77 |
| 4 | crisis | 0.100 | 15.1% | 71 |

### Crisis Regime Max DD (across folds)

| Regime | Max DD (worst fold) | Threshold | Pass? |
|--------|---------------------|-----------|-------|
| trend | 21.4% | < 50.0% | ✅ |
| range | 19.7% | < 50.0% | ✅ |
| crisis | 22.2% | < 30.0% | ✅ |

**HMM leakage audit**: per-fold re-fit verified in code review (no shared HMM across folds).

---

## A0.4 Baseline Comparisons

| Strategy | Sharpe | Max DD | Sortino | Beats baseline? |
|----------|--------|--------|---------|----------------|
| Buy And Hold | 0.143 | 29.7% | 0.174 | ✅ |
| Ma Cross | 1.134 | 20.7% | 1.409 | ❌ |
| Mean Reversion | -10.475 | 918.0% | -15.365 | ✅ |

**Outperforms at least 1 baseline**: ✅

---

## A0.5 Agent Contribution Decomposition

Meta-controller average weight per agent (across folds, regime-conditional).

| Agent | Avg OOS Weight | trend | range | crisis |
|-------|---------------|-------|-------|--------|
| cvar_ppo | 0.323 | 0.323 | 0.323 | 0.323 |
| flag_trader | 0.147 | 0.147 | 0.147 | 0.147 |
| sac | 0.202 | 0.202 | 0.202 | 0.202 |
| td3 | 0.176 | 0.176 | 0.176 | 0.176 |

> Full agent ablation (A7) in `docs/phase8/agent_ablation_decision.md` (Week 92).
> FLAG-Trader ΔSharpe vs ensemble will be quantified there.

---

## A0.6 Reality Gap

> **Data insufficient** — paper run < 30 days of realized fills.
> This section will be completed in Evidence Pack v2 after ≥ 30 days of paper trading.
> Slippage model R² from calibration: > 0.3 (Phase 7.5 SN5).

---

## A0.7 Reward / Cost Function Audit

> See `docs/phase8/reward_audit.md` for full audit.

**Verdict**: Reward is **net-of-cost** (fees + slippage deducted from `current_capital`
before portfolio log-return is computed). No train-vs-deploy mismatch on reward definition.

---

## GO / NO-GO Summary

> **Operator decision required.** This section auto-fills the pass/fail per criterion.
> Final GO/NO-GO is an operator call, not automated.

| Criterion | Value | Pass? |
|-----------|-------|-------|
| Net Sharpe > 0.5 | — | ❌ |
| DSR > 0 | — | ✅ |
| Bootstrap CI lower > 0 | — | ❌ |
| Permutation p < 0.05 | — | ❌ |
| Crisis DD < 30% | — | ✅ |
| Outperforms ≥ 1 baseline | — | ✅ |

**Automated criteria**: ❌ NO-GO (3/6 criteria met)

| | |
|---|---|
| Operator decision | ☐ GO / ☐ NO-GO |
| Date | ________ |
| Signed by | ________ |
| Notes | ________ |

---

*Generated by `scripts/generate_evidence_pack.py` on 2026-04-27T17:02:36.844817+00:00.*
*Review `docs/phase8/README.md` for Phase 8 GO/NO-GO branch criteria.*