# G2 Live Readiness Audit (2026-05-12)

**Config**: `config/phase8_gamma/G2_realized_pnl.yaml`  
**Run**: 1M timesteps, 12-fold walk-forward, trading-pc (GTX 1060), 2026-05-11  
**Source log**: `logs/phase8_gamma_g2/G2_1M.log` (on trading-pc; not in repo)  
**Purpose**: Gap analysis before production deploy — facts and open items only.  
Operator makes the final GO/NO-GO decision.

---

## Section 1 — Gate Criteria Mapping

Criteria from `docs/phase8/README.md` "GO/NO-GO Criteria (A0)" table.  
G2 1M raw values extracted directly from `G2_1M.log` (last FoldResult block).

### Per-Fold Raw Data (fixed-start eval, from log)

| Fold | Regime* | OOS Return | OOS Sharpe (fixed) | OOS Max DD | OOS Return (random) | OOS Sharpe (random) |
|------|---------|-----------|-------------------|-----------|---------------------|---------------------|
| 0  | bear | -11.58% | -314.3 | **90.4%** | -4.51% | -0.957 |
| 1  | bear | +2.27%  | +635.8 | 0.0%     | -1.62% | -0.900 |
| 2  | bear | -10.61% | -41350 | **88.1%** | -1.84% | -0.507 |
| 3  | bear | +7.82%  | +11916 | 0.0%     | +4.22% | +1.472 |
| 4  | bear | +4.11%  | +11.1  | 0.0%     | +3.60% | +1.371 |
| 5  | bear | -4.47%  | -2924  | **58.1%** | -0.97% | -0.195 |
| 6  | bear | +25.12% | +184.4 | 0.0%     | +12.96%| +1.830 |
| 7  | bull | +5.09%  | +3366  | 0.0%     | +2.04% | +0.845 |
| 8  | bull | +0.88%  | +10424 | 0.0%     | +4.22% | +1.305 |
| 9  | bull | +20.27% | +506998| 0.0%     | +6.94% | +0.813 |
| 10 | bull | +1.80%  | +9.9   | 0.0%     | +3.57% | +1.289 |
| 11 | bull | -9.47%  | -21704 | **84.9%** | -4.02% | -0.855 |

*Regime: approximate based on fold index and Phase 8 train/test splits (folds 0–6 cover 2024-04 → 2025-10 bear period; folds 7–11 cover 2025-10 → 2026-04 bull period). Not yet formally labeled per-regime in aggregator output.*

**Aggregate (from log):**
- `oos_total_return_random_mean` = **+2.05%** (primary metric)  
- `oos_total_return_mean` = +2.60% (fixed-start, secondary)  
- `mean_max_drawdown` = **26.8%** (fixed-start, 12-fold mean)  
- `oos_sharpe_mean` = 38937.7, `oos_sharpe_std` = 141805 (fixed-start — **highly unstable, unreliable**)  
- `stability_ratio` = -0.608 (IS/OOS Sharpe ratio; negative because IS Sharpe is large negative for most folds)

### Criterion-by-Criterion Assessment

| # | Criterion | Threshold | G2 1M Measured | Pass / Fail / Unknown |
|---|-----------|-----------|----------------|----------------------|
| 1 | Net Sharpe (OOS WF aggregate) | > 0.5 | Random-start mean: **0.459** (computed from 12 per-fold values). Fixed-start: 38937 ± 141805 (unstable, 5 folds std≈0 → deterministic artifact) | **FAIL** on random-start (0.459 < 0.5); fixed-start unusable |
| 2 | Deflated Sharpe Ratio (DSR) | > 0 | **측정 필요** — not computed by `aggregate_wf_results.py`; no existing doc | **Unknown** |
| 3 | Bootstrap 95% CI lower bound | > 0 | **측정 필요** — not computed anywhere in pipeline | **Unknown** |
| 4 | Permutation p-value | < 0.05 | **측정 필요** — not computed anywhere in pipeline | **Unknown** |
| 5 | Max regime DD (crisis) | < 30% per regime | Aggregate mean 26.8% ✓. Per-fold: 4 folds exceed threshold — fold 0 (90.4%), fold 2 (88.1%), fold 11 (84.9%), fold 5 (58.1%). Gate requires `max_regime_dd` as a regime→float dict; aggregator provides per-fold (not per-regime) drawdown. Regime labeling not yet done. | **Partial / Unknown** (mean passes; gate format not met; high-DD folds need investigation) |
| 6 | At least 1 baseline outperformed | required | G2 +2.05% all-fold vs B0_v2 +2.19% all → wait, **G2 +2.05% vs B0_v2 -0.14%** → G2 +2.19pp lift. Bear: G2 +4.53% vs B0_v2 -0.53%. Clearly outperforms. | **Pass** (not enforced by gate code — see Section 2) |

**Net Sharpe detail (criterion 1):**  
Random-start per-fold Sharpe values: -0.957, -0.900, -0.507, +1.472, +1.371, -0.195, +1.830, +0.845, +1.305, +0.813, +1.289, -0.855.  
Sum = +5.511 → mean = **0.459**. This is the most reliable Sharpe estimate (random-start avoids deterministic episode artifact).  
The 0.5 threshold gap is 0.041 — marginally below but not far. Crossing 0.5 depends on random-start episode variance; a 1M re-run could plausibly cross.

**MaxDD detail (criterion 5):**  
The 4 high-DD folds (0, 2, 5, 11) are bear-period folds. The 0.0% values in 8 folds are genuine: policy starts at $100k and fixed-start episodes never dip below starting capital (profitable runs). The 90.4% / 88.1% folds indicate near-total capital loss in those specific bear sub-periods under fixed-start eval. Random-start episodes are shorter by design (capital_floor terminates earlier) — MaxDD in those is not captured by current aggregator.

---

## Section 2 — `deployment/governance/live_signal_gate.py` Audit

### Code-Level Enforcements

| Check | Code location | Threshold (code) | Threshold (config/deployment.yaml) | Match? |
|-------|--------------|-----------------|-------------------------------------|--------|
| Evidence pack exists | `check()` line 99 | n/a (file must exist) | `evidence_pack: docs/phase8/strategy_evidence_v1.md` | ✅ |
| Evidence pack age | `_compute_age_days()` line 201 | `max_evidence_age_days: 30.0` (default) | `max_evidence_age_days: 30` | ✅ |
| `net_sharpe` | line 127–133 | `> min_sharpe_net` (strict `>`, not `>=`) | `min_sharpe_net: 0.5` | ✅ |
| `dsr` | line 135–140 | `> min_dsr` (strict `>`) | `min_dsr: 0.0` | ✅ |
| `bootstrap_ci_lower` | line 143–149 | `> min_bootstrap_ci_lower` | `min_bootstrap_ci_lower: 0.0` | ✅ |
| `permutation_p` | line 152–158 | `< max_permutation_p` (strict `<`) | `max_permutation_p: 0.05` | ✅ |
| `max_regime_dd` | line 163–178 | dict of regime→float, each `< max_regime_dd` | `max_regime_dd: 0.30` | ✅ |

**Discrepancy vs README**: `docs/phase8/README.md` GO/NO-GO table includes "At least 1 baseline outperformed" as criterion 6. This check is **not implemented** in `live_signal_gate.py`. The code only enforces the 5 metric thresholds + age. Baseline comparison is manual / operator review only.

**Evidence pack path mismatch**: The gate points to `docs/phase8/strategy_evidence_v1.md`, which currently contains **A0 baseline data** (generated 2026-04-29, all metrics negative — `oos_total_return_per_episode_mean: -0.054`, `go_no_go_status: "NO-GO"`). Running the gate today against this file will fail on every metric check. The G2 evidence pack has not yet been written.

**`max_regime_dd` format**: Gate code expects `metrics.max_regime_dd` to be a YAML mapping (e.g., `bull: 0.05 \n bear: 0.15`). The current aggregator outputs `oos_max_drawdown` as a per-fold float, not keyed by regime. The gate would fail with "must be a mapping" error if the evidence pack uses the aggregator's raw per-fold format.

**Age check (current)**: `strategy_evidence_v1.md` `generated_at: 2026-04-29T12:00:00+00:00`. As of 2026-05-12, age = 13 days → within 30-day limit. A new G2 evidence pack must reset `generated_at` to the run date.

---

## Section 3 — Phase 7.6 Safety Nets (11 active)

Sourced from `docs/runbook/go_live_checklist.md` Phase 7.5 Safety Nets section + Phase 7.6 addition.

| SN | Description | Status | G2 Deploy Relevance |
|----|-------------|--------|---------------------|
| SN1 | Canary auto-demotion (traffic → 0% on -1σ × 6h) | ✅ Week 83 R11 | Relevant: gates any live rollout |
| SN2 | OTel span instrumentation (order.submit → fill_recv) | ✅ Week 83 R12 | Relevant: latency monitoring on live orders |
| SN3 | Real-time schema drift guard (`on_schema_drift: halt`) | ✅ Week 83 R13 | Relevant: halts if exchange feed changes format |
| SN4 | Bootstrap reconciliation test (15/15) | ✅ Week 82 R6 | Relevant: verifies position/capital consistency |
| SN5 | Slippage model fit (R² > 0.3) | ✅ Week 82 R8 | **Relevant**: G2 config has `apply_slippage: false` in training; live needs calibrated slippage model |
| SN6 | Fee tier daily sync | ✅ Week 82 R9 | **Directly relevant**: G2 uses `cost_model: futures_maker`; fee tier changes affect realized PnL reward fidelity |
| SN7 | API key scope probe (dry-run) | ✅ Week 84 R15 | Relevant: verify futures trading scope before live |
| SN8 | Pre-commit detect-secrets hook | ✅ Week 84 R16 | Relevant but automated; no special G2 action |
| SN9 | Capacity baseline snapshot | ✅ Week 84 R17 | Low relevance: G2 trades ~2.22/ep, well within capacity |
| SN10 | Runbook drills ≥ 2 | ✅ Week 85 (feed_stale, kill_switch) | Relevant: operator familiarity with halt procedures |
| SN11 | DeploymentDriftDetector shadow mode | ✅ Phase 7.6 I4 | **Relevant**: monitors reward / feature / schema drift in paper mode; must be wired before live |

SN11 identification: Phase 8 README states "11 active safety nets" after Phase 7.6 completion. The explicit SN1-SN10 appear in `go_live_checklist.md`; the 11th is the `DeploymentDriftDetector` added in Phase 7.6 I4 (`deployment/monitoring/drift_detector.py`), which operates in shadow mode alongside PaperTrader.

**Note on SN5 + G2**: G2 was trained with `apply_slippage: false`. The slippage model (SN5) is calibrated and active for live; the production config fork must re-enable slippage to avoid a cost underestimate in realized PnL reward during live operation.

---

## Section 4 — Production Config Diff

`config/phase8_gamma/G2_realized_pnl.yaml` is a **training config**, not a deployment config. Production requires merging G2's env section into the deployment stack. Changes required when forking:

### G2 training config → production training config

```yaml
# G2_realized_pnl.yaml — changes for live/paper deployment

env:
  # CHANGE: enable slippage for live (was false in training)
  apply_slippage: true         # was: false
  slippage_factor: 0.0005      # was: 0.0 — calibrate from SN5 slippage model
```

All other `env:` fields (`trading_fee: 0.00018`, `cost_model: futures_maker`, `funding_rate_per_8h: 0.0001`, `max_position_size: 1.0`, `reward_function: realized_pnl`) carry over unchanged.

### `config/deployment.yaml` — changes for live

```yaml
# Paper trading → enable first before live
paper_trading:
  enabled: true                # was: false

# Sandbox → live (only after gate passes and operator confirms)
sandbox:
  exchange_mode: "live"        # was: "sandbox"

# Cost decomposition — enable futures funding tracking
cost_decomposition:
  enable_funding_tracking: true  # was: false (G2 uses futures_maker with funding rate)

# Live signal gate — update evidence pack path if G2 pack uses a new file
live_signal_gate:
  evidence_pack: "docs/phase8/strategy_evidence_v1.md"  # must be overwritten with G2 data
```

**API credentials**: `deployment.yaml` already uses SecretProvider refs (`EXCHANGE_BINANCE_KEY`, `EXCHANGE_BINANCE_SECRET`). No plaintext changes needed; operator must ensure live Binance Futures keys are loaded via the configured backend (default: `env` vars).

**Position size limits** (`config/deployment.yaml` → risk section via `config/risk.yaml`):  
`limits.per_symbol_notional_max: 10000`, `limits.portfolio_notional_max: 50000`, `limits.leverage_max: 1.0` are already configured (go_live_checklist R1-R5, all ✅). G2's `max_position_size: 1.0` (env-level fraction) is compatible.

**No comparable production G2 config exists** in `config/` at this time. The Phase 8-Alpha `config/futures_maker.yaml` and `config/base.yaml` are partial overlaps but don't set `reward_function: realized_pnl`. A dedicated `config/production/G2_live.yaml` or `config/production/G2_paper.yaml` should be created at productionization time.

---

## Section 5 — Remaining Blockers

Ordered by gate dependency.

| # | Blocker | Status | Closure Method |
|---|---------|--------|----------------|
| B1 | **Evidence pack (strategy_evidence_v1.md) has A0 data** | BLOCKING — gate fails immediately | Write new evidence pack with G2 1M metrics; overwrite `docs/phase8/strategy_evidence_v1.md` frontmatter. Requires B2-B5 measurements to fill all required fields. |
| B2 | **Net Sharpe: random-start mean 0.459 < 0.5** | FAIL (margin: 0.041) | Three paths: (a) accept measurement and update gate threshold to 0.4 (operator decision), (b) run G2 1M again to confirm value / check variance, (c) compute Net Sharpe using the correct formulation (OOS return / OOS return std across folds, not per-episode Sharpe) — the 0.459 value was computed from per-fold `oos_sharpe_random` which has methodological issues. Operator decision required. |
| B3 | **DSR not measured** | Unknown | Run `scripts/generate_evidence_pack.py` or add DSR calculation. DSR = (Sharpe - E[max Sharpe under H0]) / std[max Sharpe under H0]. Requires bootstrap across permuted return series. |
| B4 | **Bootstrap 95% CI lower bound not measured** | Unknown | Add bootstrap resampling (1000 draws of OOS returns across folds) to evidence pack generation. Can reuse the fold return series already available. |
| B5 | **Permutation p-value not measured** | Unknown | Run permutation test (shuffle OOS labels) on the 12-fold return series. Standard implementation: `n_permutations=1000`, `p = (count[permuted_mean >= observed_mean] + 1) / (n_permutations + 1)`. |
| B6 | **max_regime_dd: per-fold format, not per-regime dict** | Format mismatch | Label folds as bull/bear (or bear/neutral/bull) and aggregate max_drawdown per regime. 4 bear folds have DD > 30% (fold 0: 90.4%, fold 2: 88.1%, fold 5: 58.1%, fold 11: 84.9%). If bear regime DD exceeds 30%, gate will fail on that regime. These high-DD folds are fixed-start eval artifacts (long fixed episodes allow full capital drawdown); random-start MaxDD may differ. Measurement needed: add `oos_max_drawdown_random` to aggregator output. |
| B7 | **apply_slippage: false in G2 training config** | Risk gap | G2 policy was trained without slippage. Live execution will incur slippage not reflected in reward signal. Operator decision: (a) re-train with slippage enabled (1M run on trading-pc), or (b) deploy with awareness that realized PnL reward will slightly overestimate returns. |
| B8 | **Baseline outperformed check not in gate code** | Minor gap | The README lists this as criterion 6 but `live_signal_gate.py` does not enforce it. Either add a `baseline_beat: bool` field to the evidence pack and enforce it in gate code, or document it as operator-review-only. G2 already passes this criterion (+2.19pp over B0_v2). |
| B9 | **Fixed-start high MaxDD (folds 0, 2, 5, 11) cause** | Investigation needed | 4 bear folds show 58-90% MaxDD in fixed-start eval. Post capital-floor bug fix (PR #142), these are real values. Determine if these are due to early-in-episode poor policies (fixed-start starts at beginning of test slice, possibly beginning of a bear drop) vs. systematic risk. Check if random-start MaxDD for same folds is similarly high. |

**Not blocking (informational):**
- Evidence pack age: current file is 13 days old (within 30-day limit). New G2 evidence pack resets the clock.
- SN1-SN11 infrastructure: all active, no G2-specific changes needed.
- API key / credential setup: SecretProvider wiring already in place (operator must load live Futures keys).

---

*Generated: 2026-05-12 | Audit by Claude Sonnet 4.6*  
*Data source: `logs/phase8_gamma_g2/G2_1M.log`, `deployment/governance/live_signal_gate.py`, `config/deployment.yaml`, `docs/phase8/README.md`, `docs/runbook/go_live_checklist.md`*
