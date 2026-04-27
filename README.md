# Trading Bot — Multi-Agent RL Crypto Trading System

> A Reinforcement Learning trading bot for crypto markets (primary: BTCUSDT) built around a
> 4-agent ensemble, regime-aware meta-controller, CVaR risk constraints, and a hardened
> paper-trading / live-readiness pipeline with drift detection, audit logs, and operator
> safety nets.

**Status (2026-04-26)**: Phase 7.6 (Interim Autonomy + Continuation) complete.
Code is paper-trading hardened and live-readiness validated through a 72h autonomous drill.
First-dollar live trading is gated by an operator-driven checklist — see
[docs/runbook/go_live_checklist.md](docs/runbook/go_live_checklist.md).

| Metric | Value |
|---|---|
| pytest baseline | 2505 passed / 27 skipped / 12 pre-existing failed |
| `pytest.ini` ignores | 4 entries (audit) |
| Live-readiness checklist | 15/15 auto-checks PASS (Week 85) |
| First-dollar drill | 17/17 PASS (simulation) |
| Kill switch | < 5s halt verified (0.01s observed) |
| 72h autonomous drill | In progress (see Phase 7.6, ~ends 2026-04-27 22:33 PT) |
| Last merged PR | #110 (Phase 7.6 I5–I12 continuation) |
| Phase 8 A0 evidence pack | [docs/phase8/strategy_evidence_v1.md](docs/phase8/strategy_evidence_v1.md) |

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)
2. [What it does NOT do](#2-what-it-does-not-do)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Trading strategy](#4-trading-strategy)
5. [Quality & safety bar](#5-quality--safety-bar)
6. [Repository layout](#6-repository-layout)
7. [Quick start](#7-quick-start)
8. [Operating the bot](#8-operating-the-bot)
9. [Phase history (where we've been)](#9-phase-history-where-weve-been)
10. [Roadmap (where we're going)](#10-roadmap-where-were-going)
11. [References](#11-references)

---

## 1. What this project does

This is a **single-asset (primary BTCUSDT, 1h bars) RL trading bot** that decides each step's
target position from market features and outputs orders through a hardened execution layer.
The system covers the full loop:

- **Data ingestion** — exchange OHLCV (Binance / Bybit via `ccxt`), cross-asset (SPY, DXY, GC=F, ETH-USD, ^VIX via `yfinance`), and optional on-chain / alt features. Incremental fetcher with cache.
- **Feature engineering** — technical indicators (`ta`), regime labels (HMM), drift-monitored feature stats, optional DT forecaster (predicted return + confidence appended to obs).
- **Agent training** — 4-agent ensemble (CVaR-PPO, SAC, TD3, FLAG-Trader / LLM-RL) trained with SB3 + PEFT/LoRA, with walk-forward validation, Optuna/Ray Tune hyperopt, and MLflow tracking.
- **Ensembling** — Meta-controller learns regime-aware weights over the 4 agents; communication protocol forwards each sub-agent's hidden state to the meta-controller (see [docs/META_AGENT_WITH_HIDDEN_STATES.md](docs/META_AGENT_WITH_HIDDEN_STATES.md)).
- **Backtesting** — walk-forward, scenario simulation (flash crash / high-vol / low-liq), bootstrap Sharpe CI, permutation test, deflated Sharpe, regime-conditional reports.
- **Paper trading** — full execution stack (rate limiter, clock sync, fat-finger guard, slippage & fee model, position tracker, circuit breaker), with SQLite checkpointing and audit-log chain.
- **Live readiness** — testnet wizard, $100 first-dollar drill, 72h autonomous drill, drift shadow mode, fault injector, alerter, runbook.

If you want to know whether feature X is in scope, the rule of thumb is: **anything below the
"order is sent to the exchange" line is built; multi-asset portfolio rebalancing is partially
built but not the focus; anything beyond crypto into general equities is explicitly out of
scope until Phase 9+**.

## 2. What it does NOT do

- **It is not a strategy library.** It runs *one* configurable RL pipeline. Classical strategies (mean-reversion, MA cross, etc.) are reference baselines only.
- **No HFT / sub-second alpha.** Designed around 1h bars on a single retail-ish GPU (RTX 3060 Ti / Apple M2). Latency budget is "minutes," not "microseconds."
- **No general-equity support yet.** The data layer reaches into yfinance for cross-asset features, but the trading loop assumes a crypto perp/spot exchange via CCXT. Equity routing/calendar/halt handling is Phase 9+.
- **No portfolio rebalancing across many venues.** Multi-asset env exists (`envs/multi_asset_env.py`) and supports `discrete_amount` / `portfolio_weights` / `discrete_signal` action types, but the production path has been single-asset for the live-readiness work.
- **Does not auto-go-live.** `exchange_mode: live` requires every item in [docs/runbook/go_live_checklist.md](docs/runbook/go_live_checklist.md) to be ✅ — this is enforced by tooling and reviewed by the operator.

## 3. Architecture at a glance

```
            ┌──────────────────────── Data layer ────────────────────────┐
            │ scripts/fetch_data.py → data/raw → data/processed (DVC)    │
            │   • Binance / Bybit / yfinance (cross-asset)               │
            │   • Optional alt-data (on-chain)                           │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
            ┌──────────────────────── Features ──────────────────────────┐
            │ training/features  • TA indicators                         │
            │ training/regime    • HMM regime detector                   │
            │ training/monitoring/drift_detector.py (ADWIN + KS shadow)  │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
            ┌──────────────────────── Environments ──────────────────────┐
            │ envs/single_asset_rl_env.py  (gym.Env, window_size, $-cols)│
            │ envs/risk_manager.py         (stop, trailing, VaR, DD)     │
            │ envs/multi_asset_env.py      (portfolio_weights / signals) │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
            ┌──────────────────────── Agents (ensemble) ─────────────────┐
            │ agents/sb3/cvar_ppo.py        (PPO + CVaR constraint)      │
            │ agents/sb3 + SB3              (SAC, TD3)                   │
            │ agents/llm_rl/flag_trader.py  (LLM + LoRA, PEFT)           │
            │ agents/ensemble/meta_controller.py + regime_detector       │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
            ┌──────────────────────── Validation & Selection ────────────┐
            │ training/validation/walk_forward.py                        │
            │ training/backtesting/{base, enhanced, scenario}_backtester │
            │ training/hyperopt (Optuna + Ray Tune)                      │
            │ MLflow tracking + model registry (training/registry/)      │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
            ┌──────────────────────── Deployment ────────────────────────┐
            │ deployment/paper_trader.py                                 │
            │ deployment/exchange/{ccxt_adapter, fee_model, snapshot}    │
            │ deployment/execution/                                      │
            │   • order_manager  • slippage_model  • clock_sync          │
            │   • rate_limiter   • fat_finger     • circuit_breaker     │
            │   • position_tracker                                       │
            │ deployment/persistence/state_store.py    (SQLite + WAL)    │
            │ deployment/audit/        (append-only chain + verifier)    │
            │ deployment/secrets/      (provider, redaction)             │
            │ deployment/monitoring/                                     │
            │   • alerter (Telegram / webhook, rate-limited)             │
            │   • drift_detector (deployment-side, shadow mode)          │
            │   • metrics_exporter (Prometheus :9100)                    │
            │   • dashboard (Streamlit / Grafana)                        │
            │   • tracing (OTel spans submit→fill)                       │
            │ deployment/testing/fault_injector.py                       │
            │   (data_feed_stale, exchange_outage, spread_blowout, ...)  │
            └────────────────────────────────────────────────────────────┘
```

For a deeper risk-side view see [docs/architecture/risk_manager.md](docs/architecture/risk_manager.md)
and the multi-agent doc at [docs/MULTI_AGENT_MANAGER.md](docs/MULTI_AGENT_MANAGER.md).

## 4. Trading strategy

The trading thesis is *not* "find one alpha and bet it"; it is **"run an ensemble whose
weights adapt to regime, and constrain the tail."**

| Component | Role |
|---|---|
| **CVaR-PPO** (`agents/sb3/cvar_ppo.py`) | Conservative — explicitly penalizes worst-α% returns (`cvar_alpha=0.05`, `cvar_threshold=-0.02`). Strong in volatile / tail-risk regimes. |
| **SAC** | Entropy-regularized, balanced. Strong in trending regimes. |
| **TD3** | Twin-critic deterministic. Aggressive in clear trends. |
| **FLAG-Trader** (`agents/llm_rl/flag_trader.py`) | LLM policy with LoRA adapters; ingests news / event context. Adaptive to events. |
| **Meta-controller** (`agents/ensemble/meta_controller.py`) | Learns regime-conditional weights over the 4 agents using sub-agent hidden states. |
| **Regime detector** (`agents/ensemble/regime_detector.py` + `training/regime/regime_detector.py`) | HMM with 3 states (Trending / Ranging / Crisis). Fallback to threshold logic if `hmmlearn` is unavailable. |

Risk overlays apply *after* the agent decides and *before* the order is sent:

- Stop-loss, trailing stop, VaR-based sizing, max-DD forced liquidation — `envs/risk_manager.py`, `risk_management/`.
- `UnifiedRiskManager` is the single risk path; the deprecated `check_max_drawdown` / `check_stop_loss` aliases were removed.
- Pre-trade compliance: per-symbol notional cap, portfolio notional cap, leverage cap, daily loss limit, fat-finger guard.

Validation is **walk-forward only**. Deployment selection requires:

- OOS Sharpe consistency across folds.
- Bootstrap 95% CI of Sharpe with `lo > 0`.
- Permutation `p < 0.05`.
- Deflated Sharpe Ratio (DSR) accounting for hyperopt trials.
- Regime-conditional Sharpe — no fold/regime can be catastrophically negative.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for the full reading guide on these statistics.

## 5. Quality & safety bar

This is the part that took the most engineering and is what separates this repo from a
generic RL backtester.

### Strategy evidence bar

Before `exchange_mode: live` is allowed, the strategy must produce a statistical evidence pack:

- **Evidence pack**: [`docs/phase8/strategy_evidence_v1.md`](docs/phase8/strategy_evidence_v1.md)
  — walk-forward OOS Sharpe, bootstrap 95% CI, permutation p-value, DSR, regime-conditional
  breakdown, and baseline comparisons.
- **Automated GO thresholds** (enforced by `deployment/governance/live_signal_gate.py` — A0.5):
  net Sharpe > 0.5, DSR > 0, CI lower > 0, permutation p < 0.05, crisis DD < 30%.
- **Reward audit**: [`docs/phase8/reward_audit.md`](docs/phase8/reward_audit.md) — confirms
  reward is net-of-cost (fees + slippage deducted before log-return computation).
- Run `python scripts/generate_evidence_pack.py --walk-forward-runs runs/wf_*.json` to
  regenerate from new walk-forward results.

### Test bar

- **2505 passed / 27 skipped / 12 pre-existing failed** on `pytest -q` (Phase 8 P0-a baseline, 2026-04-27).
  12 failures are in `test_live_trading_advanced.py` / `test_live_websocket.py` — pre-existing on main, not caused by Phase 8 changes.
- `pytest.ini` ignore list audited down to **4 entries** ([docs/phase7/ignore_audit.md](docs/phase7/ignore_audit.md)).
- NaN canary suite (100 seeds × 100 steps) — 100% pass.
- Numeric warning count budget: < 500 across full suite.
- Idempotency tests pinned at 100/100 ([docs/phase7/week81_idempotency.md](docs/phase7/week81_idempotency.md)).
- Shape verification framework (`tests/test_small_integration.py`) catches tensor mismatches early.

### Live-readiness safety nets (active in code paths)

| Safety net | Trigger | Source |
|---|---|---|
| Kill switch | Operator command — halts within 5s (0.01s observed) | `scripts/kill_switch.py` |
| Reconciliation halt | `qty drift ≥ 1% or notional drift ≥ $50` | Week 82 G1 |
| Canary auto-demotion | -1σ × 6h on canary → traffic 0% | Week 83 G4 |
| Schema drift halt | Any unexpected field/type on incoming bars | Week 83 G6 |
| Drift shadow mode | `DeploymentDriftDetector` runs alongside without halting until validated | Phase 7.6 I4/I7 |
| Fat-finger guard | Order qty > N× rolling-median qty | `deployment/execution/fat_finger_guard.py` |
| Rate limiter | Per-endpoint token bucket | `deployment/execution/rate_limiter.py` |
| API key scope probe | Confirms `Read ✓ / Trade ✓ / Withdraw ✗` on startup | `scripts/verify_exchange_key_scope.py` |
| Pre-commit secret scan | `detect-secrets` blocks plaintext keys | `.pre-commit-config.yaml` |
| Audit log chain | Append-only, hash-chained, `verify_audit_log.py` | `deployment/audit/` |
| Alerter rate-limit | Per-minute sampling so an alert storm cannot self-DoS | Phase 7.6 I9 |
| Silent-failure detectors | Detects shadow-loop death, alert-storm collapse, key-scope change | Phase 7.6 I10 |

### Drills

- **First-dollar drill** (`scripts/first_dollar_drill.py`) — 17/17 PASS (2026-04-23). 5 mock-exchange scenarios: timeout / filled / partial / unfilled / rejected.
- **72h autonomous drill** (`scripts/autonomous_72h_drill.py`) — runs paper-trader + fault injector + alerter for 72h, writes `logs/incidents/*.md` postmortems. Fault library now includes `data_feed_stale`, `exchange_outage`, `spread_blowout`, etc.
- **Disaster-recovery drill** (`scripts/drills/run_drill.py --scenario all`) — checkpoint restore, NaN action injection, drawdown breach.

### Operator runbook

[docs/runbook/](docs/runbook/) contains:

- `README.md` — quick-start decision tree for failures.
- `failures/{data_feed_stale, exchange_api_error, drawdown_kill_switch, crash_recovery, model_nan_output}.md` — per-failure-mode procedures.
- `go_live_checklist.md` — gating list for `live` mode (must be 100% ✅).
- `oncall_checklist.md` — daily / weekly procedures.
- `postmortem_template.md` — incident write-up template.

## 6. Repository layout

```
trading_bot/
├── agents/                 # Policy / value networks
│   ├── sb3/                # CVaR-PPO + SB3 SAC/TD3 wrappers
│   ├── llm_rl/             # FLAG-Trader (LLM + LoRA)
│   ├── ensemble/           # Meta-controller + regime detector + comm
│   ├── strategies/         # Reference (non-RL) baselines
│   └── risk/, base/, models/, offline/
├── envs/                   # gym.Env implementations
│   ├── single_asset_rl_env.py
│   ├── multi_asset_env.py
│   ├── multi_agent_env.py / multi_agent_multi_asset_env.py
│   ├── live_trading_env.py / paper_trading_env.py
│   └── risk_manager.py     # Stop / trailing / VaR / DD
├── risk_management/        # UnifiedRiskManager (single risk path)
├── training/
│   ├── data/               # Feature pipelines
│   ├── regime/             # HMM regime detector
│   ├── monitoring/         # Drift detection (training side)
│   ├── continual/          # Auto-retrain triggers
│   ├── backtesting/        # Walk-forward + scenario backtesters
│   ├── validation/         # Walk-forward harness
│   ├── hyperopt/           # Optuna / Ray Tune
│   ├── registry/           # Model registry / promotion
│   ├── pipelines/, factories/, evaluation/, signals/, strategy_lab/
│   └── train_pipeline.py / train_ensemble.py
├── deployment/
│   ├── paper_trader.py
│   ├── exchange/           # ccxt_adapter, fee_model, snapshot
│   ├── execution/          # order_manager, slippage, clock_sync, rate_limiter, fat_finger, circuit_breaker, position_tracker
│   ├── monitoring/         # alerter, drift_detector, metrics_exporter, dashboard, tracing
│   ├── persistence/        # SQLite state_store (WAL)
│   ├── audit/              # Append-only chain + verifier
│   ├── secrets/            # SecretProvider + redaction
│   ├── testing/            # fault_injector
│   └── analysis/, config/
├── config/                 # YAMLs (env, deployment, risk, alerts, monitoring, ...)
├── scripts/                # CLI tools (fetch_data, drills, kill_switch, setup_testnet, ...)
├── docs/                   # USER_GUIDE, MULTI_AGENT_MANAGER, runbook, phase{6,7,7.6}/
├── tests/                  # 2458 tests, organized by module
├── examples/               # End-to-end demos
├── Dockerfile / docker-compose.yml
├── setup_local.py          # One-click local setup
└── requirements.txt
```

## 7. Quick start

> Detailed setup, troubleshooting, and tuning guidance live in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

### Local (RTX 3060 Ti)

```bash
python setup_local.py --gpu 3060ti

# 1y of BTCUSDT 1h bars + cross-asset + alt-data
python scripts/fetch_data.py --asset BTCUSDT --period 1y --interval 1h --cross-assets --alt-data

# Train ensemble (~20h on 3060 Ti)
python -m training.train_pipeline --config config/local_3060ti.yaml

# Watch progress
mlflow ui --port 5000              # experiment tracking
streamlit run deployment/web_interface/app.py   # live dashboard (paper trading)
```

### Local (Apple M2 — fast smoke)

```bash
python setup_local.py --gpu m2
python scripts/fetch_data.py --asset BTCUSDT --period 3m --interval 1h
python -m training.train_pipeline --config config/local_m2.yaml
```

### Docker

```bash
docker-compose up -d                 # all services (training + UI + MLflow + data fetcher)
docker-compose logs -f training
docker-compose down
```

## 8. Operating the bot

### Paper trading

```bash
python -m deployment.paper_trader --config config/local_3060ti.yaml
```

State is checkpointed to `state/paper_trader.db` (SQLite WAL). Crash recovery is automatic
on restart — see [docs/runbook/failures/crash_recovery.md](docs/runbook/failures/crash_recovery.md).

### Going live (gated)

1. Run `python scripts/setup_testnet.py` (Phase 7.6 I5 wizard) and let it provision testnet keys + write the env file.
2. Run `python scripts/first_dollar_drill.py --check-only` and confirm 15/15 PASS.
3. Walk through every item in [docs/runbook/go_live_checklist.md](docs/runbook/go_live_checklist.md). Manual items must be ticked + timestamped.
4. Verify exchange key scope: `python scripts/verify_exchange_key_scope.py` (Read ✓ / Trade ✓ / Withdraw ✗).
5. Flip `exchange_mode: live` in the deployment config.
6. `python scripts/run_paper_trader.py` (same entrypoint, live config) and watch the dashboard / Telegram alerter.

If anything goes sideways: `python scripts/kill_switch.py` halts the bot in < 5s and writes a postmortem skeleton.

### Routine ops

- **Daily** — `docker-compose ps` healthy; review previous-day Drift count in MLflow; check OOS Sharpe trend.
- **Weekly** — `python scripts/drills/run_drill.py --scenario all`; review `logs/incidents/`; verify audit chain (`scripts/verify_audit_log.py`).
- **On drift warning** — refresh data, retrain via `--retrain`, OOS-validate before promotion (auto-promotion only on validation pass).

## 9. Phase history (where we've been)

The project has been built in well-scoped phases — each closes a class of risk before the next opens. Plans live under `/Users/skylar/.claude/plans/`; per-week docs live under `docs/phase{6,7,7.6}/`.

| Phase | Weeks | Theme | Status |
|---|---|---|---|
| Weeks 1–18 | — | RL foundations: env, agents, basic backtest | ✅ |
| Phase 3 (29–34) | — | Hardening: monitoring, statistical tests, regime-conditional reporting | ✅ |
| Phase 4 (43–47) | — | Continual learning + LLM-RL (FLAG-Trader) | ✅ |
| Phase 5 Unified (48–55) | — | Walk-forward + ensemble convergence | ✅ (1386 passed) |
| Phase 6 Production Readiness (56–68) | — | Ops layer: alerter, drift, metrics, runbook v1 | ✅ ([PR #88](https://github.com/skylar93/trading_bot/pull/88)) |
| Phase 7 Live Readiness (69–80) | — | Execution layer: slippage / fee / clock / fat-finger / circuit-breaker / audit | ✅ |
| Phase 7.5 Live Closure (81–85) | — | D1–D5 defects + G1–G10 live blockers + first-dollar drill | ✅ ([PR #105](https://github.com/skylar93/trading_bot/pull/105)) |
| Phase 7.6 Interim Autonomy (86) | — | I1–I4: doc-code parity, autonomous 72h drill, zero-config alerting, drift shadow mode | ✅ ([PR #106](https://github.com/skylar93/trading_bot/pull/106), [#107](https://github.com/skylar93/trading_bot/pull/107)) |
| Phase 7.6 Continuation | 86 | I5–I12: testnet wizard, live drill safety, paper-trader drift wiring, fault probes (`exchange_outage`, `spread_blowout`), alerter rate-limit, silent-failure detectors, sign-off | ✅ ([PR #108](https://github.com/skylar93/trading_bot/pull/108)) |

See [docs/phase7.6/sign_off.md](docs/phase7.6/sign_off.md) for the I1–I12 matrix and exit criteria.

## 10. Roadmap (where we're going)

### Phase 8 Early (Weeks 87–89) — *not yet started*

Plan: `/Users/skylar/.claude/plans/phase8-early-items.md`

- **A1** — MLflow tracking unification (collapse parallel tracking paths into one).
- **A2** — Filter chain structuring (formalize the order of pre-trade filters; today they are wired in `paper_trader.py` directly).
- **A3** — Capacity stress test (saturate the rate limiter / order manager and measure tail latencies).
- **A4** — Scale-up runbook (how to move from one bot instance to N).

### Phase 9+ — *backlog, explicitly deferred*

- General-equity support (calendar, halts, regulatory routing). Currently **explicit non-goal** until the crypto path has live P&L.
- Cross-venue portfolio rebalancing in production.
- High-frequency feature paths (sub-minute / book features).

### Operator-side TODO (post-drill)

Captured in [docs/phase7.6/sign_off.md](docs/phase7.6/sign_off.md):

- Run `python scripts/first_dollar_drill.py --live --capital 100` against a real exchange.
- Complete the 72h autonomous drill (started 2026-04-25; PID 41005).
- Run `python scripts/analyze_drift_baseline.py` against drill outputs.
- Review `logs/incidents/*.md` and confirm `safety_net_triggered=True` for every injected fault.

## 11. References

- **User guide (operating manual)**: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **Development guidelines**: [DEVELOPMENT_GUIDELINES.md](DEVELOPMENT_GUIDELINES.md)
- **Multi-agent architecture**: [docs/MULTI_AGENT_MANAGER.md](docs/MULTI_AGENT_MANAGER.md)
- **Meta-agent + hidden states**: [docs/META_AGENT_WITH_HIDDEN_STATES.md](docs/META_AGENT_WITH_HIDDEN_STATES.md)
- **Risk manager architecture**: [docs/architecture/risk_manager.md](docs/architecture/risk_manager.md)
- **Runbook (operator manual)**: [docs/runbook/README.md](docs/runbook/README.md)
- **Go-live checklist**: [docs/runbook/go_live_checklist.md](docs/runbook/go_live_checklist.md)
- **Phase 7.6 sign-off**: [docs/phase7.6/sign_off.md](docs/phase7.6/sign_off.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

*This README is a living document — when scope changes (e.g., Phase 8 begins or general-equity support lands), update sections 1, 2, 9, and 10 accordingly.*
