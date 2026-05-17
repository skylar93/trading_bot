# G2 Paper Trading Kickoff Runbook

**Status**: Ready to execute (2026-05-17)  
**Audience**: Operator (Skylar)  
**Goal**: Start paper trading on the Phase 8-Gamma G2 (realized-PnL reward) model to
accumulate 6–12 months of out-of-sample live evidence before live-deployment decision.

> **Why paper, not live?** Bootstrap CI lower bound < 0 and permutation p = 0.16 on
> the 1M G2 walk-forward evidence. Signal is real (+2.05% mean OOS return, 7/12 folds+),
> but the confidence interval does not yet exclude zero. Paper trading fixes this by
> producing live OOS data that bypasses the evaluation artefacts of the walk-forward
> harness.

> **What this doc does NOT cover**: Live deployment. Live requires completing
> `docs/runbook/go_live_checklist.md` (Tracks E/F/G/S/R/O/SN/Z). Paper trading skips
> that checklist intentionally — it is a data-gathering phase, not a money-at-risk phase.

---

## 0. Scope and time estimate

| Task | Time |
|------|------|
| Prerequisites check | 15 min |
| Checkpoint generation on trading-pc | 60–90 min (GPU training) |
| Config fork + secret setup | 10 min |
| Launch + first-hour sanity | 30 min |
| **Total first-day effort** | **~3 h** |

---

## 1. Prerequisites Checklist

Complete every item before launch. Items marked `[auto]` are script-verified.

### 1.1 Phase 7.5 Safety Nets (11 active)

Verify all 11 safety nets are importable and wired:

```bash
# Quick structural smoke — not a connectivity test
python -c "
from deployment.monitoring.drift_detector import DeploymentDriftDetector
from deployment.paper_trader import PaperTrader
from deployment.secrets.secret_provider import get_default_provider
print('SN imports OK')
"
```

The 11 safety nets and their locations:

| # | Safety Net | Module |
|---|-----------|--------|
| SN1 | Canary auto-demotion (traffic → 0% on -1σ × 6h) | `deployment/paper_trader.py` `_run_canary_agent` |
| SN2 | OTel span instrumentation | `deployment/paper_trader.py` |
| SN3 | Real-time schema drift guard (`on_schema_drift: halt`) | `DeploymentDriftDetector.report_schema_drift` |
| SN4 | Bootstrap reconciliation | `deployment/paper_trader.py` `_reconcile_on_boot` |
| SN5 | Slippage model fit (R² > 0.3) | `config/production/G2_paper.yaml` `apply_slippage: true` |
| SN6 | Fee tier daily sync | env `trading_fee: 0.00018` matches Binance maker |
| SN7 | API key scope probe | `deployment/secrets/secret_provider.py` |
| SN8 | Pre-commit detect-secrets hook | `.git/hooks/pre-commit` |
| SN9 | Capacity baseline snapshot | `docs/phase7/week84_baseline.md` |
| SN10 | Runbook drills ≥ 2 | `docs/runbook/drills/` |
| SN11 | Deployment drift coordinator (shadow 72h) | `deployment/monitoring/drift_detector.py` |

Expected output: `SN imports OK` (any ImportError = blocking, fix before proceeding).

### 1.2 G2 Checkpoint Generation

The 1M walk-forward run on trading-pc **did not persist per-fold checkpoints** — the
`WalkForwardValidator` trains agents in memory for evaluation only. You must generate a
deployment checkpoint via a single full-data training run.

**On trading-pc** (Windows, GTX 1060):

```powershell
# SSH in
ssh trading-pc

# In trading_bot directory — pull the G2 config first
git pull origin main

# Train single agent on full dataset (walk_forward disabled via config flag)
# This writes checkpoints/G2_paper/best_agent.pt and final_agent.pt
python -c "
import pandas as pd, yaml, sys
sys.path.insert(0, '.')
from config.loader import load_raw, _deep_merge
from training.train_pipeline import train_pipeline

cfg = load_raw('config/base.yaml')
with open('config/phase8_gamma/G2_realized_pnl_slippage.yaml') as f:
    override = yaml.safe_load(f)
cfg = _deep_merge(cfg, override)
cfg['walk_forward']['enabled'] = False
cfg['training']['total_timesteps'] = 1_000_000
cfg['paths'] = {'checkpoint_dir': 'checkpoints/G2_paper'}

df = pd.read_csv('data/BTCUSDT_1h.csv', index_col=0, parse_dates=True)
result = train_pipeline(config=cfg, data=df)
print('Done:', result.get('final_model_path'))
" > logs/G2_paper_deploy_train.log 2>&1
```

Poll for completion (typically 60–90 min):

```powershell
ssh trading-pc "tasklist | findstr python"    # empty = done
ssh trading-pc "tail -5 logs/G2_paper_deploy_train.log"
```

**Copy checkpoint to Mac:**

```bash
rsync -avz trading-pc:~/trading_bot/checkpoints/G2_paper/ \
    checkpoints/G2_paper/
```

Verify:

```bash
ls -lh checkpoints/G2_paper/best_agent.pt
# Expect: file exists, size > 1 MB
python -c "
import sys; sys.path.insert(0,'.')
from training.agent_factory import load_agent
a = load_agent('checkpoints/G2_paper/best_agent.pt')
print('checkpoint loads OK, type:', type(a).__name__)
"
```

**Checkpoint selection rationale**: Single full-data training trains on all 17,520 bars
(2024-04 → 2026-04), maximising in-sample coverage. Walk-forward fold 12 checkpoint
(last fold, most recent data) would also be acceptable if you prefer the regime
alignment, but it is not saved by the current harness without modification.

### 1.3 Binance Futures API Key Scope

Paper trading uses the Binance USDⓈ-M Futures testnet (exchange_mode=paper). Create or
confirm a **read + place_order** key scoped to USDⓈ-M futures with **no withdraw
permission**.

Required scopes:
- `FUTURES` read (balance, position, account info)
- `FUTURES` trade (submit/cancel limit orders)
- No `SPOT` trade, no `WITHDRAWAL` — reject at key creation

Store via SecretProvider (env backend, recommended on Mac):

```bash
export EXCHANGE_BINANCE_KEY="your-testnet-api-key"
export EXCHANGE_BINANCE_SECRET="your-testnet-api-secret"
# Persist in shell profile (.zshrc / .bashrc) so they survive reboots
echo 'export EXCHANGE_BINANCE_KEY="..."' >> ~/.zshrc
echo 'export EXCHANGE_BINANCE_SECRET="..."' >> ~/.zshrc
```

Verify SecretProvider can resolve:

```bash
python -c "
from deployment.secrets.secret_provider import get_default_provider
p = get_default_provider()
k = p.get('EXCHANGE_BINANCE_KEY')
print('Key found, length:', len(k))   # do NOT print the key itself
"
```

### 1.4 Data Feed Connectivity

Paper trader polls price from exchange. Verify testnet reachability:

```bash
python -c "
import ccxt
ex = ccxt.binance({'options': {'defaultType': 'future'}, 'urls': {'api': {'public': 'https://testnet.binancefuture.com'}}})
ticker = ex.fetch_ticker('BTC/USDT')
print('BTC/USDT last:', ticker['last'], '  feed OK')
"
```

Expected: price appears within 5 s. Any timeout → check VPN / firewall before proceeding.

---

## 2. Config Fork

The paper trading config lives at [`config/production/G2_paper.yaml`](../../config/production/G2_paper.yaml).
It was forked from `config/phase8_gamma/G2_realized_pnl_slippage.yaml`.

**Key diffs from G2_realized_pnl_slippage.yaml:**

| Key | G2_realized_pnl_slippage.yaml | G2_paper.yaml | Reason |
|-----|-------------------------------|---------------|--------|
| `walk_forward.enabled` | (absent, True in run_wf.py) | `false` | paper trader uses `train_pipeline` not `run_wf.py` |
| `env.initial_balance` | 100000.0 | 10000.0 | conservative paper notional |
| `env.max_position_size` | 1.0 (100%) | 0.10 (10%) | $1000 notional cap on $10k paper capital |
| `paper_trading.max_drawdown_threshold` | (absent) | 0.15 | halt if paper DD > 15% |
| `paper_trading.enabled` | (absent) | true | activates deployment runtime |
| `paths.checkpoint` | (absent) | `checkpoints/G2_paper/best_agent.pt` | deployment checkpoint |
| `paths.checkpoint_dir` | (absent) | `checkpoints/G2_paper` | per-run isolation |
| `persistence.db_path` | (absent) | `state/G2_paper_trader.db` | state isolation |
| `drift.shadow_mode_hours` | (absent) | 72 | SN11: first 72h log-only |
| `apply_slippage` | true | true (unchanged) | **must stay true** — matches 1M eval |
| `env.trading_fee` | 0.00018 | 0.00018 (unchanged) | **must stay** — futures maker with BNB |

**Do not change `apply_slippage: true` or `trading_fee: 0.00018`** — changing either
invalidates the comparison between paper results and the G2 1M OOS baseline (+2.05%).

Update the checkpoint path if you chose a different filename:

```bash
# Confirm path matches your generated checkpoint
grep "checkpoint:" config/production/G2_paper.yaml
# Should print:   checkpoint: "checkpoints/G2_paper/best_agent.pt"
```

---

## 3. Launch

### 3.1 Pre-launch sanity

```bash
# Run training-pipeline smoke tests (required per project convention)
python scripts/smoke_train.py
python scripts/smoke_walkforward.py
# Both must exit 0 before proceeding
```

### 3.2 tmux session (recommended for 24/7)

Paper trading must run continuously. Use tmux so it survives terminal disconnects.

```bash
# Create dedicated session
tmux new-session -d -s g2_paper

# Launch paper trader inside it
tmux send-keys -t g2_paper \
  "python scripts/run_paper_trader.py \
    --config config/production/G2_paper.yaml \
    --exchange-mode paper \
    --log-dir logs/G2_paper/ \
    --pid-file state/G2_paper.pid" \
  Enter

# Attach to watch startup output
tmux attach -t g2_paper
# Detach with Ctrl-B D when satisfied
```

### 3.3 Startup verification (first 5 minutes)

Watch for these log lines in `logs/G2_paper/paper_trader_<ts>.log`:

```
INFO  run_paper_trader  : PID <N> written to state/G2_paper.pid
INFO  paper_trader      : Warmup mode active — size_fraction=0.50, 30 min
INFO  paper_trader      : DeploymentDriftDetector shadow mode (72.0 h remaining)
INFO  paper_trader      : Step 1 — price=<BTC_PRICE>, action=<-1..1>, hold/buy/sell
```

If you see `ERROR` or `CRITICAL` within the first 60 s → abort (see §5).

### 3.4 PID file and log locations

| File | Purpose |
|------|---------|
| `state/G2_paper.pid` | Process ID; removed on clean shutdown |
| `state/G2_paper_trader.db` | StateStore checkpoint (SQLite) |
| `logs/G2_paper/paper_trader_<ts>.log` | Structured run log |
| `logs/G2_paper/paper_trader_<ts>.json` | Final report (written on shutdown) |
| `audit_log/G2_paper_audit.jsonl` | Fill/cost audit trail (A6) |

### 3.5 nohup fallback (if tmux is unavailable)

```bash
nohup python scripts/run_paper_trader.py \
    --config config/production/G2_paper.yaml \
    --exchange-mode paper \
    --log-dir logs/G2_paper/ \
    --pid-file state/G2_paper.pid \
  > logs/G2_paper/nohup.out 2>&1 &
echo "PID: $!"
```

---

## 4. Monitoring

### 4.1 Daily checks (< 5 min/day)

```bash
# P&L snapshot from latest daily report
python -c "
import json, glob, os
reports = sorted(glob.glob('logs/G2_paper/paper_trader_*.json'))
if reports:
    r = json.load(open(reports[-1]))
    print(f'Return: {r[\"total_return\"]:.4%}')
    print(f'Trades: {r[\"num_trades\"]}')
    print(f'MaxDD:  {r[\"max_drawdown\"]:.4%}')
    print(f'Sharpe: {r[\"sharpe_ratio\"]:.3f}')
else:
    print('No report yet — trader still running')
"

# Process alive?
cat state/G2_paper.pid && ps -p $(cat state/G2_paper.pid) | grep python \
  && echo "RUNNING" || echo "STOPPED — check logs"

# Recent log tail (any ERRORs?)
tail -50 logs/G2_paper/paper_trader_*.log | grep -E "ERROR|CRITICAL|halt"
```

### 4.2 Weekly review (30 min/week)

Compute realized return for the week vs G2 1M OOS baseline:

```bash
# Full report from running instance (interrupts nothing)
python -c "
import json, glob
reports = sorted(glob.glob('logs/G2_paper/paper_trader_*.json'))
# Aggregate across all report files if multiple (restarts)
total_return = sum(json.load(open(r))['total_return'] for r in reports)
weeks_elapsed = len(reports)  # rough proxy; one report per run segment
print(f'Cumulative paper return: {total_return:.4%}')
print(f'G2 1M OOS baseline (mean episode): +2.05%')
print(f'G2 1M bear-fold baseline: +4.53%')
"
```

**Baseline interpretation**: The G2 1M "+2.05% all" is the mean return per OOS
evaluation episode (random-start, 20 episodes over ~2-month OOS windows). Mapping to
real-time: treat it as a directional benchmark over a comparable price-regime window,
not a strict monthly rate — the episode length in evaluation (~days per episode) differs
from continuous paper trading.

Weekly: compare sign and rough magnitude. Expect noise at weekly granularity; the signal
window is 3+ months.

### 4.3 Automatic halt triggers (hard stops)

These are enforced by the paper trader at runtime — no manual action needed:

| Trigger | Threshold | Source |
|---------|-----------|--------|
| Paper MaxDD | > 15% | `paper_trading.max_drawdown_threshold` |
| Reward drift (Z-score) | > 2.0σ (after 72h shadow) | SN11 `drift.reward_return_sigma_threshold` |
| Feature PSI drift | > 0.2 (after 72h shadow) | SN11 `drift.feature_psi_threshold` |
| Schema drift on price feed | any | SN3 `report_schema_drift` |
| NaN / Inf in price | any | `data_pipeline_safety` |

After any automatic halt: check `logs/G2_paper/paper_trader_*.log` for the halt reason
before restarting.

### 4.4 Alert delivery

Alerts emit via `TradingAlerter`. With the default config (`monitoring.alert_on_halt:
true`), CRITICAL-level halts are written to the log file and any alerter configured at
launch. To add Slack/email alerting, wire a `TradingAlerter` instance in a wrapper
script — the runbook does not require this for paper trading, but it is recommended for
24/7 unattended operation.

---

## 5. Abort Criteria (operator must halt immediately)

Stop the paper trader and investigate before restarting if any of the following occur:

| Condition | Action |
|-----------|--------|
| Paper MaxDD > 15% (unrecovered) | `kill $(cat state/G2_paper.pid)` → investigate |
| 3 consecutive weeks with realized return < −2% | Halt, reassess model; see §6 |
| Price feed gap > 1 hour (SN3 schema drift or stale data) | Halt; restart only after feed restored |
| Drift detector switches from shadow → active with CRITICAL | Review drift details in log; operator decides restart vs abort |
| Operator manual abort (any reason) | `kill -SIGTERM $(cat state/G2_paper.pid)` (graceful) |
| API key scope error or unauthorized order | Halt; rotate keys; verify scope (§1.3) |

**Graceful shutdown:**

```bash
kill -SIGTERM $(cat state/G2_paper.pid)
# Wait up to 30 s; trader cancels open orders, liquidates, writes final report
```

**Force shutdown (last resort):**

```bash
kill -SIGKILL $(cat state/G2_paper.pid)
# WARNING: position may remain open — verify on Binance testnet manually
```

---

## 6. Success Criteria (gate to live deployment)

All six must be met before opening the live-deployment decision:

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| **Minimum duration** | ≥ 6 months continuous paper trading | Clock starts from first clean launch |
| **Realized Sharpe** | > 0.5 over the 6-month period | Computed from daily P&L series |
| **No catastrophic drawdown** | MaxDD < 15% at any point | Matches halt threshold |
| **Trade frequency** | ≥ 1.0 trades/episode average | G2 1M baseline: 2.22/ep; collapse → < 1.0 = model degraded |
| **Bootstrap CI** | Monthly return CI lower bound > 0 | Recalculate after ≥ 6 monthly observations |
| **Operator + review** | Both Skylar sign-off and objective third-party review | No code-only gate |

**Bootstrap CI recalculation** (after ≥ 6 months):

```bash
python -c "
import json, glob, numpy as np
# Collect monthly returns from daily report files or audit log
# Placeholder — replace with actual monthly series extraction
monthly_returns = [0.01, 0.02, -0.005, 0.015, 0.008, 0.012]  # example
n_bootstrap = 10000
means = [np.mean(np.random.choice(monthly_returns, len(monthly_returns), replace=True))
         for _ in range(n_bootstrap)]
ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
print(f'Bootstrap 95% CI: [{ci_lo:.4%}, {ci_hi:.4%}]')
print('GO' if ci_lo > 0 else 'NOT YET')
"
```

---

## 7. Reproducibility and Rollback

### 7.1 Version pinning

Before launch, record the exact git state:

```bash
git log --oneline -1
# Copy this commit hash into your ops log
```

Key config and checkpoint hashes:

```bash
sha256sum config/production/G2_paper.yaml checkpoints/G2_paper/best_agent.pt \
  | tee docs/reports/G2_paper/launch_manifest.txt
```

Store `launch_manifest.txt` in `docs/reports/G2_paper/` (not gitignored) so any future
comparison can verify bit-for-bit reproduction.

### 7.2 Returning to "no paper trading" state

The paper trader is fully opt-in. Stopping it has zero effect on the rest of the system:

```bash
# Graceful stop
kill -SIGTERM $(cat state/G2_paper.pid)

# Confirm stopped
sleep 5 && [ -f state/G2_paper.pid ] && echo "STILL RUNNING" || echo "Stopped clean"
```

No code changes needed. The config in `config/production/G2_paper.yaml` and the StateStore
at `state/G2_paper_trader.db` persist but are inert while the process is stopped.

### 7.3 Resuming after a crash or restart

The paper trader persists state to `state/G2_paper_trader.db` every 100 steps
(`persistence.checkpoint_every_n_steps: 100`). On restart, it calls `PaperTrader.restore()`
automatically if the db file exists — no manual state recovery needed.

```bash
# Restart with the same command — restore is automatic
tmux send-keys -t g2_paper \
  "python scripts/run_paper_trader.py \
    --config config/production/G2_paper.yaml \
    --exchange-mode paper \
    --log-dir logs/G2_paper/ \
    --pid-file state/G2_paper.pid" \
  Enter
```

### 7.4 State backup schedule

Back up the StateStore weekly (add to cron or run manually):

```bash
cp state/G2_paper_trader.db \
   state/G2_paper_trader_$(date +%Y%m%d).db.bak
# Keep 4 rolling weekly backups; prune older ones:
ls -t state/G2_paper_trader_*.db.bak | tail -n +5 | xargs rm -f
```

---

## Quick Reference

```bash
# Launch
python scripts/run_paper_trader.py \
  --config config/production/G2_paper.yaml \
  --exchange-mode paper \
  --log-dir logs/G2_paper/ \
  --pid-file state/G2_paper.pid

# Status check
cat state/G2_paper.pid && ps -p $(cat state/G2_paper.pid) | grep python

# Graceful stop
kill -SIGTERM $(cat state/G2_paper.pid)

# Attach to live session
tmux attach -t g2_paper

# Log tail
tail -f logs/G2_paper/paper_trader_*.log
```

---

*Runbook version: 2026-05-17. Next review: after first monthly P&L report.*  
*Related docs: [`go_live_checklist.md`](go_live_checklist.md), [`phase8_gamma_winner_handoff`](../../memory/phase8_gamma_winner_handoff.md)*
