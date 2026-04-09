# On-Call Checklist

**Phase 6 Week 59 (S19)**

Checklists for the person on-call for the trading bot.  Run these checks in
order; each item is a specific command or observation.

---

## Before starting a trading session

Run these **every time** before enabling live or paper trading.

### Environment

- [ ] Python environment is correct:
  ```bash
  which python  # must be /Users/skylar/anaconda3/bin/python
  python --version  # must be 3.10.x
  ```

- [ ] No uncommitted secrets:
  ```bash
  git status --short | grep -E "secrets|\.env|\.db"
  # must return empty
  ```

- [ ] Config secret refs resolve (no plain keys):
  ```bash
  grep -r "api_key\s*[:=]\s*['\"]" config/ && echo "FAIL: plain key found" || echo "OK"
  ```

- [ ] `.gitignore` covers sensitive paths:
  ```bash
  git check-ignore -v secrets.json state/ audit_log/ "*.db"
  # each line must show it is ignored
  ```

### State

- [ ] SQLite checkpoint is either fresh (< 1 h old) or absent (fresh start):
  ```bash
  sqlite3 state/paper_trader.db \
    "SELECT updated_at, cash, equity FROM account_state WHERE id=1;" 2>/dev/null \
    || echo "No checkpoint — fresh start"
  ```

- [ ] If checkpoint exists, verify it is not corrupted:
  ```bash
  sqlite3 state/paper_trader.db "PRAGMA integrity_check;"
  # must print: ok
  ```

### Audit log

- [ ] If audit log exists, verify chain:
  ```bash
  python scripts/verify_audit_log.py audit_log/audit.jsonl
  # must exit 0
  ```

### Risk config

- [ ] `max_drawdown_threshold` in config matches intended risk appetite:
  ```bash
  grep max_drawdown config/paper_trading.yaml
  ```
  Expected value: `0.10` – `0.20` for paper trading.

- [ ] `max_position_size` is set correctly (≤ 1.0 for single-asset):
  ```bash
  grep max_position_size config/paper_trading.yaml
  ```

---

## Daily morning checks (before market open)

- [ ] Bot process is running:
  ```bash
  ps aux | grep paper_trader | grep -v grep
  ```

- [ ] Logs show recent activity (< 10 min ago):
  ```bash
  tail -5 logs/paper_trader.log
  ```

- [ ] No ERROR lines in the last 12 h:
  ```bash
  grep -c "ERROR" logs/paper_trader.log
  # if count > 0, review: grep "ERROR" logs/paper_trader.log | tail -20
  ```

- [ ] Audit log chain still intact:
  ```bash
  python scripts/verify_audit_log.py audit_log/audit.jsonl
  ```

- [ ] Checkpoint timestamp is recent (< 5 min if step-every-1):
  ```bash
  sqlite3 state/paper_trader.db \
    "SELECT updated_at FROM account_state WHERE id=1;"
  ```

- [ ] No drawdown shutdown overnight:
  ```bash
  grep "SHUTDOWN" logs/paper_trader.log | tail -5
  # must be empty; if not → follow drawdown_kill_switch.md
  ```

- [ ] Portfolio value is within expected range:
  ```bash
  sqlite3 state/paper_trader.db \
    "SELECT cash, equity FROM account_state WHERE id=1;"
  ```

---

## Weekend checks (Friday evening / Monday morning)

- [ ] All daily checks above.

- [ ] Nightly drill job passed in GitHub Actions:
  - Go to Actions → "Disaster-Recovery Drills (nightly)"
  - Confirm last run is green.

- [ ] Disk usage for logs and state is not growing unboundedly:
  ```bash
  du -sh audit_log/ state/ logs/
  ```
  If `audit_log/audit.jsonl` > 500 MB, consider rotating.

- [ ] Reconciliation report is clean:
  ```bash
  python scripts/reconcile.py 2>&1 | tail -20
  ```

- [ ] Model checkpoint date is recent (re-train if > 30 days):
  ```bash
  ls -lh checkpoints/ | tail -5
  ```

---

## Emergency contacts / escalation

| Situation | Action |
|-----------|--------|
| Exchange API down > 1 h | Follow `failures/exchange_api_error.md`, set `simulation_mode: true` |
| Kill-switch fired | Follow `failures/drawdown_kill_switch.md`, do NOT restart without review |
| Process OOM | Increase swap, restart with `--restore`, investigate memory leak |
| Audit chain broken | Preserve log file, open incident report, do NOT continue trading |
| NaN model output | Follow `failures/model_nan_output.md`, consider retraining |

---

## Quick-reference commands

```bash
# Check if bot is alive
ps aux | grep paper_trader

# View last 50 log lines
tail -50 logs/paper_trader.log

# Verify audit chain
python scripts/verify_audit_log.py audit_log/audit.jsonl

# Run all drills
python scripts/drills/run_drill.py --scenario all

# Inspect checkpoint
sqlite3 state/paper_trader.db ".tables"
sqlite3 state/paper_trader.db "SELECT * FROM account_state;"
sqlite3 state/paper_trader.db "SELECT * FROM positions;"

# Emergency: clear checkpoint for fresh start
python -c "
from deployment.persistence.state_store import StateStore
StateStore('state/paper_trader.db').clear()
print('Checkpoint cleared.')
"
```
