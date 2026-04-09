# Trading Bot Runbook

**Phase 6 Week 59 — Ops Readiness**

This runbook covers operational procedures for the trading bot in paper-trading
and live-trading modes.  Every section links to a dedicated failure-mode
document that walks through diagnosis and recovery.

---

## Quick-start decision tree

```
Bot is not working correctly
│
├─ No new prices arriving? → failures/data_feed_stale.md
├─ Exchange API returning errors? → failures/exchange_api_error.md
├─ Bot stopped trading / logged "SHUTDOWN"? → failures/drawdown_kill_switch.md
├─ Process crashed / restarting? → failures/crash_recovery.md
└─ Model outputting NaN actions? → failures/model_nan_output.md
```

---

## Failure-mode runbooks

| File | Scenario |
|------|----------|
| [data_feed_stale.md](failures/data_feed_stale.md) | Live price feed has gone silent or stale |
| [exchange_api_error.md](failures/exchange_api_error.md) | CCXT / exchange API returns errors |
| [drawdown_kill_switch.md](failures/drawdown_kill_switch.md) | Kill-switch fired due to drawdown breach |
| [crash_recovery.md](failures/crash_recovery.md) | Process crash — restore from SQLite checkpoint |
| [model_nan_output.md](failures/model_nan_output.md) | RL model produces NaN/inf actions |

---

## On-call checklist

See [oncall_checklist.md](oncall_checklist.md) for daily and weekly procedures.

---

## Key log locations

| Log | Path |
|-----|------|
| PaperTrader stdout | `logs/paper_trader.log` |
| Audit chain | `audit_log/audit.jsonl` |
| SQLite checkpoint | `state/paper_trader.db` |
| MLflow experiment | `mlruns/` |

---

## Useful one-liners

```bash
# Verify audit log chain integrity
python scripts/verify_audit_log.py audit_log/audit.jsonl

# Run all disaster-recovery drills
python scripts/drills/run_drill.py --scenario all

# Check SQLite checkpoint (last snapshot timestamp)
sqlite3 state/paper_trader.db "SELECT updated_at, cash, equity FROM account_state WHERE id=1;"

# Tail live logs
tail -f logs/paper_trader.log | grep -E "SHUTDOWN|ERROR|WARNING|restored"
```
