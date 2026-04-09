# Failure: Process Crash — Crash Recovery

**Scenario**: The bot process was killed (OOM, SIGKILL, power loss, hardware
fault) mid-episode.  Position state, cash, and step count must be restored from
the SQLite checkpoint before resuming.

---

## Symptoms

- Process not running: `ps aux | grep paper_trader` returns nothing.
- `state/paper_trader.db` exists and contains a recent snapshot.
- Last audit log entry `ts` is close to crash time.
- Exchange may still have an open position.

## Log locations

| What | Where |
|------|-------|
| System journal | `journalctl -u trading-bot --since "1 hour ago"` |
| Last audit entry | `tail -1 audit_log/audit.jsonl \| python -m json.tool` |
| Checkpoint age | `sqlite3 state/paper_trader.db "SELECT updated_at FROM account_state;"` |

---

## Diagnosis steps

1. **Verify the checkpoint is fresh enough to trust**

   ```bash
   sqlite3 state/paper_trader.db \
     "SELECT id, cash, equity, updated_at FROM account_state;"
   ```

   The `updated_at` should be within seconds of the crash.  If it is hours old,
   the persistence layer was not checkpointing — investigate `_checkpoint_every_n_steps`.

2. **Compare checkpoint position with exchange position**

   ```bash
   python - <<'EOF'
   import json
   from deployment.persistence.state_store import StateStore
   snap = StateStore("state/paper_trader.db").load_latest()
   print("checkpoint position:", snap["position"], "entry_price:", snap["entry_price"])
   EOF
   ```

   Then compare to live exchange balance.  If they diverge, the crash happened
   after an order fill but before the checkpoint was written.

3. **Inspect the audit log for the last recorded fill**

   ```bash
   python - <<'EOF'
   import json
   records = [json.loads(l) for l in open("audit_log/audit.jsonl") if l.strip()]
   fills = [r for r in records if r["type"] == "fill"]
   if fills: print(json.dumps(fills[-1], indent=2))
   EOF
   ```

---

## Recovery steps

1. **Restore from checkpoint** (normal case — checkpoint is fresh):

   ```bash
   python -m deployment.paper_trader \
     --config config/paper_trading.yaml \
     --restore \
     --duration 3600
   ```

   Internally this calls `PaperTrader.restore(state_store, ...)`, which:
   - Loads `state/paper_trader.db` snapshot.
   - Rebuilds `TradingState` (position, cash, step, portfolio history).
   - Logs `"restored: resuming PaperTrader at step=N"` on first price tick.

2. **Checkpoint diverges from exchange** (crash after fill, before checkpoint):
   - Manually reconcile: adjust `state/paper_trader.db` to match actual exchange
     position OR clear the checkpoint and let the bot start fresh.
   - To clear: `python -c "from deployment.persistence.state_store import StateStore; StateStore('state/paper_trader.db').clear()"`

3. **Checkpoint is corrupted** (SQLite integrity error):

   ```bash
   sqlite3 state/paper_trader.db "PRAGMA integrity_check;"
   ```

   If broken: delete the file and restart fresh (position loss of in-flight episode).

---

## Post-incident checklist

- [ ] First log line after restart contains `"restored: resuming PaperTrader at step=N"`.
- [ ] Position and cash in logs match the pre-crash values.
- [ ] Audit log chain intact (no hash break across the gap):
  `python scripts/verify_audit_log.py audit_log/audit.jsonl`
- [ ] Root-cause of crash identified (OOM → increase swap; segfault → update deps).
