# Failure: Drawdown Kill-Switch Fired

**Scenario**: The bot detected that portfolio value dropped past the configured
`max_drawdown_threshold` (default 20 %).  It liquidated the open position and
halted trading.

---

## Symptoms

- Log line: `"SHUTDOWN triggered: Max drawdown X% >= threshold Y%"` or
  `"SHUTDOWN triggered: RiskManager: max drawdown exceeded"`.
- `state.shutdown_triggered = True` in checkpoint.
- Position reduced to zero (liquidation executed at kill-switch price).
- No further `model_decision` entries in audit log.

## Log locations

| What | Where |
|------|-------|
| PaperTrader stdout | `logs/paper_trader.log` |
| Audit log | `audit_log/audit.jsonl` (look for `risk_event` with `type=drawdown`) |
| Checkpoint | `state/paper_trader.db` |

---

## Diagnosis steps

1. **Confirm the trigger and reason**

   ```bash
   grep "SHUTDOWN" logs/paper_trader.log | tail -5
   ```

2. **Review the portfolio history leading up to shutdown**

   ```bash
   sqlite3 state/paper_trader.db \
     "SELECT cash, equity, updated_at FROM account_state WHERE id=1;"
   ```

3. **Inspect the last N audit records before the risk event**

   ```bash
   python - <<'EOF'
   import json, sys
   records = [json.loads(l) for l in open("audit_log/audit.jsonl") if l.strip()]
   for r in records[-20:]:
       print(r["ts"], r["type"], r.get("payload", {}).get("reason", ""))
   EOF
   ```

4. **Assess whether the drawdown was model error or genuine market move**:
   - Check if the loss aligns with market price action during the same window.
   - If the model was the cause (e.g., repeated wrong-direction trades), consider
     retraining before restarting.

---

## Recovery steps

> **Do not restart the bot immediately after a kill-switch fire without
> reviewing the cause.**  The same model will make the same mistakes.

1. **Verify position is flat** (exchange confirms zero position):

   ```bash
   python - <<'EOF'
   import ccxt, os
   ex = ccxt.binance({"apiKey": os.environ["EXCHANGE_BINANCE_KEY"],
                       "secret": os.environ["EXCHANGE_BINANCE_SECRET"]})
   print(ex.fetch_positions())
   EOF
   ```

2. **Reset checkpoint so the bot can start fresh** (if restarting with same model):

   ```bash
   python - <<'EOF'
   from deployment.persistence.state_store import StateStore
   StateStore("state/paper_trader.db").clear()
   EOF
   ```

3. **Restart bot** (fresh episode, lower `max_drawdown_threshold` if desired):

   ```bash
   python -m deployment.paper_trader --config config/paper_trading.yaml --duration 3600
   ```

4. **If model needs retraining**: run full pipeline on recent data before restarting.

---

## Post-incident checklist

- [ ] Confirm position is zero on exchange.
- [ ] Record total loss and P&L attribution (slippage vs. model decision).
- [ ] Decide: resume same model or retrain?
- [ ] If resuming: adjust `max_drawdown_threshold` in config if threshold was too loose.
- [ ] Verify audit chain after restart.
