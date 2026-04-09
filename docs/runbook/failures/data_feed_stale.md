# Failure: Data Feed Stale

**Scenario**: The live price feed has stopped delivering new bars.  The bot is
running but no new data arrives.

---

## Symptoms

- Log lines stop showing `"step N"` progress.
- Last price timestamp is older than `max_staleness_sec` (default 300 s).
- Alerter may emit a `DATA_FEED_STALE` alert (if wired).
- Dashboard price chart flatlines.

## Log locations

| What | Where |
|------|-------|
| PaperTrader stdout | `logs/paper_trader.log` |
| Audit log | `audit_log/audit.jsonl` (look for last `model_decision` entry) |

---

## Diagnosis steps

1. **Check the feed process**

   ```bash
   ps aux | grep -E "paper_trader|ccxt|data"
   ```

2. **Check network connectivity to exchange**

   ```bash
   curl -s https://api.binance.com/api/v3/time | python -m json.tool
   ```

3. **Check last bar in audit log**

   ```bash
   tail -5 audit_log/audit.jsonl | python -m json.tool
   ```

4. **Check process resource usage** — memory leak or CPU peg can starve feed

   ```bash
   top -p $(pgrep -f paper_trader)
   ```

---

## Recovery steps

1. **If feed process has died**: restart with the existing SQLite checkpoint

   ```bash
   python -m deployment.paper_trader \
     --config config/paper_trading.yaml \
     --restore --duration 3600
   ```

   The `--restore` flag calls `PaperTrader.restore()` which reloads positions
   and cash from `state/paper_trader.db`.

2. **If exchange is unreachable**: wait for exchange maintenance window to
   finish.  Check the exchange status page.  Do NOT restart the bot multiple
   times — it will attempt recovery on next startup.

3. **If feed is slow but not dead**: increase `poll_interval_seconds` in
   config to reduce load, then restart.

---

## Post-incident checklist

- [ ] Confirm audit log chain is intact after restart:
  `python scripts/verify_audit_log.py audit_log/audit.jsonl`
- [ ] Confirm SQLite checkpoint `updated_at` reflects pre-crash step.
- [ ] Note exchange downtime in incident log.
- [ ] If staleness was >1 h, review if any positions need manual reconciliation.
