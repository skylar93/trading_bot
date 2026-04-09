# Failure: Exchange API Error

**Scenario**: CCXT calls to the exchange return HTTP errors, rate-limit
rejections, or raise exceptions.

---

## Symptoms

- Log lines: `ERROR ccxt.BaseError`, `ExchangeNotAvailable`, `RateLimitExceeded`.
- `OrderManager` logs failed submit retries.
- No fills arriving; `num_trades` stays flat.
- Audit log contains `risk_event` entries with `type=exchange_error`.

## Log locations

| What | Where |
|------|-------|
| PaperTrader stdout | `logs/paper_trader.log` |
| Audit log | `audit_log/audit.jsonl` |

---

## Diagnosis steps

1. **Identify error type from logs**

   ```bash
   grep -E "ccxt|ExchangeError|RateLimit|NetworkError" logs/paper_trader.log | tail -20
   ```

2. **Check if it is a rate-limit issue**

   ```bash
   grep "RateLimit\|429" logs/paper_trader.log | wc -l
   ```

   If count is high and growing: the bot is being throttled.

3. **Test API credentials manually**

   ```bash
   python - <<'EOF'
   import ccxt, os
   ex = ccxt.binance({"apiKey": os.environ["EXCHANGE_BINANCE_KEY"],
                       "secret": os.environ["EXCHANGE_BINANCE_SECRET"]})
   print(ex.fetch_balance())
   EOF
   ```

4. **Check exchange status** — Binance: https://www.binance.com/en/support/announcement
   (do not hard-code URLs in runbook; check exchange status page directly)

---

## Recovery steps

1. **Rate-limit hit**:
   - Increase `rate_limiter.requests_per_second` in config (lower number).
   - Restart bot (it will restore from checkpoint).

2. **Auth error (`AuthenticationError`)**:
   - Rotate API keys via exchange web UI.
   - Update `~/.trading_bot/secrets.json` or environment variables.
   - Restart bot.

3. **Exchange unavailable (maintenance)**:
   - Let `OrderManager` exponential backoff exhaust (max retries in config).
   - Bot will continue running but skip order submission until exchange returns.
   - If bot crashes: restart with `--restore`.

4. **Persistent unknown error**:
   - Set `simulation_mode: true` in config temporarily.
   - Restart bot to paper-trade without live exchange, avoiding further exposure.

---

## Post-incident checklist

- [ ] Verify all open orders reconciled: `python scripts/reconcile.py`.
- [ ] Check audit log chain: `python scripts/verify_audit_log.py audit_log/audit.jsonl`.
- [ ] Review `OrderManager` retry counts in logs — confirm no duplicate orders.
- [ ] Confirm `secrets.json` is gitignored: `git status --short | grep secrets`.
