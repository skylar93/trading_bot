# Week 72: CCXT Sandbox Wire

**Track**: F — Real Connectivity (F1–F6)  
**Commit**: 98bd5cb  
**Tests**: 72 new, 2164 total passed

---

## Files Added / Modified

| File | Change | Purpose |
|------|--------|---------|
| `deployment/exchange/ccxt_adapter.py` | New (417 lines) | WebSocket-first CCXT connectivity |
| `deployment/execution/order_manager.py` | Modified | exchange_mode field, sandbox wiring |
| `config/schema.py` | Modified | ExecutionConfig: exchange_mode enum |
| `config/deployment.yaml` | Modified | Sandbox section + testnet credential_ref |
| `config/trading.yaml` | Modified | exchange_mode, timeframe, heartbeat_timeout |
| `deployment/audit/audit_logger.py` | Modified | Credential redaction (_REDACT_KEYS) |
| `scripts/sandbox_smoke.py` | New (172 lines) | Manual sandbox smoke (local_only) |

---

## Design Decisions

**WebSocket-first with REST fallback (F1)**  
CCXTAdapter subscribes to `ticker`, `orderbook`, and `OHLCV` channels via ccxt.pro. If ccxt.pro is unavailable, falls back to REST polling via lazy import check `_check_ccxt_pro()`. Thread-safe caches (`_latest_ticker`, `_latest_orderbook`, `_latest_ohlcv`) protected by RLock.

**exchange_mode field (F3)**  
New config field: `exchange_mode: "paper" | "sandbox" | "live"`. Replaces legacy `paper_mode` boolean (backward compatible). In `sandbox` mode, `exchange.set_sandbox_mode(True)` is called on the CCXT exchange object.

**Exponential-backoff reconnect + heartbeat watchdog (F4)**  
Max 5 retries, base=1.0s, cap=30.0s. Watchdog thread `_heartbeat_watchdog` checks every 5s; silence beyond `heartbeat_timeout` (default 60s) calls `alerter.check_connection_lost()` and `on_halt()` callback.

**Credential redaction in audit log (F6)**  
`AuditLogger` uses `_REDACT_KEYS` set to recursively mask sensitive fields before hashing. All API keys appear as `***REDACTED***` in audit JSONL.

---

## Test Coverage

- CCXTAdapter unit (connect, subscribe, fallback, reconnect, callback dispatch)
- Sandbox mode activation
- Heartbeat watchdog timeout path
- Credential redaction (api_key, api_secret, token, password, passphrase)
- `@local_only` marker: sandbox smoke excluded from CI
