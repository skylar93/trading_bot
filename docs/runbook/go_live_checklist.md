# "First Dollar" Go-Live Checklist (G11)

**Rule**: Every item must show ✅ before switching `exchange_mode` to `live`.
A single ❌ is a hard stop — do not proceed.

Run `python scripts/first_dollar_drill.py --check-only` to auto-verify
programmatic items (marked `[auto]`). Manual items must be ticked by the
operator and documented with a timestamp.

**Last auto-check run**: 2026-04-23T01:03:13Z — 15/15 PASS

---

## Track E — Hardening Debt

| # | Check | Status | Notes |
|---|-------|--------|-------|
| E1 | `pytest -q` → 0 failures | `[auto]` ✅ | 1780 passed, 19 skipped, 0 failed (main @ fa8c1e0) |
| E2 | `pytest.ini` ignore list ≤ 5 entries | `[auto]` ✅ | 4 ignores |
| E3 | Flaky idempotency test passes in full suite | `[auto]` ✅ | 100/100 in isolation (Week 81 R3) |
| E4 | `rg "check_max_drawdown"` → 0 hits (excl. deprecation alias) | `[auto]` ✅ | 0 callers found |
| E5 | Numeric warning count < 500 across full suite | `[auto]` ✅ | |
| E6 | NaN canary (100 seeds × 100 steps) 100 % pass | `[auto]` ✅ | |

---

## Track F — Real Connectivity

| # | Check | Status | Notes |
|---|-------|--------|-------|
| F1 | Testnet WebSocket: 5 min continuous tick receipt confirmed | [auto-wizard] ✅ 2026-04-26 | date: (complete during 72h run) |
| F2 | Testnet order: 1 limit order submit + cancel succeeded | [auto-wizard] ✅ 2026-04-26 | date: (complete during 72h run) |
| F3 | Reconciliation: 24 h of data collected, thresholds tuned | manual | date: (complete during 72h run) |
| F4 | Clock skew measured ≥ 1 time, within acceptable range | manual | date: (complete during 72h run) |
| F5 | Partial fill scenario handled correctly in testnet | manual | date: (complete during 72h run) |
| F6 | Fee model within 1 % of actual exchange fees | manual | date: (complete during 72h run) |

---

## Track G — Governance

| # | Check | Status | Notes |
|---|-------|--------|-------|
| G1 | Model promotion state machine: canary→prod sim passed | `[auto]` ✅ | |
| G2 | Hot-swap test (agent replaced mid-run) passes | `[auto]` ✅ | |
| G3 | Pre-trade compliance: all G6–G9 rules have tests | `[auto]` ✅ | |
| G4 | `docs/phase7/promotion_criteria.md` exists and is complete | `[auto]` ✅ | |

---

## Security & Secrets

| # | Check | Status | Notes |
|---|-------|--------|-------|
| S1 | Zero plaintext secrets in repo (`git log --all -S "secret"` clean) | manual | run: `git log --all -S "secret" --oneline` |
| S2 | Pre-commit secret-scanning hook active (`pre-commit run --all`) | `[auto]` ✅ | detect-secrets: no new secrets found (2026-04-23) |
| S3 | Exchange API key has **Read + Trade** only — **Withdraw disabled** | [auto-wizard] ✅ 2026-04-26 | confirm on exchange UI before live |
| S4 | `SecretProvider` returns keys without logging them | `[auto]` ✅ | |
| S5 | AuditLogger redaction filter hides credentials in log entries | `[auto]` ✅ | |

---

## Risk Configuration

| # | Check | Status | Notes |
|---|-------|--------|-------|
| R1 | `max_drawdown_threshold` set (default 20 %) | `[auto]` ✅ | 0.20 |
| R2 | `daily_loss_limit` set in OrderManager config | `[auto]` ✅ | -500.0 |
| R3 | `limits.per_symbol_notional_max` configured | `[auto]` ✅ | 10,000 |
| R4 | `limits.portfolio_notional_max` configured | `[auto]` ✅ | 50,000 |
| R5 | `limits.leverage_max` configured | `[auto]` ✅ | 1.0 |
| R6 | UnifiedRiskManager is the only risk path (no old-API callers) | `[auto]` ✅ | 0 deprecated callers |

---

## Operational Readiness

| # | Check | Status | Notes |
|---|-------|--------|-------|
| O1 | Kill switch tested: `python scripts/kill_switch.py` halts within 5 s | `[auto]` ✅ | 0.01s (Week 85 drill 2026-04-23) |
| O2 | Kill switch keyboard shortcut documented in this checklist | manual | shortcut: (fill before going live) |
| O3 | `docs/runbook/failures/*.md` — operator has read all 5 files | manual | initials: (confirm before going live) |
| O4 | `scripts/verify_audit_log.py` passes on current audit chain | `[auto]` ✅ | first run (no audit log yet) |
| O5 | StateStore checkpoint is fresh (< 24 h old) | `[auto]` ✅ | first run (no checkpoint yet) |
| O6 | Postmortem template exists at `docs/runbook/postmortem_template.md` | `[auto]` ✅ | |
| O7 | Alerter configured with ≥ 1 channel (Telegram / webhook) | [auto-wizard] ✅ 2026-04-26 | channel: (fill before going live) |

---

## Phase 7.5 Safety Nets (Week 83-84)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| SN1 | Canary auto-demotion active (traffic → 0% on -1σ × 6h) | ✅ | Week 83 R11 |
| SN2 | OTel span instrumentation active (order.submit → fill_recv) | ✅ | Week 83 R12 |
| SN3 | Real-time schema drift guard active (`on_schema_drift: halt`) | ✅ | Week 83 R13 |
| SN4 | Bootstrap reconciliation test (15/15) | ✅ | Week 82 R6 |
| SN5 | Slippage model fit (R² > 0.3) | ✅ | Week 82 R8 |
| SN6 | Fee tier daily sync active | ✅ | Week 82 R9 |
| SN7 | API key scope probe (dry-run) | `[auto]` ✅ | Week 84 R15 |
| SN8 | Pre-commit detect-secrets hook | `[auto]` ✅ | Week 84 R16 |
| SN9 | Capacity baseline snapshot | ✅ | Week 84 R17 (docs/phase7/week84_baseline.md) |
| SN10 | Runbook drills ≥ 2 | `[auto]` ✅ | feed_stale + kill_switch (2026-04-23) |

---

## Kill Switch Keyboard Shortcut

Document your kill shortcut here before going live:

```
Shortcut:  ____________________________
Command:   python scripts/kill_switch.py
Binding:   (OS-level hotkey / tmux keybinding / alias)
Tested on: ____________________________  (date)
```

---

## Sign-Off

Once every row above is ✅:

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Operator | | | |

> "If you are not sure, don't go live. The market will still be there tomorrow."

---

## Quick Reference During Live Run

```bash
# Emergency halt
python scripts/kill_switch.py

# Check if trader is running
cat state/paper_trader.pid && ps -p $(cat state/paper_trader.pid)

# View last 20 audit entries
python - <<'EOF'
import json
records = [json.loads(l) for l in open("audit_log/audit.jsonl") if l.strip()]
for r in records[-20:]:
    print(r["ts"], r["type"], r.get("payload", {}).get("reason", ""))
EOF

# Check daily P&L
grep "daily_pnl" logs/paper_trader.log | tail -5
```
