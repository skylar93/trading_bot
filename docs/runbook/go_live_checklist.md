# "First Dollar" Go-Live Checklist (G11)

**Rule**: Every item must show ✅ before switching `exchange_mode` to `live`.
A single ❌ is a hard stop — do not proceed.

Run `python scripts/first_dollar_drill.py --check-only` to auto-verify
programmatic items (marked `[auto]`). Manual items must be ticked by the
operator and documented with a timestamp.

---

## Track E — Hardening Debt

| # | Check | Status | Notes |
|---|-------|--------|-------|
| E1 | `pytest -q` → 0 failures | `[auto]` | |
| E2 | `pytest.ini` ignore list ≤ 5 entries | `[auto]` | |
| E3 | Flaky idempotency test passes in full suite | `[auto]` | |
| E4 | `rg "check_max_drawdown"` → 0 hits (excl. deprecation alias) | `[auto]` | |
| E5 | Numeric warning count < 500 across full suite | `[auto]` | |
| E6 | NaN canary (100 seeds × 100 steps) 100 % pass | `[auto]` | |

---

## Track F — Real Connectivity

| # | Check | Status | Notes |
|---|-------|--------|-------|
| F1 | Testnet WebSocket: 5 min continuous tick receipt confirmed | manual | date: |
| F2 | Testnet order: 1 limit order submit + cancel succeeded | manual | date: |
| F3 | Reconciliation: 24 h of data collected, thresholds tuned | manual | date: |
| F4 | Clock skew measured ≥ 1 time, within acceptable range | manual | date: |
| F5 | Partial fill scenario handled correctly in testnet | manual | date: |
| F6 | Fee model within 1 % of actual exchange fees | manual | date: |

---

## Track G — Governance

| # | Check | Status | Notes |
|---|-------|--------|-------|
| G1 | Model promotion state machine: canary→prod sim passed | `[auto]` | |
| G2 | Hot-swap test (agent replaced mid-run) passes | `[auto]` | |
| G3 | Pre-trade compliance: all G6–G9 rules have tests | `[auto]` | |
| G4 | `docs/phase7/promotion_criteria.md` exists and is complete | `[auto]` | |

---

## Security & Secrets

| # | Check | Status | Notes |
|---|-------|--------|-------|
| S1 | Zero plaintext secrets in repo (`git log --all -S "secret"` clean) | manual | |
| S2 | Pre-commit secret-scanning hook active (`pre-commit run --all`) | manual | |
| S3 | Exchange API key has **Read + Trade** only — **Withdraw disabled** | manual | confirm on exchange UI |
| S4 | `SecretProvider` returns keys without logging them | `[auto]` | |
| S5 | AuditLogger redaction filter hides credentials in log entries | `[auto]` | |

---

## Risk Configuration

| # | Check | Status | Notes |
|---|-------|--------|-------|
| R1 | `max_drawdown_threshold` set (default 20 %) | `[auto]` | |
| R2 | `daily_loss_limit` set in OrderManager config | `[auto]` | |
| R3 | `limits.per_symbol_notional_max` configured | `[auto]` | |
| R4 | `limits.portfolio_notional_max` configured | `[auto]` | |
| R5 | `limits.leverage_max` configured | `[auto]` | |
| R6 | UnifiedRiskManager is the only risk path (no old-API callers) | `[auto]` | |

---

## Operational Readiness

| # | Check | Status | Notes |
|---|-------|--------|-------|
| O1 | Kill switch tested: `python scripts/kill_switch.py` halts within 5 s | `[auto]` | |
| O2 | Kill switch keyboard shortcut documented in this checklist | manual | shortcut: |
| O3 | `docs/runbook/failures/*.md` — operator has read all 5 files | manual | initials: |
| O4 | `scripts/verify_audit_log.py` passes on current audit chain | `[auto]` | |
| O5 | StateStore checkpoint is fresh (< 24 h old) | `[auto]` | |
| O6 | Postmortem template exists at `docs/runbook/postmortem_template.md` | `[auto]` | |
| O7 | Alerter configured with ≥ 1 channel (Telegram / webhook) | manual | channel: |

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
