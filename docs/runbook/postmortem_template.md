# Postmortem Template (G14)

**Rule**: File within 24 hours of any incident that causes:
- Unintended live order(s)
- Kill switch activation
- Position mismatch > threshold
- Daily loss limit breach
- Exchange API error lasting > 60 s
- Model NaN / shutdown due to feed staleness

Copy this file to `docs/runbook/postmortems/YYYY-MM-DD_<slug>.md` and fill it in.

---

## Incident Summary

| Field | Value |
|-------|-------|
| **Date / Time (UTC)** | |
| **Duration** | |
| **Severity** | P1 (loss) / P2 (halt, no loss) / P3 (degraded, no halt) |
| **Affected symbol(s)** | |
| **Total P&L impact** | |
| **Author** | |

---

## Timeline

_Use UTC timestamps. Include every significant event._

| Time (UTC) | Event |
|------------|-------|
| HH:MM | First anomaly detected (describe) |
| HH:MM | Kill switch / halt triggered |
| HH:MM | Position confirmed flat |
| HH:MM | Root cause identified |
| HH:MM | Recovery action taken |
| HH:MM | System back to normal / retired for investigation |

---

## Root Cause

_One paragraph. Be specific: which line of code, which config value, which market event._

---

## Impact

- **Positions affected**: 
- **Gross P&L loss/gain**:
- **Slippage during emergency liquidation**:
- **Time trading was halted**:
- **Any external impact** (e.g., exchange rate limits hit):

---

## What Went Well

_List 2–5 things the system or operator did correctly._

- 
- 

---

## What Went Wrong

_List 2–5 things that failed or were missing._

- 
- 

---

## Action Items

_Each item must have an owner and a due date. No action items without both._

| # | Action | Owner | Due |
|---|--------|-------|-----|
| 1 | | | |
| 2 | | | |

---

## Audit Log Evidence

```bash
# Retrieve relevant window from audit log
python - <<'EOF'
import json, sys
start = "2026-01-01T00:00:00"  # replace
end   = "2026-01-01T01:00:00"  # replace
records = [json.loads(l) for l in open("audit_log/audit.jsonl") if l.strip()]
window = [r for r in records if start <= r["ts"] <= end]
for r in window:
    print(r["ts"], r["type"], r.get("payload", {}))
EOF
```

Paste relevant output here:

```
(paste audit log excerpt)
```

---

## Lessons Learned

_One or two sentences that a new engineer reading this 6 months from now should take away._

---

## Checklist Before Restarting

- [ ] Position confirmed flat on exchange
- [ ] Root cause fixed or workaround in place
- [ ] Relevant tests added or updated
- [ ] `docs/runbook/go_live_checklist.md` still fully green
- [ ] If model was culprit: retrain or rollback before restart
- [ ] Audit chain verified after restart (`python scripts/verify_audit_log.py`)
