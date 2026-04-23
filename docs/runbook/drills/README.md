# Runbook Drill Log — Week 84 (R18 / G10)

Drills must be performed before go-live and re-run after any significant system
change.  Each execution is recorded in a dedicated file using the naming
convention `YYYYMMDD_{drill_name}.md`.

## Required Drills (go-live gate)

| # | Drill | Script / Action | Pass Criteria |
|---|-------|-----------------|---------------|
| 1 | **Feed Stale** | Kill ccxt_live feed process | PaperTrader halts within 60s of feed silence |
| 2 | **Kill Switch** | `python scripts/kill_switch.py` | All orders cancelled + position flat within 5s |

`first_dollar_drill.py` enforces that ≥ 2 drill records exist in this directory
before go-live sign-off.

## Drill File Template

Each file must include:
- **Date / time** of execution
- **Environment** (sandbox / paper)
- **Scenario** description
- **Steps performed** (numbered)
- **Expected outcome** vs **Actual outcome**
- **Elapsed times** for each critical step
- **Issues found** (if any)
- **Resolution** (if issues found)

## Completed Drills

| Date | Drill | Result |
|------|-------|--------|
| 2026-04-23 | [Kill Switch](20260423_kill_switch.md) | PASS — 2.1s |
| 2026-04-23 | [Feed Stale](20260423_feed_stale.md) | PASS — 42s to halt |

---

*Runbook drills are required pre-live.  Re-run after any OrderManager, PaperTrader, or
kill_switch.py change.*
