# Week 77 Baseline — Go-Live Checklist & Sign-off Gate (G11-G14)

**Date**: 2026-04-19
**Branch**: claude/fervent-newton-b26058
**Prereq baseline**: Week 76 (G6-G10, pre-trade compliance), main branch

---

## Deliverables

| Task | File | Status |
|------|------|--------|
| G11 First Dollar Checklist | `docs/runbook/go_live_checklist.md` | ✅ |
| G12 $100 Drill | `scripts/first_dollar_drill.py` | ✅ |
| G13 Kill Switch | `scripts/kill_switch.py` + `PaperTrader` SIGUSR1 | ✅ |
| G14 Postmortem Template | `docs/runbook/postmortem_template.md` | ✅ |
| Tests | `tests/deployment/test_go_live_gate.py` | ✅ |

---

## Code Changes

### `deployment/execution/order_manager.py`
- Added `cancel_all_orders()` — iterates pending/partial orders, calls `cancel_order()` for each; returns count cancelled.

### `deployment/paper_trader.py`
- Added `import os, signal` to imports.
- `__init__`: writes PID to `state/paper_trader.pid` (configurable via `pid_file` config key); registers `SIGUSR1` → `_handle_kill_signal`.
- `_handle_kill_signal()`: SIGUSR1 handler — calls `order_manager.cancel_all_orders()` then `_trigger_shutdown("kill_switch: SIGUSR1")`.
- `_trigger_shutdown()`: extended — now calls `cancel_all_orders()` before liquidating, and removes PID file on exit.

### `scripts/kill_switch.py` (new)
- Reads PID from `state/paper_trader.pid` (or `--pid` direct).
- Sends `SIGUSR1` to the running PaperTrader process.
- Polls for process exit, confirms within `--timeout` (default 5 s).
- Exit codes: 0 clean halt, 1 already gone, 2 timeout, 3 error.

### `scripts/first_dollar_drill.py` (new)
- `--check-only`: runs structural auto-checks (ignore count, old API, docs, kill switch script, checkpoint freshness, audit chain, risk config).
- `--capital N`: runs `PaperTrader` simulation with N dollars, verifies it completes without error.
- `--skip-kill-switch-test`: skip the in-process timing test (useful in CI).
- Writes JSON report via `--report`.

### `docs/runbook/go_live_checklist.md` (new)
- Comprehensive go-live checklist covering Track E/F/G checks, security, risk config, and operational items.
- Sections: E hardening, F connectivity, G governance, Security, Risk Config, Operational.
- Every programmatic item marked `[auto]` for `first_dollar_drill.py`.

### `docs/runbook/postmortem_template.md` (new)
- Standard 24 h postmortem template.
- Sections: Incident Summary, Timeline, Root Cause, Impact, What Went Well, What Went Wrong, Action Items, Audit Log Evidence, Lessons Learned, Checklist Before Restarting.

---

## Test Results

Run: `pytest tests/deployment/test_go_live_gate.py -v`

Expected groups:
- `TestCancelAllOrders` (3 tests)
- `TestKillSwitch` (4 tests)
- `TestKillSwitchSignal` (2 tests)
- `TestDocuments` (7 tests)
- `TestFirstDollarDrill` (2 tests)

**Completion criterion**: all pass, kill switch timing < 5 s confirmed.

---

## Go-Live Gate Status

| Gate | Status |
|------|--------|
| Track E (Weeks 69-71) hardening debt | ✅ (Week 71 completed) |
| Track F (Weeks 72-74) connectivity | ✅ (Week 74 completed) |
| Track G (Weeks 75-77) governance | ✅ (Week 77 — this PR) |
| Go-live checklist signed off | ⏳ (manual operator sign-off pending) |
| $100 drill passed | ⏳ (simulation only — testnet drill pending) |
