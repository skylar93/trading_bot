# Week 59: Runbook & Disaster Recovery Drills (S16-S19)

**Date**: 2026-04-09
**Branch**: claude/cranky-moore
**Sections**: S16 – S19 (Track A final)

---

## What was done

### S16 — Runbook documents

Created `docs/runbook/` with the following structure:

```
docs/runbook/
├── README.md                         # overview + quick-start decision tree
├── oncall_checklist.md               # S19 (see below)
└── failures/
    ├── data_feed_stale.md
    ├── exchange_api_error.md
    ├── drawdown_kill_switch.md
    ├── crash_recovery.md
    └── model_nan_output.md
```

Each failure doc: symptom list → log locations → diagnosis steps → recovery
steps → post-incident checklist.

### S17 — Drill scripts

- `tests/deployment/test_drills.py` — 7 pytest tests covering 3 scenarios:
  - `TestDrillCrashMidEpisode` (2 tests): SIGKILL simulation via StateStore restore
  - `TestDrillDataGap` (2 tests): +15% and -20% single-step price gap injection
  - `TestDrillRiskBreach` (3 tests): drawdown>10% → kill-switch → position zero

- `scripts/drills/run_drill.py` — CLI runner with `--scenario {crash_mid_episode,data_gap,risk_breach,all}`
  - All 3 drills: **exit 0** on first run.

### S18 — GitHub Actions nightly job

`.github/workflows/drills.yml`:
- Triggers: daily 03:00 UTC + `workflow_dispatch`
- Runs `pytest tests/deployment/test_drills.py` then `scripts/drills/run_drill.py --scenario all`
- Not PR-blocking (separate workflow from `main.yml`)
- Uploads drill-failure-logs artifact on failure

### S19 — On-call checklist

`docs/runbook/oncall_checklist.md` with three sections:
1. **Before starting a trading session** (env, state, audit, risk config)
2. **Daily morning checks** (process alive, log health, checkpoint freshness)
3. **Weekend checks** (drill job status, disk usage, reconciliation, model age)

---

## Test results

| Suite | Before | After | Delta |
|-------|--------|-------|-------|
| Full regression | 1386 passed, 0 failed | 1455 passed, 0 failed | +69 |
| New drill tests | — | 7/7 passed | +7 |
| Drill CLI | — | 3/3 exit 0 | ✓ |

No regressions. pytest.ini ignore list unchanged.

---

## Gotchas

- `run_drill.py` adds `_REPO_ROOT` to `sys.path` so it can be run from project
  root without installing the package (for CI environments that don't `pip install -e .`
  before the CLI step — the drills workflow does install it but the guard is safe).
- `drill_data_gap` uses a 15% up-gap and a 20% down-gap.  The 20% down-gap
  does NOT trigger the kill-switch in `test_large_gap_does_not_crash` because
  `max_drawdown_threshold=0.99` (99%), while `test_gap_does_not_corrupt_position`
  uses `action=0.5` which keeps position small enough that portfolio stays above
  even a 20% price crash (cash portion dominates).
- `risk_breach` scenario uses `action=0.8` (large long) + 35% price crash from
  peak.  This reliably triggers the 10% kill-switch within the first crash bar.
- The `test_no_further_trades_after_shutdown` test checks that no **buy** trades
  occur after the liquidation sell, not that no trades at all occur (the
  liquidation itself is a sell trade, which is expected).

---

## Track A completion

Week 59 completes Track A (Ops Readiness, Weeks 56-59):

| Week | Title | Status |
|------|-------|--------|
| 56 | State Persistence | ✓ merged |
| 57 | Immutable Audit Log | ✓ merged |
| 58 | Secrets Management | ✓ merged |
| **59** | **Runbook & Drills** | **this PR** |

**Next**: Track B — Architecture Consolidation (Week 60: RiskManager 통합)
