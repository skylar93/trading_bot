# Phase 8 — Evidence-First + Signal Gate (Weeks 87–95)

**Plan**: `/Users/skylar/.claude/plans/phase8-restructured.md`
**Supersedes**: `/Users/skylar/.claude/plans/phase8-early-items.md`
**Started**: 2026-04-27

---

## Core Thesis

Operations and safety layers are thick enough (Phase 7.6 complete, 11 active safety nets). The
next bet is:

1. **Statistical evidence that the strategy makes money** — a single publishable report showing
   walk-forward OOS Sharpe, bootstrap CI, permutation p-value, DSR, regime-conditional breakdown,
   and baseline comparisons.
2. **An automatic code-level gate** that blocks `exchange_mode: live` until that evidence exists
   and passes defined thresholds.

These two items (A0 + A0.5) close before anything else. Structural debt (MLflow unification,
filter chain refactor) is deferred until A0 delivers a GO signal — cost-avoidance against
sunk-cost risk.

---

## Track Progress

| Track | Item | Status | PR |
|-------|------|--------|----|
| P0 | Baseline sync + docs | ✅ 2026-04-27 | PR-P0 |
| A | A0 Strategy evidence pack | in progress | PR-A0 |
| A | A0.5 Live-readiness signal gate | pending | PR-A0.5 |
| B | A5 Model prediction-quality drift | pending | PR-A5 |
| B | A6 Cost decomposition dashboard | pending | PR-A6 |
| C | A3 Capacity stress test | pending | PR-A3 |
| C | A4 Scale-up protocol doc | pending | PR-A4 |
| D | A7 Agent ablation study | pending | PR-A7 |
| E | E1–E8 Missing safety nets | pending | PR-E1, PR-E2 |
| F | A1 MLflow refactor (deferred) | deferred (A0 GO first) | — |
| F | A2 Filter chain refactor (deferred) | deferred (A0 GO first) | — |

---

## GO/NO-GO Criteria (A0)

The operator makes the GO/NO-GO call after reviewing `docs/phase8/strategy_evidence_v1.md`.
Automated thresholds enforced by `deployment/governance/live_signal_gate.py` (A0.5):

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| Net Sharpe (OOS, walk-forward aggregate) | > 0.5 | higher is better |
| Deflated Sharpe Ratio (DSR) | > 0 | positive |
| Bootstrap 95% CI lower bound | > 0 | positive |
| Permutation p-value | < 0.05 | lower is better |
| Max regime DD (crisis) | < 30% | lower is better |
| At least 1 baseline outperformed | — | required |

**GO** → proceed with Track B–F.  
**NO-GO** → Phase 8-Alpha: feature rethink + reward shaping review. Track B–F on hold. No
automatic progression; operator + Opus collaboration required for the new plan.

---

## Files in this Directory

| File | Purpose |
|------|---------|
| `README.md` | This document — Phase 8 intent + progress |
| `strategy_evidence_v1.md` | A0 evidence pack (walk-forward + stat tests + regime + baselines) |
| `reward_audit.md` | A0.7 reward function audit (net-of-cost verification) |
| `drill_postmortem_2026-04-27.md` | 72h autonomous drill postmortem (operator fill) |
| `agent_ablation_decision.md` | A7 ablation results + ensemble decision (Week 92) |
| `capacity_stress_*.md` | A3 capacity stress results (Week 91) |
| `scale_up_protocol.md` | A4 operator decision doc (Week 91) |

---

## Execution Constraints

1. Do **not** touch: `logs/alerts.jsonl`, `logs/fault_injection.jsonl`,
   `logs/drill_snapshots.jsonl`, `logs/incidents/*.md`, `state/paper_trader.pid`.
2. All new docs go under `docs/phase8/`. Never modify `docs/phase7/` or `docs/phase7.6/`.
3. Test fixtures must use `tmp_path` / `monkeypatch` — never touch real `logs/` or `state/`.
4. A1/A2 must not start before A0 GO signal.
5. A0 GO/NO-GO is an **operator decision** — code only generates the evidence.
