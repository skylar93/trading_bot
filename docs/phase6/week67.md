# Week 67: Drift & Regime (S56-S60)

**Track D — Model Lifecycle**
**Date**: 2026-04-11
**PR scope**: S56–S60

---

## What was done

### S56 — FeatureDriftDetector
Added `FeatureDriftDetector` to `training/monitoring/drift_detector.py`.

- Wraps one `DriftDetector` (ADWIN or Page-Hinkley) per named feature.
- Accepts both dict and numpy array inputs.
- NaN/inf values skipped silently (no state change for that feature).
- Per-feature alarms fire to: `TradingAlerter.notify_drift()` + AuditLogger `feature_drift_alarm` record.
- `any_drift`, `drift_features`, `n_detections`, `total_detections` properties for easy querying.
- `reset(feature_name=None)` for targeted or full reset after retraining.

### S57 — Regime detection live wire
Extended `PaperTrader` and `MetricsExporter`:

- `PaperTrader.__init__` gains `regime_detector`, `feature_drift_detector`, `on_regime_change` optional params.
- `_check_regime()` called every step; uses `self._price_history` as the window.
- Regime change (argmax flip): logs at INFO, writes `regime_change` audit event, calls `on_regime_change(prev, new, probs)` hook.
- Hook exceptions are caught and logged (trading loop cannot crash from hook errors).
- `MetricSnapshot` gains `current_regime: int = -1` and `feature_drift_alarms: int = 0`.
- `MetricsExporter.to_json()` exposes both new fields.
- `_log_step_metrics()` now called unconditionally (previously only when mlflow_manager set); MLflow guard is internal.
- Config: `config/monitoring.yaml` has new `use_regime_detection`, `regime_n_regimes`, `regime_method`, `use_feature_drift_detection`, `feature_drift_method` knobs.

### S58 — RetrainingTrigger
New file: `deployment/monitoring/retraining_trigger.py`

- Two conditions (independent, both checked per call):
  - **A** (drawdown): `drawdown_pct >= drawdown_trigger_pct` (default 15%)
  - **B** (drift): `drift_count >= drift_alarm_trigger_count` (default 5)
- Cooldown (`cooldown_steps`, default 100) prevents flooding.
- Drawdown takes priority if both conditions fire simultaneously.
- Firing: logs WARNING + writes AuditLogger `retraining_trigger` record + calls `on_trigger` callback.
- `events` property for history; `reset()` for tests and drills.
- Thread-safe (single RLock).

### S59 — ModelRegistry (lightweight)
New package: `training/registry/` (`__init__.py` + `model_registry.py`)

- JSON file backend, atomic write (temp-file + rename).
- Versions auto-increment (`v1`, `v2`, …).
- Per-version fields: `name`, `path`, `metrics`, `config`, `tags`, `created_at`.
- `register()`, `get()`, `latest()`, `list_versions()`, `delete()`, `update_metrics()`.
- Thread-safe; survives reload (persisted immediately on every write).
- No MLflow, no database required.

### S60 — Tests
`tests/deployment/test_week67_drift_regime.py` — 54 tests, 0 failures.

- `TestFeatureDriftDetector` (17 tests): init, dict/array update, NaN/inf skip, distribution shift detection, properties, reset.
- `TestRegimeLiveWire` (7 tests): hook invocation, no-op on unchanged regime, MetricsExporter export, audit log, hook exception safety, insufficient history guard.
- `TestRetrainingTrigger` (12 tests): each condition individually, priority, cooldown, accumulation, reset, callback, audit.
- `TestModelRegistry` (13 tests): full CRUD, disk persistence, JSON validity.
- `TestWeek67Integration` (5 tests): PaperTrader end-to-end with all Week 67 components.

---

## Regression

| Baseline | This week |
|----------|-----------|
| 1386 passed | 1754 passed |
| 19 skipped | 19 skipped |
| 0 failed | **0 failed** |

All existing tests pass. pytest.ini ignore list unchanged.

---

## Why / rationale

- **FeatureDriftDetector** catches input distribution shift *before* it manifests as performance degradation. Returns-only drift detection (existing `DriftDetector`) is a lagging indicator; features drift first.
- **Regime live wire** bridges the gap between the trained `RegimeDetector` (used in backtesting) and the live trading loop. The hook is intentionally minimal (log-only by default) to avoid coupling the detector to a specific model-switching strategy — that belongs in Week 68 shadow deploy.
- **RetrainingTrigger** makes the retraining signal explicit and auditable. The operator is still in the loop (no automatic retraining), but the trigger creates a paper trail.
- **ModelRegistry** is the prerequisite for Week 68's `rollback_model.py`. Even if MLflow is added later, this file-based registry provides a zero-dependency fallback.

---

## Gotchas

1. `_log_step_metrics()` was only called when `mlflow_manager` is set — changed to unconditional. The internal MLflow block still has a try/except guard, so no regression.
2. ADWIN detection speed depends heavily on the signal-to-noise ratio. The distribution-shift test uses `confidence=0.1` (faster) and a large shift (`0.5` mean) to keep the test deterministic and fast. Production should use the default `0.002`.
3. `RegimeDetector.predict()` requires at least 5 price points; `_check_regime()` skips silently below that threshold. On first regime assignment (`-1 → X`), the hook fires — this is intentional (initial regime detection is a meaningful event).
4. `RetrainingTrigger` uses step-based cooldown, not time-based. In live trading where steps map to bar intervals this is stable; for high-frequency setups consider adding a `cooldown_seconds` option.

---

## Phase 7 candidates surfaced this week

- Auto-retraining pipeline triggered by `RetrainingTrigger` events (currently manual only).
- ModelRegistry cloud sync (S3 / GCS) for multi-machine setups.
- Regime-conditional position sizing (pass `current_regime` into `RiskManager`).
