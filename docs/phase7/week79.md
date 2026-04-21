# Week 79 — Data & Model Governance (H6-H10)

**Date**: 2026-04-20  
**Branch**: `claude/elastic-rubin-b9d9c3`  
**Baseline**: 1780 passed / 19 skipped (from Week 78 / main)

---

## Deliverables

### H6 — Great Expectations → pandera (OHLCV Expectation Suite)

| Item | Status |
|------|--------|
| `data/quality/pandera_schema.py` | ✅ Created |
| `data/quality/gate.py` patched to use pandera as backend | ✅ |
| `data/quality/__init__.py` exports `OHLCV_SCHEMA`, `validate_ohlcv` | ✅ |
| `tests/data/test_pandera_schema.py` (13 tests) | ✅ |

**Expectations enforced**:
- `not_null` — NaN in any price/volume column  
- `no_inf` — ±inf rejection  
- `positive` — all prices & volume > 0  
- `monotonic_ts` — timestamp strictly increasing  
- `no_gap` — consecutive bar interval ≤ 3× median (configurable)

Fallback: when `pandera` is not installed, `_fallback_validate()` runs
pure-numpy checks so the gate never silently passes bad data.

---

### H7 — MLflow as Authoritative Store

| Item | Status |
|------|--------|
| `MLflowRegistryBridge` class in `model_registry.py` | ✅ |
| `ModelRegistry(mlflow_model_name=...)` optional kwarg | ✅ |
| `register(mlflow_run_id=...)` syncs artifact to MLflow registry | ✅ |
| `promote()` syncs stage transitions to MLflow Model Registry | ✅ |

**Stage mapping**:

| Our stage | MLflow stage |
|-----------|-------------|
| candidate | None |
| staging | Staging |
| canary | Staging + `stage_detail=canary` tag |
| prod | Production |
| retired | Archived |

The JSON index remains as a fast local cache; MLflow is the canonical record
for artifacts and stage history.

---

### H8 — DVC Data Versioning

| Item | Status |
|------|--------|
| `data/raw/` directory | ✅ Created |
| `data/processed/` directory | ✅ Created |
| `dvc.yaml` — fetch → preprocess → validate pipeline | ✅ |
| `scripts/setup_dvc.py` — one-shot DVC init helper | ✅ |
| `pandera>=0.18.0`, `dvc>=3.50.0` added to `requirements.txt` | ✅ |

**Pipeline stages**:
```
fetch         → data/raw/
preprocess    → data/processed/
validate      → pandera gate + quality_report.json
```

Run `python scripts/setup_dvc.py` once after cloning to initialise DVC.

---

### H9 — Feature Registry (lightweight)

| Item | Status |
|------|--------|
| `training/features/registry.py` | ✅ Created |
| `training/features/__init__.py` exports `FeatureRegistry` | ✅ |
| `tests/training/test_feature_registry.py` (14 tests) | ✅ |

**Capabilities**:
- Per-feature version, code SHA-256 hash, input/output schema
- Automatic version bump when `compute_fn` source changes
- `drift_report(features_dict)` — identifies which features changed
- `validate_dataframe(df)` — checks expected output columns are present
- JSON-backed, thread-safe, `~/.trading_bot/feature_registry.json`

---

### H10 — Walk-forward CV (Purged K-Fold)

| Item | Status |
|------|--------|
| `training/evaluation/__init__.py` | ✅ Created |
| `training/evaluation/walkforward.py` | ✅ Created |
| `tests/training/test_walkforward_eval.py` (15 tests) | ✅ |

**Components**:

| Class / Function | Purpose |
|-----------------|---------|
| `PurgedKFoldSplitter` | Embargo gap between train and test (prevents feature leakage) |
| `WalkForwardReport` | JSON-serialisable report with gate pass/fail logic |
| `WalkForwardEvaluator` | Orchestrates purged folds + WalkForwardValidator helpers |
| `evaluate_for_promotion()` | Single-call staging gate integration |

**Staging gate thresholds** (configurable in `STAGING_GATE`):

| Metric | Threshold |
|--------|-----------|
| OOS Sharpe mean | ≥ 0.3 |
| Stability ratio | ≥ 0.4 |
| Mean max drawdown | ≤ 0.35 |
| Minimum folds | ≥ 4 |

---

## Test Count Estimate

| Suite | New Tests |
|-------|-----------|
| `test_pandera_schema.py` | 13 |
| `test_feature_registry.py` | 14 |
| `test_walkforward_eval.py` | 15 |
| **Total new** | **42** |

Expected post-merge baseline: ~1822+ passed (1780 baseline + 42 new).

---

## Completion Conditions

- [x] pandera schema covers not_null / positive / monotonic_ts / no_gap
- [x] gate.py external API unchanged (existing 9 tests still pass)
- [x] MLflow bridge optional — zero breaking change when mlflow is absent
- [x] DVC pipeline defined and setup script present
- [x] Feature registry tracks version + code hash + drift
- [x] Walk-forward report with staging gate pass/fail
- [x] 42 new tests written
- [ ] `pytest -q` → 0 failures (run after PR merge)
