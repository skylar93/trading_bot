# Week 65: Data Pipeline Safety (S47-S50)

**Date**: 2026-04-11
**Branch**: claude/loving-banzai
**Sections**: S47, S48, S49, S50

---

## What was done

### S47 — Feed staleness halt

Added a staleness interface to the DataSource hierarchy:

- `DataSource.last_updated_at()` — returns `time.monotonic()` timestamp of last update, or `None` for static sources.
- `DataSource.is_stale(max_staleness_sec)` — base implementation uses `last_updated_at()`.
- `MockLiveDataSource`: now records `_last_updated_at` on `__init__`, `tick()`, and `reset()`.  Also accepts `max_staleness_sec` constructor arg for convenience.
- `PaperTrader`: accepts optional `data_source` and `audit_logger` parameters.  New config block `data_pipeline_safety.staleness_enabled / max_staleness_sec` controls the check.  `_check_feed_staleness()` is called at the top of each loop iteration (before building obs), triggers `_trigger_shutdown()` + audit event on stale feed.

### S48 — NaN/inf in computed features

- `PaperTrader._check_obs_nan(obs, step)`: called after `_build_observation()` returns a non-None array.
- Checks `np.all(np.isfinite(obs))`.  On failure: skips step (`continue`), increments `_consecutive_nan_steps`, logs warning + audit event.
- If `consecutive >= nan_halt_after_n` (and `nan_halt_after_n > 0`): calls `_trigger_shutdown()` with `nan_in_features` reason.
- Counter resets to 0 on any clean observation.
- Config: `data_pipeline_safety.nan_check_enabled`, `nan_halt_after_n`.

### S49 — Survivorship bias warning

New file: `data/quality/survivorship.py`

- `BiasWarning` dataclass: `kind`, `severity`, `detail`.
- `SurvivorshipBiasChecker(min_lookback_bars, warn_single_asset)`:
  - `short_history`: fires when `len(df) < min_lookback_bars`.
  - `late_start`: fires when first timestamp > `expected_start` arg.
  - `single_asset_universe`: info-level reminder, always fires unless suppressed.
- `check_survivorship(df, ...)` module-level shortcut.
- Exported from `data/quality/__init__.py`.
- **Does not halt** — only emits warnings for caller to handle.

### S50 — Tests

New file: `tests/deployment/test_data_pipeline_safety.py`

- `TestFeedStalenessHalt`: 7 tests covering staleness detection on MockLiveDataSource, halt trigger, audit log event, disabled-staleness passthrough, no-data_source passthrough.
- `TestNanInfFeatureCheck`: 6 tests covering step skip on NaN, halt after N consecutive, counter reset on good obs, audit events, inf treated as NaN, disabled check passthrough.
- `TestSurvivorshipBiasChecker`: 14 tests covering all warning kinds, suppressions, datetime index, module shortcut, logger integration, symbol in message.
- `TestSurvivorshipAtBacktestStart`: 2 integration tests using real `test_data.csv`.

### Config

Added `DataPipelineSafetyConfig` to `config/schema.py` with Pydantic v2 validation for all 6 fields.  Added `FullConfig.data_pipeline_safety` field.  Added `data_pipeline_safety:` block to `config/risk.yaml`.

---

## Why

Without feed staleness detection, a trader could silently re-use a stale price for multiple steps — which in live trading means the agent acts on wrong prices while real prices have moved.  The staleness halt ensures the system fails loudly rather than silently continuing with stale data.

NaN/inf in features causes `agent.predict()` to return garbage actions (or crash).  Skipping bad steps prevents broken trades; halting after N consecutive prevents the system from running indefinitely in a broken data state.

Survivorship bias in backtests inflates reported Sharpe ratios and win rates.  The checker surfaces this risk at initialisation so the developer can decide whether to proceed or improve their dataset.

---

## Gotchas

1. **`MockLiveDataSource` init time**: `_last_updated_at` is recorded at `__init__` time.  In tests that construct the source and then sleep before calling `tick()`, the source may appear stale relative to the threshold.  Always call `tick()` or manually set `_last_updated_at` in tests that need a "just updated" feed.

2. **PaperTrader `data_source` is separate from price_stream**: `data_source` is only used for staleness checking in the current implementation.  The price stream still drives price updates.  In future live wiring, the data source would replace the price stream entirely.

3. **`nan_halt_after_n=0` disables the halt** (not the skip): a 0 value means "warn and skip forever, never halt".  This is intentional for paper trading environments where operators prefer the system to limp along.

4. **Schema field placement**: `data_pipeline_safety` lives in `config/risk.yaml` (alongside circuit breaker / fat finger) rather than a separate file, because it is a runtime safety control like the others.  It is also a top-level key in `FullConfig`, not nested under `risk_management`.

---

## Phase 7 candidates

- Wire `data_source` as the live price provider in `PaperTrader.run()` — currently it is only used for staleness checks.
- Multi-asset survivorship: diff current universe vs historical universe using a membership file.
- Persisted NaN event counts in StateStore for post-crash analysis.
