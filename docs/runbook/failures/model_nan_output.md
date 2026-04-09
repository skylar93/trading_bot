# Failure: Model NaN / Inf Output

**Scenario**: The RL model's `predict()` call returns `NaN` or `inf` values in
the action array.  This causes `_execute_action` to produce bad order sizes or
prices, and can propagate into the portfolio history.

---

## Symptoms

- Log line: `"WARNING NaN/inf action received, skipping step"` (if guard is in
  place) or arithmetic errors further downstream.
- `portfolio_history` contains `nan` entries.
- `StateStore.save_snapshot` raises `ValueError: Refusing to persist non-finite value`.
- Audit log `model_decision` payload contains `"action": null` or `"action": NaN`.

## Log locations

| What | Where |
|------|-------|
| PaperTrader stdout | `logs/paper_trader.log` |
| Audit log | `audit_log/audit.jsonl` (model_decision entries) |

---

## Diagnosis steps

1. **Find first NaN action in logs**

   ```bash
   grep -n "nan\|inf\|NaN\|Inf" logs/paper_trader.log | head -20
   ```

2. **Inspect the observation that triggered the NaN**

   The audit log stores `obs_hash = sha256(obs.tobytes())`.  To re-create the
   failing observation you need the data slice at the step in question.
   Use `state.step` from the checkpoint to identify the data row.

3. **Reproduce with a minimal test**

   ```python
   from training.factories import load_agent
   import numpy as np

   agent = load_agent("checkpoints/model_latest.zip")
   bad_obs = np.full((20, 5), np.nan, dtype=np.float32)  # trigger NaN obs
   action, _ = agent.predict(bad_obs, deterministic=True)
   print(action)  # should reproduce NaN
   ```

4. **Check if the observation is the cause** (NaN in features):

   - Review feature pipeline for division-by-zero (e.g., ATR over zero-volume bar).
   - Check `SingleAssetRLTradingEnv._get_obs()` for NaN propagation.

---

## Recovery steps

1. **Immediate: restart bot with NaN guard enabled**

   In config set:
   ```yaml
   paper_trading:
     action_nan_guard: true   # skips step if action contains NaN/inf
   ```

   If the guard is not yet implemented, add a one-line check in
   `paper_trader.py::run()` after `agent.predict()`:
   ```python
   if not np.all(np.isfinite(action)):
       logger.warning("NaN/inf action at step %d, skipping", step)
       continue
   ```

2. **Long-term: identify root observation**

   - If the bad observation was caused by a stale/missing market data bar, add
     zero-volume bar rejection in the feature pipeline.
   - If caused by model weight corruption: retrain from last good checkpoint.

3. **Retrain** if NaN outputs are frequent (>5 consecutive steps):

   ```bash
   python scripts/run_full_pipeline.py --config config/local_3060ti.yaml
   ```

---

## Post-incident checklist

- [ ] Confirm portfolio_history is finite after recovery:
  `python -c "from deployment.persistence.state_store import StateStore; snap = StateStore('state/paper_trader.db').load_latest(); print('hist NaN:', any(v != v for v in snap['portfolio_history']))"`
- [ ] Audit chain verified: `python scripts/verify_audit_log.py audit_log/audit.jsonl`.
- [ ] Root observation identified and data pipeline fix deployed.
- [ ] Consider adding feature NaN test to CI if not already present.
