# Model Drift Detector — 5-Dimension Coverage (Phase 8 A5)

**Added**: 2026-04-28  
**Module**: `deployment/monitoring/model_drift.py`  
**Policy**: WARNING only, no halt (72h shadow mode, same as DeploymentDriftDetector)

---

## Why

`deployment/monitoring/drift_detector.py` covers reward / feature / schema drift.
**Model behaviour drift** — "the agent is making systematically different decisions than it used to" — was a blind spot.
This module closes that gap with 5 independent signals that a solo operator can monitor in real time.

---

## Dimensions

| # | Name | Signal | Threshold | Notes |
|---|------|---------|-----------|-------|
| 1 | **Action distribution KL** | KL(first-half ‖ second-half) of rolling action buffer | > 0.5 | Detects gradual policy shift. Fires when buffer ≥ `kl_window_steps`. |
| 2 | **Meta-controller weight collapse** | One agent weight > `meta_weight_collapse_pct` for full `collapse_window` steps | > 80% for 6h | Signals ensemble degeneration — one agent dominating the rest. |
| 3 | **Predicted vs realized return corr** | Spearman(predicted_return, realized_return) over 100-step window | < 0.05 | Silent when sample count < window. "Model no longer forecasts future." |
| 4 | **Per-regime hit rate** | Recent 100 trades per HMM regime vs training baseline | baseline − 20% | Requires `baseline_hit_rates` dict at init. Silent when < 10 regime samples. |
| 5 | **Action entropy collapse** | Normalised Shannon entropy of last 200 actions | < 0.5 (from `drift.action_entropy_min`) | Reuses existing `alerts.yaml` threshold. Agent fixation signal. |

---

## Integration

```python
from deployment.monitoring.model_drift import ModelDriftDetector
import yaml

with open("config/alerts.yaml") as f:
    alerts_cfg = yaml.safe_load(f)

detector = ModelDriftDetector(
    config=alerts_cfg,
    alerter=alerter,                          # optional TradingAlerter
    baseline_hit_rates={0: 0.55, 1: 0.50, 2: 0.45},  # from training eval
)

# Per step (in PaperTrader._check_model_drift or your own loop):
events = detector.update(
    action=float(action),
    predicted_return=model_pred,   # optional
    realized_return=step_return,   # optional
    regime=current_regime,         # optional
    trade_won=last_trade_won,      # optional
    meta_weights={"ppo": 0.4, "sac": 0.3, "td3": 0.2, "flag": 0.1},  # optional
)

# Dashboard / reporting:
snap = detector.snapshot()
```

**PaperTrader wiring** — pass via `model_drift_detector=` kwarg (default `None`, no-op):

```python
trader = PaperTrader(
    agent, config,
    model_drift_detector=detector,
)
```

**Dashboard** — `GET /model-drift` returns `detector.snapshot()` JSON when wired into `start_dashboard(metrics_exporter, model_drift_detector=detector)`.

---

## Config (`config/alerts.yaml`)

```yaml
model_drift:
  action_kl_threshold: 0.5
  meta_weight_collapse_pct: 0.80
  meta_weight_collapse_window_h: 6
  pred_realized_corr_min: 0.05
  pred_realized_window: 100
  hit_rate_drop_pct: 0.20
  shadow_mode_hours: 72
  steps_per_hour: 60
```

Dim-5 threshold (`action_entropy_min`) is read from `drift.action_entropy_min` (already present in config).

---

## Tests

`tests/monitoring/test_model_drift.py` — 21 cases across all 5 dimensions + shadow mode + integration.

```
TestActionKL                       (3 cases)
TestMetaWeightCollapse             (3 cases)
TestPredRealizedCorr               (3 cases)
TestRegimeHitRate                  (4 cases)
TestActionEntropy                  (2 cases)
TestShadowMode                     (2 cases)
TestIntegrationEntropyCollapse     (2 cases)
TestSnapshot                       (2 cases)
```

---

## Runbook

| Symptom | Most likely dimension | Action |
|---------|----------------------|--------|
| Sudden KL spike | Dim 1 — policy shift | Check recent retraining or data change |
| One agent weight → 100% | Dim 2 — collapse | Inspect meta-controller gradients |
| Corr drop to negative | Dim 3 — prediction quality | Check if feature pipeline changed |
| Win rate collapse in regime X | Dim 4 — regime shift | Check if HMM regime boundaries drifted |
| Entropy → 0 | Dim 5 — fixation | Agent stuck; may need retraining or temp halt |

All signals are **WARNING only**. The operator decides whether to halt.
