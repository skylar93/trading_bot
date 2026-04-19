# Slippage Model Calibration

**Week 74 (F15)** — `deployment/analysis/slippage_model.py`

## What is slippage?

Slippage is the difference between the price expected at order submission and
the actual fill price.  For a market order of size Q at expected price P:

```
slippage_frac = |fill_price - expected_price| / expected_price
slippage_cost = slippage_frac × Q × fill_price
```

In paper mode every order fills instantly at the last mid-price.  This
understates execution cost.  A calibrated model injects realistic slippage so
that paper P&L matches live P&L more closely.

---

## Model: Linear Regression

### Features

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `intercept` | 1 | base slippage floor |
| `log_volume` | log(1 + bar_volume) | larger market → better liquidity → less slippage |
| `realized_vol` | annualised vol at submission | volatile market → wider spreads |
| `side_enc` | 0=buy, 1=sell | buy pressure / sell pressure asymmetry |
| `size_frac` | order_size / bar_volume | market-impact proxy |

### Target

`slippage_frac` = |fill_price − expected_price| / expected_price

### Estimation

Ordinary least squares with L2 regularisation (ridge λ = 1e-6):

```
β = (XᵀX + λI)⁻¹ Xᵀy
```

### Prediction

```python
model = SlippageModel()
model.fit(observations)
frac = model.predict(volume=1e6, realized_vol=0.02, side="buy", size=0.1)
# → e.g., 0.00047  (4.7 bps)
```

Predictions are clipped to `[0, max_slippage_frac]` (default 2%) to prevent
runaway estimates on out-of-distribution inputs.

---

## Data Collection

Observations are collected from `OrderManager` fill events.  Each fill produces
a `SlippageObservation`:

```python
from deployment.analysis.slippage_model import SlippageObservation

obs = SlippageObservation(
    side="buy",
    order_size=0.01,          # BTC
    fill_price=30_012.50,
    expected_price=30_010.00,
    bar_volume=150.0,         # BTC volume in the bar
    realized_vol=0.018,       # 1-day realised vol
)
```

In live/sandbox mode, fill data comes directly from CCXT fill events.
In paper mode, synthetic observations can be generated via `SlippageModel.record()`.

---

## Calibration Workflow

1. **Collect** ≥ 10 live or sandbox fill observations.
2. **Fit**:
   ```python
   result = model.fit(observations)
   print(result)
   # {'n_samples': 42, 'r2': 0.71, 'coeffs': [...], 'fitted': True}
   ```
3. **Inspect** `model.summary()` — check R², mean/median slippage, coefficients.
4. **Inject** into `OrderManager` paper mode via `fee_model` or custom `_draw_partial_fill_ratio` hook.
5. **Re-calibrate** weekly (or after regime change) as market conditions shift.

---

## Benchmarks & Validation

| Metric | Target | How to measure |
|--------|--------|---------------|
| R² | ≥ 0.5 | `model.fit(obs)["r2"]` |
| Mean predicted error | < 1 bp | compare `predict()` vs held-out `slippage_frac` |
| Fee model error | < 1% | `test_execution_realism.py::test_fee_model_accuracy` |

If R² < 0.3, the model has little predictive power.  Possible causes:
- Insufficient data (< 50 observations)
- High noise (market microstructure dominated by randomness)
- Missing features (e.g., bid-ask spread, order book depth)

---

## Limitations & Future Work

- **Linear model**: does not capture non-linear market impact (e.g., Kyle's λ).
  Future: gradient boosting or Almgren-Chriss model.
- **Single asset**: model fitted per symbol.  Multi-asset calibration is Phase 8.
- **No order book depth**: size_frac uses bar volume as liquidity proxy, not L2
  book.  More accurate with exchange depth snapshot.
- **Stale calibration**: slippage regime can shift (funding rate spikes, low-vol
  weekends).  Automated re-calibration trigger is a Phase 8 item.
