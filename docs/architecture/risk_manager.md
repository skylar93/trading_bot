# RiskManager Architecture — Week 60 Unification

## Background

Prior to Week 60, two concrete risk manager implementations existed in parallel:

| Class | File | VaR Method | Context |
|---|---|---|---|
| `BacktestingRiskManager` | `risk_management/backtesting_risk_manager.py` | Historical percentile | Backtesting + paper trading |
| `RLRiskManager` | `risk_management/rl_risk_manager.py` | Parametric *or* historical (config flag) | Multi-agent RL environment |

Both inherited from `RiskManagerBase` but duplicated the core math independently — meaning bugs or tuning in one class did not automatically carry over to the other.

## Week 60 Goal

Extract shared computation logic into a single `UnifiedRiskManager` so both existing classes delegate to the same implementation. Preserve full backward compatibility: no public API changes, no test regressions.

## New Class: UnifiedRiskManager

**File:** `risk_management/unified_risk_manager.py`

```
RiskManagerBase (abstract)
├── BacktestingRiskManager  ─── composes ──► UnifiedRiskManager
└── RLRiskManager           ─── composes ──► UnifiedRiskManager
```

`UnifiedRiskManager` is **not** a base class — it is a stateless computation engine used via composition.

### Constructor parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `mode` | `"backtest"` \| `"live"` | `"backtest"` | Execution context (does not change math) |
| `var_method` | `"parametric"` \| `"historical"` | `"historical"` | VaR calculation strategy, independent of `mode` |

### Public methods

| Method | Returns | Notes |
|---|---|---|
| `check_drawdown(peak, current, max_pct)` | `bool` | True = breach |
| `check_trailing_stop(current, reference, buffer, is_long)` | `bool` | True = triggered |
| `compute_var(returns, confidence_level, var_method=None)` | `Optional[float]` | None if < 10 samples |
| `check_correlation(correlation_value, threshold)` | `bool` | True = exceeded (risky) |
| `check_position_limit(position_value, portfolio_value, max_fraction)` | `bool` | True = within limit |

### Thread safety

All methods acquire `self._lock` — a `threading.RLock` (reentrant). This means composing classes that already hold their own lock can call `UnifiedRiskManager` methods without deadlock.

## VaR Design

VaR strategy is **independent of mode**:

```
var_method = "parametric"  →  VaR = -(μ + z_α · σ)
                               where z_α = norm.ppf(1 - CL) < 0
var_method = "historical"  →  VaR = -percentile(returns, (1-CL)×100)
```

Both formulas return a **non-negative loss amount** (clamped at 0).

The key motivation: a backtesting simulation might want parametric VaR for speed, while a live system might want historical VaR for robustness. These are orthogonal concerns.

## Migration Guide

Existing code using `BacktestingRiskManager` or `RLRiskManager` requires **no changes** in Week 60. The classes are still importable and functional.

To use `UnifiedRiskManager` directly in new code:

```python
from risk_management.unified_risk_manager import UnifiedRiskManager

rm = UnifiedRiskManager(mode="live", var_method="historical")

# Drawdown check
breached = rm.check_drawdown(peak_value=10000.0, current_value=8400.0, max_drawdown_pct=0.15)

# VaR
import numpy as np
returns = np.array(...)  # at least 10 observations
var = rm.compute_var(returns, confidence_level=0.95)

# Trailing stop
triggered = rm.check_trailing_stop(
    current_price=90.0,
    reference_price=100.0,   # high-water mark for long
    trailing_stop_buffer=0.05,
    is_long=True,
)
```

## Deprecation Schedule

Both `BacktestingRiskManager` and `RLRiskManager` emit `DeprecationWarning` on instantiation as of Week 60. They are scheduled for **removal in a future phase** (not Week 60 or 61). During Week 61 (DI Refactor), they will be further isolated behind injection points.

## Allowed Divergences

The following behavioral differences between the two classes are **intentional** and documented:

1. **Correlation semantics are inverted:**
   - `BRM.check_correlation_limits(a, b)` → `True` = within limit (safe)
   - `RLRiskManager._check_correlation(a, b)` → `True` = exceeded (risky)
   - Both delegate to `UnifiedRiskManager.check_correlation` which returns `True = exceeded`. The BRM method applies `not` internally.

2. **Stop-loss interface:**
   - BRM: `check_stop_loss(symbol, current_price)` — looks up stored `StopLossConfig`
   - RLRM: `check_stop_loss(agent_id, position_size, entry_price, current_price)` — stateless per-call
   - Different state models; parity at math level only.

3. **VaR with different `var_method`:**
   - RLRM supports `use_parametric_var=True`; BRM is always historical.
   - Numeric results differ by design when methods differ.

## Parity Tests

`tests/test_parity.py` covers:
- Drawdown: BRM, RLRM, UnifiedRiskManager agree for same inputs (S22-A)
- VaR: historical results match across all three (S22-B)
- Trailing stop: UnifiedRiskManager logic matches RLRiskManager math (S22-C)
- Correlation: semantic correctness for both managers (S22-D)
- Position limit: UnifiedRiskManager check (S22-E)
- Thread safety: concurrent access to UnifiedRiskManager (S22-F)
