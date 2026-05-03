"""Phase 8-Alpha: cost_model plumbing tests.

Verifies:
- SingleAssetRLTradingEnv accepts cost_model kwarg
- Default is "spot_taker"
- "futures_maker" is accepted
- Invalid value raises ValueError
- config/schema.py EnvConfig validates the field
"""

import pytest
import numpy as np
import pandas as pd

from envs.single_asset_rl_env import SingleAssetRLTradingEnv
from config.schema import EnvConfig


# ---------------------------------------------------------------------------
# Minimal data fixture
# ---------------------------------------------------------------------------

def _make_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(0)
    price = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    price = np.maximum(price, 1.0)
    return pd.DataFrame({
        "$open": price,
        "$high": price * 1.001,
        "$low": price * 0.999,
        "$close": price,
        "$volume": np.ones(n) * 1000.0,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_cost_model():
    env = SingleAssetRLTradingEnv(data=_make_data(), window_size=10)
    assert env.cost_model == "spot_taker"


def test_spot_taker_accepted():
    env = SingleAssetRLTradingEnv(data=_make_data(), window_size=10, cost_model="spot_taker")
    assert env.cost_model == "spot_taker"


def test_futures_maker_accepted():
    env = SingleAssetRLTradingEnv(data=_make_data(), window_size=10, cost_model="futures_maker")
    assert env.cost_model == "futures_maker"


def test_invalid_cost_model_raises():
    with pytest.raises(ValueError, match="cost_model must be one of"):
        SingleAssetRLTradingEnv(data=_make_data(), window_size=10, cost_model="invalid_model")


def test_futures_maker_overrides_apply_slippage():
    """futures_maker forces apply_slippage=False even when True is passed."""
    env = SingleAssetRLTradingEnv(
        data=_make_data(), window_size=10, cost_model="futures_maker", apply_slippage=True
    )
    assert env.apply_slippage is False


def test_spot_taker_preserves_apply_slippage():
    env = SingleAssetRLTradingEnv(
        data=_make_data(), window_size=10, cost_model="spot_taker", apply_slippage=True
    )
    assert env.apply_slippage is True


def test_schema_cost_model_default():
    cfg = EnvConfig()
    assert cfg.cost_model == "spot_taker"


def test_schema_futures_maker_valid():
    cfg = EnvConfig(cost_model="futures_maker")
    assert cfg.cost_model == "futures_maker"


def test_schema_invalid_cost_model_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EnvConfig(cost_model="bad_value")
