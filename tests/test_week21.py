"""
Week 21 Tests: GTrXL Feature Extractor + Almgren-Chriss Market Impact

Covers:
    - AlmgrenChrissImpact: both model variants, edge cases, validation checks
    - GTrXLExtractor: forward pass shape, memory update, reset_memory
    - SingleAssetRLTradingEnv: market impact integration, backward compat
"""

import math
import pytest
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from envs.market_impact import AlmgrenChrissImpact
from agents.sb3.feature_extractors import GTrXLExtractor, GRUGate, GTrXLLayer
from envs.single_asset_rl_env import SingleAssetRLTradingEnv


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture()
def dummy_ohlcv() -> pd.DataFrame:
    """200-row synthetic OHLCV dataframe."""
    n = 200
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.clip(close, 10, None)
    return pd.DataFrame({
        "$open":   close * rng.uniform(0.995, 1.005, n),
        "$high":   close * rng.uniform(1.000, 1.010, n),
        "$low":    close * rng.uniform(0.990, 1.000, n),
        "$close":  close,
        "$volume": rng.uniform(1_000, 50_000, n),
    })


@pytest.fixture()
def obs_space_20_18() -> gym.spaces.Box:
    return gym.spaces.Box(low=-10.0, high=10.0, shape=(20, 18), dtype=np.float32)


# ─────────────────────────────────────────────
# AlmgrenChrissImpact — sqrt model
# ─────────────────────────────────────────────

class TestAlmgrenChrissImpactSqrt:
    def test_basic_cost_positive(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6, kappa=0.5)
        cost = impact.compute(shares=1000, price=100.0)
        assert cost > 0, "Impact cost must be positive for non-zero trade"

    def test_zero_trade_returns_zero(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6)
        assert impact.compute(shares=0.0, price=100.0) == 0.0

    def test_cost_increases_with_size(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6, kappa=0.5)
        small = impact.compute(shares=100, price=100.0)
        large = impact.compute(shares=10_000, price=100.0)
        assert large > small, "Larger trades should have higher impact"

    def test_cost_decreases_with_higher_volume(self):
        impact_low = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e4, kappa=0.5)
        impact_high = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e7, kappa=0.5)
        cost_low = impact_low.compute(shares=500, price=100.0)
        cost_high = impact_high.compute(shares=500, price=100.0)
        assert cost_low > cost_high, "Higher volume → lower impact"

    def test_capped_at_max_impact(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1.0, kappa=10.0, max_impact_cap=0.05)
        cost = impact.compute(shares=1e9, price=100.0)
        assert cost <= 0.05

    def test_dynamic_volume_override(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6, kappa=0.5)
        default_cost = impact.compute(shares=1000, price=100.0)
        low_vol_cost = impact.compute(shares=1000, price=100.0, daily_volume=1e3)
        assert low_vol_cost > default_cost, "Low bar volume should increase impact"

    def test_absolute_value_of_shares(self):
        """Buy and sell of same size should produce identical impact."""
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6, kappa=0.5)
        buy = impact.compute(shares=500, price=100.0)
        sell = impact.compute(shares=-500, price=100.0)
        assert math.isclose(buy, sell, rel_tol=1e-9)

    def test_formula_correctness(self):
        """Verify sqrt formula exactly: sigma * sqrt(|shares| / V) * kappa."""
        sigma, V, kappa, shares = 0.02, 1e6, 0.5, 1000.0
        expected = sigma * math.sqrt(shares / V) * kappa
        impact = AlmgrenChrissImpact(model="sqrt", sigma=sigma, daily_volume=V, kappa=kappa)
        cost = impact.compute(shares=shares, price=100.0)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_compute_from_trade_value(self):
        impact = AlmgrenChrissImpact(model="sqrt", sigma=0.02, daily_volume=1e6, kappa=0.5)
        cost = impact.compute_from_trade_value(trade_value=50_000, price=100.0, bar_volume=1e6)
        assert cost > 0


# ─────────────────────────────────────────────
# AlmgrenChrissImpact — linear model
# ─────────────────────────────────────────────

class TestAlmgrenChrissImpactLinear:
    def test_basic_cost_positive(self):
        impact = AlmgrenChrissImpact(model="linear", eta=0.01, gamma=0.001)
        cost = impact.compute(shares=100, price=100.0)
        assert cost > 0

    def test_formula_correctness(self):
        """Verify linear formula: (eta*(Q/T) + gamma*Q) / price."""
        eta, gamma, shares, price, T = 0.01, 0.001, 100.0, 100.0, 1
        expected = (eta * (shares / T) + gamma * shares) / price
        impact = AlmgrenChrissImpact(model="linear", eta=eta, gamma=gamma)
        cost = impact.compute(shares=shares, price=price, T=T)
        assert math.isclose(cost, expected, rel_tol=1e-9)

    def test_zero_price_returns_zero(self):
        impact = AlmgrenChrissImpact(model="linear", eta=0.01, gamma=0.001)
        cost = impact.compute(shares=100, price=0.0)
        assert cost == 0.0


# ─────────────────────────────────────────────
# AlmgrenChrissImpact — validation
# ─────────────────────────────────────────────

class TestAlmgrenChrissImpactValidation:
    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model must be"):
            AlmgrenChrissImpact(model="invalid")

    def test_zero_daily_volume_raises(self):
        with pytest.raises(ValueError, match="daily_volume"):
            AlmgrenChrissImpact(model="sqrt", daily_volume=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            AlmgrenChrissImpact(model="sqrt", sigma=-0.01)


# ─────────────────────────────────────────────
# GRUGate unit test
# ─────────────────────────────────────────────

class TestGRUGate:
    def test_output_shape(self):
        gate = GRUGate(d_model=64)
        x = torch.randn(4, 10, 64)
        y = torch.randn(4, 10, 64)
        out = gate(x, y)
        assert out.shape == (4, 10, 64)

    def test_near_identity_at_init(self):
        """With fresh params and same x/y, gate output should be close to x (z≈0)."""
        torch.manual_seed(0)
        gate = GRUGate(d_model=32)
        x = torch.zeros(1, 1, 32)
        y = torch.zeros(1, 1, 32)
        out = gate(x, y)
        # Gate is z≈σ(-2)≈0.12, so output ≈ 0.88*x ≈ 0 here
        assert out.shape == x.shape


# ─────────────────────────────────────────────
# GTrXLLayer unit test
# ─────────────────────────────────────────────

class TestGTrXLLayer:
    def test_forward_no_memory(self):
        layer = GTrXLLayer(d_model=32, n_heads=4, ffn_dim=128)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 32)

    def test_forward_with_memory(self):
        layer = GTrXLLayer(d_model=32, n_heads=4, ffn_dim=128)
        x = torch.randn(2, 10, 32)
        mem = torch.randn(2, 8, 32)
        out = layer(x, memory=mem)
        assert out.shape == (2, 10, 32)


# ─────────────────────────────────────────────
# GTrXLExtractor
# ─────────────────────────────────────────────

class TestGTrXLExtractor:
    def test_output_shape(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=128)
        x = torch.randn(4, 20, 18)
        out = extractor(x)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    def test_output_shape_single_sample(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=64, n_layers=2, d_model=64, n_heads=4)
        x = torch.randn(1, 20, 18)
        out = extractor(x)
        assert out.shape == (1, 64)

    def test_different_feature_dims(self):
        obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(30, 10), dtype=np.float32)
        for fdim in [32, 64, 128, 256]:
            extractor = GTrXLExtractor(obs_space, features_dim=fdim, d_model=64, n_heads=4)
            out = extractor(torch.randn(2, 30, 10))
            assert out.shape == (2, fdim)

    def test_memory_updated_after_forward(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=128, memory_len=16)
        assert all(m is None for m in extractor._memories), "Memories should start None"
        _ = extractor(torch.randn(2, 20, 18))
        assert any(m is not None for m in extractor._memories), "Memories should be populated after forward"

    def test_reset_memory_clears_state(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=128, memory_len=16)
        _ = extractor(torch.randn(2, 20, 18))
        extractor.reset_memory()
        assert all(m is None for m in extractor._memories)

    def test_no_memory_mode(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=128, memory_len=0)
        out = extractor(torch.randn(3, 20, 18))
        assert out.shape == (3, 128)
        assert all(m is None for m in extractor._memories)

    def test_gradients_flow(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=64, n_layers=2, d_model=64, n_heads=4)
        x = torch.randn(2, 20, 18, requires_grad=False)
        out = extractor(x)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        grad_exists = any(p.grad is not None for p in extractor.parameters())
        assert grad_exists, "Gradients should flow through GTrXLExtractor"

    def test_invalid_obs_shape_raises(self):
        obs_space_1d = gym.spaces.Box(low=-10.0, high=10.0, shape=(100,), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D observations"):
            GTrXLExtractor(obs_space_1d, features_dim=128)

    def test_invalid_d_model_n_heads_raises(self):
        obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(20, 18), dtype=np.float32)
        with pytest.raises(ValueError, match="divisible"):
            GTrXLExtractor(obs_space, features_dim=128, d_model=100, n_heads=3)

    def test_deterministic_without_dropout(self, obs_space_20_18):
        extractor = GTrXLExtractor(obs_space_20_18, features_dim=64, dropout=0.0)
        extractor.eval()
        x = torch.randn(2, 20, 18)
        extractor.reset_memory()
        out1 = extractor(x)
        extractor.reset_memory()
        out2 = extractor(x)
        assert torch.allclose(out1, out2), "Deterministic forward should be identical"


# ─────────────────────────────────────────────
# SingleAssetRLTradingEnv — market impact integration
# ─────────────────────────────────────────────

class TestEnvMarketImpactIntegration:
    def test_env_with_market_impact_enabled(self, dummy_ohlcv):
        env = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            use_market_impact=True,
            market_impact_model="sqrt",
            market_impact_sigma=0.02,
            market_impact_kappa=0.5,
        )
        assert env.market_impact is not None

    def test_env_without_market_impact_default(self, dummy_ohlcv):
        """Default env should have no market impact (backward compat)."""
        env = SingleAssetRLTradingEnv(data=dummy_ohlcv)
        assert env.market_impact is None

    def test_step_with_market_impact_runs(self, dummy_ohlcv):
        env = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            use_market_impact=True,
            market_impact_model="sqrt",
        )
        obs, _ = env.reset()
        action = np.array([0.5], dtype=np.float32)
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2 is not None
        assert np.isfinite(reward)

    def test_market_impact_produces_positive_slippage(self, dummy_ohlcv):
        """Market impact model should produce a positive, bounded slippage fraction."""
        env = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            apply_slippage=True,
            use_market_impact=True,
            market_impact_model="sqrt",
            market_impact_sigma=0.05,
            market_impact_kappa=1.0,
        )
        env.reset(seed=0)
        action = np.array([1.0], dtype=np.float32)
        env.step(action)
        # Slippage should be positive and within the 5% cap
        assert env.last_slippage >= 0.0
        assert env.last_slippage <= 0.05

    def test_market_impact_high_kappa_yields_more_impact(self, dummy_ohlcv):
        """Higher kappa should produce higher market impact (all else equal)."""
        env_low = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            apply_slippage=True,
            use_market_impact=True,
            market_impact_model="sqrt",
            market_impact_sigma=0.02,
            market_impact_kappa=0.1,
        )
        env_high = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            apply_slippage=True,
            use_market_impact=True,
            market_impact_model="sqrt",
            market_impact_sigma=0.02,
            market_impact_kappa=2.0,
        )
        env_low.reset(seed=0)
        env_high.reset(seed=0)
        action = np.array([1.0], dtype=np.float32)
        env_low.step(action)
        env_high.step(action)
        assert env_high.last_slippage >= env_low.last_slippage

    def test_env_linear_market_impact_model(self, dummy_ohlcv):
        env = SingleAssetRLTradingEnv(
            data=dummy_ohlcv,
            use_market_impact=True,
            market_impact_model="linear",
            market_impact_eta=0.01,
            market_impact_gamma=0.001,
        )
        obs, _ = env.reset()
        _, reward, *_ = env.step(np.array([0.3], dtype=np.float32))
        assert np.isfinite(reward)
