"""
Week 10 Tests: Enhanced Risk Management

Covers:
- PositionSizer: Kelly fraction (continuous + binary), sizing methods, regime-aware,
  portfolio sizing, leverage cap, win-rate interface, edge cases
- RLRiskManager (Week 10 enhancements): real portfolio VaR, daily loss limit,
  regime position limits, max-correlation constraint, config fields, reset behaviour
- Integration: PositionSizer.from_config(), metrics dict, multi-asset VaR round-trip
"""

import math
import pytest
import numpy as np
import pandas as pd
from collections import deque
from datetime import date
from unittest.mock import patch

from risk_management.position_sizer import PositionSizer, PositionSizerConfig
from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_sizer():
    return PositionSizer()


@pytest.fixture
def half_kelly_sizer():
    return PositionSizer(PositionSizerConfig(method="kelly_half"))


@pytest.fixture
def full_kelly_sizer():
    return PositionSizer(PositionSizerConfig(method="kelly_full"))


@pytest.fixture
def fractional_sizer():
    return PositionSizer(PositionSizerConfig(method="kelly_fractional", kelly_fraction=0.25))


@pytest.fixture
def fixed_sizer():
    return PositionSizer(PositionSizerConfig(method="fixed", max_position_fraction=0.20))


@pytest.fixture
def base_config():
    return RLRiskConfig(
        use_daily_loss_limit=True,
        daily_loss_limit=0.05,
        use_correlation=True,
        correlation_window=20,
        max_correlation=0.8,
        use_portfolio_var=True,
        var_confidence_level=0.95,
    )


@pytest.fixture
def risk_manager(base_config):
    return RLRiskManager(base_config)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – PositionSizer: Kelly fraction (continuous)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyFractionContinuous:

    def test_basic_positive_edge(self, default_sizer):
        # mu=0.10, sigma=0.20 → f* = 0.10/0.04 = 2.5 → clipped to 1.0
        f = default_sizer.kelly_fraction(0.10, 0.20)
        assert 0.0 < f <= 1.0

    def test_known_value(self, default_sizer):
        # mu=0.04, sigma=0.20 → f* = 0.04/0.04 = 1.0
        f = default_sizer.kelly_fraction(0.04, 0.20)
        assert abs(f - 1.0) < 1e-6

    def test_small_edge(self, default_sizer):
        # mu=0.01, sigma=0.20 → f* = 0.01/0.04 = 0.25
        f = default_sizer.kelly_fraction(0.01, 0.20)
        assert abs(f - 0.25) < 1e-6

    def test_zero_volatility_returns_zero(self, default_sizer):
        assert default_sizer.kelly_fraction(0.10, 0.0) == 0.0

    def test_negative_volatility_returns_zero(self, default_sizer):
        assert default_sizer.kelly_fraction(0.10, -0.05) == 0.0

    def test_negative_return_returns_zero(self, default_sizer):
        assert default_sizer.kelly_fraction(-0.05, 0.20) == 0.0

    def test_zero_return_returns_zero(self, default_sizer):
        assert default_sizer.kelly_fraction(0.0, 0.20) == 0.0

    def test_risk_free_rate_reduces_fraction(self, default_sizer):
        f_no_rf = default_sizer.kelly_fraction(0.05, 0.20)
        f_with_rf = default_sizer.kelly_fraction(0.05, 0.20, risk_free_rate=0.02)
        assert f_with_rf < f_no_rf

    def test_higher_return_higher_fraction(self, default_sizer):
        # Use small values so neither clips to 1.0
        f1 = default_sizer.kelly_fraction(0.005, 0.20)  # f* = 0.125
        f2 = default_sizer.kelly_fraction(0.008, 0.20)  # f* = 0.200
        assert f2 > f1

    def test_higher_vol_lower_fraction(self, default_sizer):
        f1 = default_sizer.kelly_fraction(0.05, 0.10)
        f2 = default_sizer.kelly_fraction(0.05, 0.30)
        assert f2 < f1

    def test_clamped_to_one(self, default_sizer):
        # Very large expected return → clipped to 1
        f = default_sizer.kelly_fraction(100.0, 0.01)
        assert f == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – PositionSizer: Kelly fraction (binary)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyFractionBinary:

    def test_basic(self, default_sizer):
        # p=0.6, b=2 → f* = (0.6*2 - 0.4)/2 = 0.8/2 = 0.4
        f = default_sizer.kelly_fraction_binary(0.6, 2.0)
        assert abs(f - 0.4) < 1e-6

    def test_breakeven_returns_zero(self, default_sizer):
        # p=0.5, b=1 → f* = 0
        f = default_sizer.kelly_fraction_binary(0.5, 1.0)
        assert abs(f) < 1e-6

    def test_negative_edge_clamped_zero(self, default_sizer):
        # p=0.3, b=1 → f* < 0, clamped to 0
        f = default_sizer.kelly_fraction_binary(0.3, 1.0)
        assert f == 0.0

    def test_invalid_win_prob_zero(self, default_sizer):
        assert default_sizer.kelly_fraction_binary(0.0, 2.0) == 0.0

    def test_invalid_win_prob_one(self, default_sizer):
        assert default_sizer.kelly_fraction_binary(1.0, 2.0) == 0.0

    def test_invalid_ratio_zero(self, default_sizer):
        assert default_sizer.kelly_fraction_binary(0.6, 0.0) == 0.0

    def test_clamped_to_one(self, default_sizer):
        # Very favourable odds: p=0.99, b=100
        f = default_sizer.kelly_fraction_binary(0.99, 100.0)
        assert f <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 – PositionSizer: size_position methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestSizePosition:

    def test_half_kelly_is_half_full(self, half_kelly_sizer, full_kelly_sizer):
        # mu=0.01, sigma=0.20 → f*=0.25; full=0.25, half=0.125 (below 0.25 cap)
        kwargs = dict(expected_return=0.01, volatility=0.20,
                      portfolio_value=100_000, price=100.0)
        units_half, _ = half_kelly_sizer.size_position(**kwargs)
        units_full, _ = full_kelly_sizer.size_position(**kwargs)
        assert abs(units_half - units_full * 0.5) < 1e-6

    def test_kelly_fractional_0_25(self, fractional_sizer):
        # Use max_position_fraction=1.0 so the cap doesn't interfere
        sizer_full = PositionSizer(PositionSizerConfig(method="kelly_full",
                                                        max_position_fraction=1.0))
        frac_sizer = PositionSizer(PositionSizerConfig(method="kelly_fractional",
                                                        kelly_fraction=0.25,
                                                        max_position_fraction=1.0))
        kwargs = dict(expected_return=0.01, volatility=0.20,
                      portfolio_value=100_000, price=100.0)
        units_full, _ = sizer_full.size_position(**kwargs)
        units_frac, _ = frac_sizer.size_position(**kwargs)
        assert abs(units_frac - units_full * 0.25) < 1e-6

    def test_fixed_method(self, fixed_sizer):
        units, info = fixed_sizer.size_position(
            expected_return=0.05, volatility=0.20,
            portfolio_value=100_000, price=100.0)
        assert info["kelly_scaled"] == pytest.approx(0.20, abs=1e-6)

    def test_fixed_method_no_signal_is_zero(self, fixed_sizer):
        # negative expected return → f_star=0 → fixed returns 0 too
        units, info = fixed_sizer.size_position(
            expected_return=-0.05, volatility=0.20,
            portfolio_value=100_000, price=100.0)
        assert units == 0.0

    def test_confidence_scaling_reduces(self, half_kelly_sizer):
        # mu=0.01, sigma=0.20 → f*=0.25, half=0.125; won't cap, so confidence matters
        u_full, _ = half_kelly_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, confidence=1.0)
        u_half, _ = half_kelly_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, confidence=0.5)
        assert u_half < u_full

    def test_regime_low_vol_no_change(self, default_sizer):
        # Use small mu so f* won't cap before regime multiplier is applied
        u_no_reg, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0)
        u_low, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, regime="low_vol")
        assert abs(u_no_reg - u_low) < 1e-6  # both use multiplier 1.0

    def test_regime_medium_vol_reduces(self, default_sizer):
        # mu=0.01 → half-kelly f_scaled=0.125 < 0.25 cap; regime mult can reduce
        u_low, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, regime="low_vol")
        u_med, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, regime="medium_vol")
        assert u_med < u_low

    def test_regime_high_vol_smallest(self, default_sizer):
        u_med, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, regime="medium_vol")
        u_high, _ = default_sizer.size_position(
            expected_return=0.01, volatility=0.20,
            portfolio_value=100_000, price=100.0, regime="high_vol")
        assert u_high < u_med

    def test_max_position_fraction_clamp(self):
        cfg = PositionSizerConfig(method="kelly_full", max_position_fraction=0.05)
        sizer = PositionSizer(cfg)
        # Very high expected return would exceed cap without clamp
        units, info = sizer.size_position(
            expected_return=50.0, volatility=0.05,
            portfolio_value=100_000, price=100.0)
        assert info["kelly_scaled"] <= 0.05 + 1e-10

    def test_min_threshold_floors_to_zero(self):
        cfg = PositionSizerConfig(method="kelly_half", min_position_fraction=0.20)
        sizer = PositionSizer(cfg)
        # Very small edge won't reach 20%
        units, info = sizer.size_position(
            expected_return=0.001, volatility=0.20,
            portfolio_value=100_000, price=100.0)
        assert units == 0.0

    def test_zero_price_returns_zero_units(self, default_sizer):
        units, info = default_sizer.size_position(
            expected_return=0.05, volatility=0.20,
            portfolio_value=100_000, price=0.0)
        assert units == 0.0

    def test_info_dict_keys(self, default_sizer):
        _, info = default_sizer.size_position(
            expected_return=0.05, volatility=0.20,
            portfolio_value=100_000, price=100.0)
        for key in ("kelly_full", "kelly_scaled", "regime_multiplier",
                    "confidence", "capital_to_invest", "units", "method"):
            assert key in info

    def test_units_match_capital_over_price(self, default_sizer):
        price = 250.0
        units, info = default_sizer.size_position(
            expected_return=0.05, volatility=0.20,
            portfolio_value=100_000, price=price)
        expected_units = info["capital_to_invest"] / price
        assert abs(units - expected_units) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 – PositionSizer: portfolio sizing & leverage cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestSizePortfolio:

    def _signals(self):
        return {
            "A": {"expected_return": 0.10, "volatility": 0.20, "confidence": 1.0},
            "B": {"expected_return": 0.08, "volatility": 0.18, "confidence": 0.9},
        }

    def test_returns_both_assets(self, default_sizer):
        result = default_sizer.size_portfolio(
            signals=self._signals(),
            portfolio_value=100_000,
            prices={"A": 100.0, "B": 50.0},
        )
        assert "A" in result and "B" in result

    def test_skips_zero_price(self, default_sizer):
        signals = self._signals()
        signals["C"] = {"expected_return": 0.05, "volatility": 0.15}
        result = default_sizer.size_portfolio(
            signals=signals,
            portfolio_value=100_000,
            prices={"A": 100.0, "B": 50.0, "C": 0.0},
        )
        assert "C" not in result

    def test_leverage_cap_enforced(self):
        # Use full-Kelly with no max so fractions could exceed 1.0 total
        cfg = PositionSizerConfig(
            method="kelly_full",
            max_position_fraction=1.0,
            max_leverage=1.0,
        )
        sizer = PositionSizer(cfg)
        signals = {f"asset_{i}": {"expected_return": 0.20, "volatility": 0.10}
                   for i in range(5)}
        prices = {f"asset_{i}": 100.0 for i in range(5)}
        result = sizer.size_portfolio(
            signals=signals, portfolio_value=100_000, prices=prices)
        total_invested = sum(info["capital_to_invest"] for _, info in result.values())
        assert total_invested <= 100_000 * 1.0 + 1e-6

    def test_empty_signals(self, default_sizer):
        result = default_sizer.size_portfolio(
            signals={}, portfolio_value=100_000, prices={})
        assert result == {}

    def test_leverage_scale_in_info(self, default_sizer):
        result = default_sizer.size_portfolio(
            signals=self._signals(),
            portfolio_value=100_000,
            prices={"A": 100.0, "B": 50.0},
        )
        for _, info in result.values():
            assert "leverage_scale" in info


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 – PositionSizer: from_win_rate
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromWinRate:

    def test_basic(self, half_kelly_sizer):
        units, info = half_kelly_sizer.from_win_rate(
            win_rate=0.6, avg_win=0.05, avg_loss=0.03,
            portfolio_value=100_000, price=100.0)
        assert units >= 0.0
        assert "win_loss_ratio" in info

    def test_positive_kelly_edge(self, half_kelly_sizer):
        units, info = half_kelly_sizer.from_win_rate(
            win_rate=0.7, avg_win=0.04, avg_loss=0.02,
            portfolio_value=100_000, price=100.0)
        assert units > 0.0

    def test_negative_edge_is_zero(self, half_kelly_sizer):
        # 30% win rate with equal payoff → f* < 0, clamped to 0
        units, info = half_kelly_sizer.from_win_rate(
            win_rate=0.3, avg_win=0.02, avg_loss=0.02,
            portfolio_value=100_000, price=100.0)
        assert units == 0.0

    def test_zero_avg_loss_returns_zero(self, default_sizer):
        units, info = default_sizer.from_win_rate(
            win_rate=0.6, avg_win=0.05, avg_loss=0.0,
            portfolio_value=100_000, price=100.0)
        assert units == 0.0

    def test_regime_reduces_position(self, default_sizer):
        u_low, _ = default_sizer.from_win_rate(
            0.6, 0.05, 0.03, 100_000, 100.0, regime="low_vol")
        u_high, _ = default_sizer.from_win_rate(
            0.6, 0.05, 0.03, 100_000, 100.0, regime="high_vol")
        assert u_high < u_low

    def test_win_loss_ratio_in_info(self, default_sizer):
        _, info = default_sizer.from_win_rate(
            0.6, 0.06, 0.03, 100_000, 100.0)
        assert abs(info["win_loss_ratio"] - 2.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 – PositionSizer: from_config factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromConfig:

    def test_default_method(self):
        sizer = PositionSizer.from_config({})
        assert sizer.config.method == "kelly_half"

    def test_custom_config(self):
        cfg = {
            "kelly": {
                "method": "kelly_fractional",
                "kelly_fraction": 0.3,
                "max_position_fraction": 0.15,
            },
            "regime_position_limits": {
                "low_vol": 1.0, "high_vol": 0.4,
            },
        }
        sizer = PositionSizer.from_config(cfg)
        assert sizer.config.method == "kelly_fractional"
        assert sizer.config.kelly_fraction == pytest.approx(0.3)
        assert sizer.config.max_position_fraction == pytest.approx(0.15)
        assert sizer.config.regime_limits["high_vol"] == pytest.approx(0.4)

    def test_returns_position_sizer_instance(self):
        sizer = PositionSizer.from_config({"kelly": {"method": "fixed"}})
        assert isinstance(sizer, PositionSizer)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 – RLRiskManager: config & new fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestRLRiskConfig:

    def test_default_daily_loss_disabled(self):
        cfg = RLRiskConfig()
        assert cfg.use_daily_loss_limit is False

    def test_daily_loss_limit_default_value(self):
        cfg = RLRiskConfig()
        assert cfg.daily_loss_limit == pytest.approx(0.03)

    def test_max_correlation_default(self):
        cfg = RLRiskConfig()
        assert cfg.max_correlation == pytest.approx(0.8)

    def test_regime_position_limits_present(self):
        cfg = RLRiskConfig()
        assert "low_vol" in cfg.regime_position_limits
        assert "medium_vol" in cfg.regime_position_limits
        assert "high_vol" in cfg.regime_position_limits

    def test_regime_position_limits_values(self):
        cfg = RLRiskConfig()
        assert cfg.regime_position_limits["low_vol"] == pytest.approx(1.0)
        assert cfg.regime_position_limits["high_vol"] == pytest.approx(0.5)

    def test_custom_daily_loss_limit(self):
        cfg = RLRiskConfig(use_daily_loss_limit=True, daily_loss_limit=0.02)
        assert cfg.daily_loss_limit == pytest.approx(0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 – RLRiskManager: daily loss limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestDailyLossLimit:

    def test_disabled_always_false(self):
        mgr = RLRiskManager(RLRiskConfig(use_daily_loss_limit=False))
        mgr.record_daily_start(100_000)
        assert mgr.check_daily_loss_limit(50_000) is False  # huge loss but disabled

    def test_not_exceeded(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        # 2% loss; limit is 5%
        assert risk_manager.check_daily_loss_limit(98_000) is False

    def test_exactly_at_limit_not_exceeded(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        # Exactly 5% loss → NOT exceeded (strict >)
        assert risk_manager.check_daily_loss_limit(95_000) is False

    def test_exceeded(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        # 6% loss; limit is 5%
        assert risk_manager.check_daily_loss_limit(94_000) is True

    def test_no_start_value_returns_false(self, risk_manager):
        # No call to record_daily_start
        assert risk_manager.check_daily_loss_limit(90_000) is False

    def test_new_day_resets_baseline(self, risk_manager):
        day1 = date(2024, 1, 2)
        day2 = date(2024, 1, 3)
        risk_manager.record_daily_start(100_000, day1)
        # "today" is day2 → baseline resets to current value automatically
        triggered = risk_manager.check_daily_loss_limit(95_000, day2)
        assert triggered is False   # new day reset
        assert risk_manager.daily_start_value == pytest.approx(95_000)

    def test_same_day_no_reset(self, risk_manager):
        day = date(2024, 1, 2)
        risk_manager.record_daily_start(100_000, day)
        triggered = risk_manager.check_daily_loss_limit(94_000, day)
        assert triggered is True

    def test_event_counter_increments(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        before = risk_manager.daily_loss_limit_events
        risk_manager.check_daily_loss_limit(80_000)
        assert risk_manager.daily_loss_limit_events == before + 1

    def test_event_counter_not_increments_when_safe(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        before = risk_manager.daily_loss_limit_events
        risk_manager.check_daily_loss_limit(99_000)
        assert risk_manager.daily_loss_limit_events == before

    def test_reset_clears_daily_state(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        risk_manager.reset()
        assert risk_manager.daily_start_value is None
        assert risk_manager.daily_loss_limit_events == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 – RLRiskManager: regime-aware position limits
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePositionLimits:

    def test_low_vol_no_change(self, risk_manager):
        result = risk_manager.apply_regime_position_limit(0.20, "low_vol")
        assert result == pytest.approx(0.20)

    def test_medium_vol_reduces(self, risk_manager):
        result = risk_manager.apply_regime_position_limit(0.20, "medium_vol")
        assert result == pytest.approx(0.20 * 0.75)

    def test_high_vol_reduces_most(self, risk_manager):
        result = risk_manager.apply_regime_position_limit(0.20, "high_vol")
        assert result == pytest.approx(0.20 * 0.5)

    def test_none_regime_no_change(self, risk_manager):
        result = risk_manager.apply_regime_position_limit(0.20, None)
        assert result == pytest.approx(0.20)

    def test_unknown_regime_no_change(self, risk_manager):
        # Unknown key → defaults to multiplier 1.0
        result = risk_manager.apply_regime_position_limit(0.20, "unknown_regime")
        assert result == pytest.approx(0.20)

    def test_custom_regime_limits(self):
        cfg = RLRiskConfig(regime_position_limits={"low_vol": 0.9, "high_vol": 0.1})
        mgr = RLRiskManager(cfg)
        assert mgr.apply_regime_position_limit(0.20, "low_vol") == pytest.approx(0.20 * 0.9)
        assert mgr.apply_regime_position_limit(0.20, "high_vol") == pytest.approx(0.20 * 0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10 – RLRiskManager: max-correlation constraint
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxCorrelationConstraint:

    def _inject_correlation_matrix(self, mgr, assets, matrix_values):
        """Helper: directly set a correlation matrix on the manager."""
        mgr.correlation_matrix = pd.DataFrame(
            matrix_values, index=assets, columns=assets
        )

    def test_no_matrix_returns_false(self, risk_manager):
        assert risk_manager.check_max_correlation_constraint(
            "A", {"B": 1.0}) is False

    def test_below_threshold_returns_false(self, risk_manager):
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.5], [0.5, 1.0]])
        assert risk_manager.check_max_correlation_constraint("A", {"B": 1.0}) is False

    def test_at_threshold_returns_false(self, risk_manager):
        # Strictly >, so exactly at 0.8 is False
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.8], [0.8, 1.0]])
        assert risk_manager.check_max_correlation_constraint("A", {"B": 1.0}) is False

    def test_above_threshold_returns_true(self, risk_manager):
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.9], [0.9, 1.0]])
        assert risk_manager.check_max_correlation_constraint("A", {"B": 1.0}) is True

    def test_ignores_zero_positions(self, risk_manager):
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.95], [0.95, 1.0]])
        # B has zero position → should not trigger
        assert risk_manager.check_max_correlation_constraint("A", {"B": 0.0}) is False

    def test_ignores_self(self, risk_manager):
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.95], [0.95, 1.0]])
        # Checking A against A itself → skip
        assert risk_manager.check_max_correlation_constraint("A", {"A": 1.0}) is False

    def test_negative_correlation_uses_abs(self, risk_manager):
        # -0.9 absolute value > 0.8 threshold
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, -0.9], [-0.9, 1.0]])
        assert risk_manager.check_max_correlation_constraint("A", {"B": 1.0}) is True

    def test_asset_not_in_matrix(self, risk_manager):
        assets = ["A", "B"]
        self._inject_correlation_matrix(risk_manager, assets, [[1.0, 0.5], [0.5, 1.0]])
        # "C" not in matrix → False
        assert risk_manager.check_max_correlation_constraint("C", {"A": 1.0}) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11 – RLRiskManager: real portfolio VaR
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioVaR:

    def _make_manager_with_history(self, assets, n=50, seed=42):
        """Build an RLRiskManager with synthetic return history + covariance."""
        rng = np.random.default_rng(seed)
        cfg = RLRiskConfig(
            use_portfolio_var=True,
            use_correlation=True,
            var_confidence_level=0.95,
        )
        mgr = RLRiskManager(cfg)

        # Populate return histories
        for a in assets:
            returns = rng.normal(0.001, 0.02, n)
            mgr.asset_returns_history[a] = deque(returns, maxlen=100)

        # Build covariance matrix
        data = {a: list(mgr.asset_returns_history[a]) for a in assets}
        df = pd.DataFrame(data)
        mgr.covariance_matrix = df.cov()
        return mgr

    def test_empty_positions_returns_zero(self):
        mgr = self._make_manager_with_history(["A", "B"])
        var = mgr.calculate_portfolio_var({}, {"A": 100.0, "B": 200.0})
        assert var == 0.0

    def test_single_asset_uses_historical(self):
        mgr = self._make_manager_with_history(["A", "B"])
        # Only A has a position
        var = mgr.calculate_portfolio_var({"A": 10.0, "B": 0.0}, {"A": 100.0, "B": 200.0})
        assert var is not None
        assert var >= 0.0

    def test_multi_asset_returns_positive(self):
        mgr = self._make_manager_with_history(["A", "B"])
        var = mgr.calculate_portfolio_var(
            {"A": 5.0, "B": 10.0}, {"A": 100.0, "B": 200.0})
        assert var is not None
        assert var >= 0.0

    def test_no_cov_matrix_returns_default(self):
        cfg = RLRiskConfig(use_portfolio_var=True, var_confidence_level=0.95)
        mgr = RLRiskManager(cfg)
        var = mgr.calculate_portfolio_var(
            {"A": 5.0, "B": 10.0}, {"A": 100.0, "B": 200.0})
        assert var == pytest.approx(0.02)

    def test_result_is_non_negative(self):
        mgr = self._make_manager_with_history(["A", "B", "C"])
        var = mgr.calculate_portfolio_var(
            {"A": 3.0, "B": 3.0, "C": 3.0},
            {"A": 100.0, "B": 150.0, "C": 200.0})
        assert var >= 0.0

    def test_diversified_portfolio_lower_var(self):
        """Negatively correlated assets should yield lower portfolio VaR."""
        rng = np.random.default_rng(0)
        n = 200
        base = rng.normal(0, 0.02, n)

        cfg = RLRiskConfig(use_portfolio_var=True, use_correlation=True, var_confidence_level=0.95)
        mgr = RLRiskManager(cfg)

        # Asset A and B are perfectly negatively correlated
        returns_a = base
        returns_b = -base  # perfect hedge

        mgr.asset_returns_history["A"] = deque(returns_a, maxlen=n + 10)
        mgr.asset_returns_history["B"] = deque(returns_b, maxlen=n + 10)

        df = pd.DataFrame({"A": returns_a, "B": returns_b})
        mgr.covariance_matrix = df.cov()

        var_hedged = mgr.calculate_portfolio_var(
            {"A": 5.0, "B": 5.0}, {"A": 100.0, "B": 100.0})
        var_single = mgr.calculate_portfolio_var(
            {"A": 10.0, "B": 0.0}, {"A": 100.0, "B": 100.0})

        # Hedged portfolio should have lower or similar VaR
        assert var_hedged is not None
        assert var_single is not None
        assert var_hedged <= var_single + 1e-10  # allow floating-point tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12 – RLRiskManager: get_risk_metrics includes Week 10 counter
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskMetrics:

    def test_daily_loss_limit_events_in_metrics(self, risk_manager):
        metrics = risk_manager.get_risk_metrics()
        assert "daily_loss_limit_events" in metrics

    def test_metrics_initial_zero(self, risk_manager):
        metrics = risk_manager.get_risk_metrics()
        assert metrics["daily_loss_limit_events"] == 0

    def test_metrics_after_breach(self, risk_manager):
        risk_manager.record_daily_start(100_000)
        risk_manager.check_daily_loss_limit(80_000)   # triggers
        metrics = risk_manager.get_risk_metrics()
        assert metrics["daily_loss_limit_events"] == 1

    def test_existing_metrics_still_present(self, risk_manager):
        metrics = risk_manager.get_risk_metrics()
        for key in ("stop_loss_events", "var_exceed_events",
                    "portfolio_stop_loss_events"):
            assert key in metrics
