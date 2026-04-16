"""
Week 60 — S22: Parity tests for RiskManager unification.

Verifies that BacktestingRiskManager and RLRiskManager produce identical
risk decisions when given equivalent inputs and the same var_method.

Allowed divergences (documented per test):
- VaR with different var_method settings (parametric vs historical)
- check_stop_loss: different call signatures and state mechanisms
- Correlation handling: BRM.check_correlation_limits returns True=within-limit,
  RLRiskManager._check_correlation returns True=exceeded — semantics are inverted
  but logically equivalent.
"""

import threading
import numpy as np
import pytest

from risk_management.backtesting_risk_manager import BacktestingRiskManager, BacktestingRiskConfig
from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
from risk_management.unified_risk_manager import UnifiedRiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brm():
    config = BacktestingRiskConfig(
        max_drawdown_pct=0.15,
        use_var=True,
        var_confidence_level=0.95,
        max_correlation=0.7,
    )
    return BacktestingRiskManager(config)


@pytest.fixture
def rlrm():
    config = RLRiskConfig(
        max_drawdown_pct=0.15,
        use_var=True,
        var_confidence_level=0.95,
        use_parametric_var=False,  # historical — matches BRM default
        correlation_threshold=0.7,
        use_correlation=True,
    )
    return RLRiskManager(config)


@pytest.fixture
def unified_hist():
    return UnifiedRiskManager(mode="backtest", var_method="historical")


@pytest.fixture
def unified_param():
    return UnifiedRiskManager(mode="live", var_method="parametric")


# ---------------------------------------------------------------------------
# S22-A: check_drawdown parity
# ---------------------------------------------------------------------------

class TestDrawdownParity:
    """Both managers must agree on drawdown breach with identical inputs."""

    @pytest.mark.parametrize("peak,current,expected_breach", [
        (1000.0, 850.0, False),  # exactly 15% drawdown — strict > 0.15 → False (boundary)
        (1000.0, 849.0, True),   # 15.1% → breach
        (1000.0, 860.0, False),  # 14% → ok
        (1000.0, 1000.0, False), # 0% → ok
        (1000.0, 1100.0, False), # profit → ok
    ])
    def test_brm_vs_rlrm(self, brm, rlrm, peak, current, expected_breach):
        brm_result = brm.check_drawdown(peak, current)
        # RLRiskManager pattern 1: (peak_float, current_float)
        rl_result = rlrm.check_drawdown(peak, current)
        assert brm_result == rl_result, (
            f"Drawdown check diverged: BRM={brm_result}, RL={rl_result} "
            f"(peak={peak}, current={current})"
        )
        assert brm_result == expected_breach

    def test_unified_agrees_with_both(self, brm, rlrm, unified_hist):
        peak, current = 1000.0, 840.0  # 16% → breach
        threshold = 0.15
        unified_result = unified_hist.check_drawdown(peak, current, threshold)
        brm_result = brm.check_drawdown(peak, current)
        rl_result = rlrm.check_drawdown(peak, current)
        assert unified_result == brm_result == rl_result == True

    def test_zero_peak_safe(self, brm, rlrm, unified_hist):
        """All three must return False (not breach) when peak <= 0."""
        assert brm.check_drawdown(0.0, 100.0) == False
        assert rlrm.check_drawdown(0.0, 100.0) == False
        assert unified_hist.check_drawdown(0.0, 100.0, 0.15) == False


# ---------------------------------------------------------------------------
# S22-B: compute_var parity (same var_method → same result)
# ---------------------------------------------------------------------------

class TestVaRParity:
    """Historical VaR must match across UnifiedRiskManager, BRM, and RL (historical mode)."""

    @pytest.fixture
    def returns_30(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 0.01, 30)

    def test_historical_var_brm_vs_unified(self, brm, unified_hist, returns_30):
        brm_var = brm.compute_var(returns_30, confidence_level=0.95)
        uni_var = unified_hist.compute_var(returns_30, confidence_level=0.95)
        assert uni_var is not None
        assert abs(brm_var - uni_var) < 1e-10, (
            f"Historical VaR mismatch: BRM={brm_var}, Unified={uni_var}"
        )

    def test_historical_var_rlrm_vs_unified(self, rlrm, unified_hist, returns_30):
        rl_var = rlrm.compute_var(returns_30)
        uni_var = unified_hist.compute_var(returns_30, confidence_level=0.95)
        assert rl_var is not None
        assert uni_var is not None
        assert abs(rl_var - uni_var) < 1e-10, (
            f"Historical VaR mismatch: RL={rl_var}, Unified={uni_var}"
        )

    def test_all_three_agree_historical(self, brm, rlrm, unified_hist, returns_30):
        brm_var = brm.compute_var(returns_30, 0.95)
        rl_var = rlrm.compute_var(returns_30)
        uni_var = unified_hist.compute_var(returns_30, 0.95)
        assert rl_var is not None and uni_var is not None
        assert abs(brm_var - rl_var) < 1e-10
        assert abs(brm_var - uni_var) < 1e-10

    def test_insufficient_data_returns_none_or_zero(self, brm, rlrm, unified_hist):
        short_returns = np.array([0.01, -0.02, 0.03])
        # BRM returns 0.0 for len < 2 ... actually len=3 is >= 2, it just computes
        # RL returns None for len < 10
        rl_var = rlrm.compute_var(short_returns)
        uni_var = unified_hist.compute_var(short_returns)
        assert rl_var is None
        assert uni_var is None

    def test_parametric_var_unified(self, unified_param):
        rng = np.random.default_rng(123)
        returns = rng.normal(0, 0.01, 50)
        var = unified_param.compute_var(returns, 0.95)
        assert var is not None
        assert var >= 0.0

    def test_parametric_var_rlrm(self):
        config = RLRiskConfig(
            use_var=True,
            var_confidence_level=0.95,
            use_parametric_var=True,
        )
        rm = RLRiskManager(config)
        rng = np.random.default_rng(99)
        returns = rng.normal(0, 0.01, 50)
        # Both should give same result
        rl_var = rm.compute_var(returns)
        uni_var = UnifiedRiskManager(mode="live", var_method="parametric").compute_var(returns, 0.95)
        assert rl_var is not None and uni_var is not None
        assert abs(rl_var - uni_var) < 1e-10, (
            f"Parametric VaR mismatch: RL={rl_var}, Unified={uni_var}"
        )

    def test_var_always_nonnegative(self, unified_hist, unified_param):
        rng = np.random.default_rng(7)
        # Strongly positive returns — VaR should still be >= 0
        returns = rng.normal(0.05, 0.001, 30)
        assert unified_hist.compute_var(returns, 0.95) >= 0.0
        assert unified_param.compute_var(returns, 0.95) >= 0.0


# ---------------------------------------------------------------------------
# S22-C: check_trailing_stop parity
# ---------------------------------------------------------------------------

class TestTrailingStopParity:
    """UnifiedRiskManager trailing stop must match the logic in both managers."""

    @pytest.mark.parametrize("current,reference,buffer,is_long,expected", [
        (90.0, 100.0, 0.05, True, True),   # 10% drop > 5% buffer → triggered
        (96.0, 100.0, 0.05, True, False),  # 4% drop < 5% buffer → not triggered
        (100.0, 90.0, 0.05, False, True),  # 11% rise > 5% → triggered (short)
        (94.0, 90.0, 0.05, False, False),  # 4.4% rise < 5% → not triggered (short)
        (100.0, 100.0, 0.05, True, False), # no move
        (0.0, 100.0, 0.05, True, True),    # full loss
    ])
    def test_unified_trailing_stop(self, unified_hist, current, reference, buffer, is_long, expected):
        result = unified_hist.check_trailing_stop(current, reference, buffer, is_long)
        assert result == expected, (
            f"TrailingStop: current={current}, ref={reference}, buffer={buffer}, "
            f"long={is_long} → got {result}, expected {expected}"
        )

    def test_zero_reference_price_safe(self, unified_hist):
        """Should not raise; returns False when reference_price <= 0."""
        assert unified_hist.check_trailing_stop(50.0, 0.0, 0.05) == False

    def test_rlrm_trailing_stop_uses_same_math(self):
        """RLRiskManager.check_trailing_stop long/short logic must match UnifiedRiskManager."""
        config = RLRiskConfig(use_trailing_stop=True, trailing_stop_buffer=0.05)
        rm = RLRiskManager(config)
        unified = UnifiedRiskManager()

        # Seed state: position high-water mark at 100
        rm.position_highest_values["agent1_BTC"] = 100.0

        # Price dropped 10% — should trigger
        triggered_rl = rm.check_trailing_stop("agent1", "BTC", position_size=1.0, current_price=90.0)
        triggered_uni = unified.check_trailing_stop(90.0, 100.0, 0.05, is_long=True)
        assert triggered_rl == triggered_uni == True


# ---------------------------------------------------------------------------
# S22-D: check_correlation parity
# ---------------------------------------------------------------------------

class TestCorrelationParity:
    """Correlation checks must produce consistent results."""

    @pytest.mark.parametrize("corr,threshold,unified_exceeded,brm_within", [
        (0.8, 0.7, True, False),   # high correlation
        (0.6, 0.7, False, True),   # within limits
        (-0.8, 0.7, True, False),  # high negative correlation also exceeds
        (0.7, 0.7, False, True),   # exactly at threshold → not exceeded (strict >)
    ])
    def test_unified_vs_brm_semantics(self, brm, unified_hist, corr, threshold, unified_exceeded, brm_within):
        # UnifiedRiskManager: True = exceeded (risky)
        uni_result = unified_hist.check_correlation(corr, threshold)
        assert uni_result == unified_exceeded

        # BRM.check_correlation_limits: True = within limit (safe)
        # We test the semantics directly
        # BRM uses self.config.max_correlation; patch _correlation_matrix manually
        import pandas as pd
        brm._correlation_matrix = pd.DataFrame(
            {"A": {"A": 1.0, "B": corr}, "B": {"A": corr, "B": 1.0}}
        )
        brm.config.max_correlation = threshold
        brm_result = brm.check_correlation_limits("A", "B")
        assert brm_result == brm_within, (
            f"BRM correlation_limits(corr={corr}, threshold={threshold}): "
            f"expected {brm_within}, got {brm_result}"
        )

    def test_unified_correlation_no_matrix(self, brm, unified_hist):
        """BRM returns True (safe) when no matrix; UnifiedRiskManager uses explicit value."""
        brm._correlation_matrix = None
        assert brm.check_correlation_limits("A", "B") == True  # safe default
        # UnifiedRiskManager always uses the passed value
        assert unified_hist.check_correlation(0.9, 0.7) == True


# ---------------------------------------------------------------------------
# S22-E: check_position_limit
# ---------------------------------------------------------------------------

class TestPositionLimitParity:
    """UnifiedRiskManager.check_position_limit matches BRM internal logic."""

    @pytest.mark.parametrize("pos_val,port_val,max_frac,expected", [
        (200.0, 1000.0, 0.2, True),   # exactly at limit → within (<=)
        (201.0, 1000.0, 0.2, False),  # 20.1% → exceeds
        (100.0, 1000.0, 0.2, True),   # 10% → well within
        (0.0, 1000.0, 0.2, True),     # zero position
        (100.0, 0.0, 0.2, False),     # zero portfolio → False (guard)
    ])
    def test_unified_position_limit(self, unified_hist, pos_val, port_val, max_frac, expected):
        result = unified_hist.check_position_limit(pos_val, port_val, max_frac)
        assert result == expected


# ---------------------------------------------------------------------------
# S22-F: Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """UnifiedRiskManager must be safe to call from multiple threads simultaneously."""

    def test_concurrent_compute_var(self, unified_hist):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 100)
        results = []
        errors = []

        def worker():
            try:
                v = unified_hist.compute_var(returns, 0.95)
                results.append(v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in concurrent VaR: {errors}"
        assert len(results) == 20
        # All threads must see identical result
        assert all(abs(r - results[0]) < 1e-10 for r in results if r is not None)

    def test_concurrent_check_drawdown(self, unified_hist):
        results = []
        errors = []

        def worker():
            try:
                r = unified_hist.check_drawdown(1000.0, 840.0, 0.15)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == True for r in results)


# ---------------------------------------------------------------------------
# E10: 100 random scenario parity
# ---------------------------------------------------------------------------

class TestRandomScenarioParity:
    """E10: 100 random scenarios — all three managers must agree."""

    def test_drawdown_parity_100_random(self):
        rng = np.random.default_rng(2026_04_16)
        config_b = BacktestingRiskConfig(max_drawdown_pct=0.15)
        config_r = RLRiskConfig(max_drawdown_pct=0.15, use_parametric_var=False)
        brm = BacktestingRiskManager(config_b)
        rlrm = RLRiskManager(config_r)
        unified = UnifiedRiskManager(mode="backtest", var_method="historical")

        peaks = rng.uniform(100.0, 10000.0, 100)
        currents = peaks * rng.uniform(0.5, 1.2, 100)

        for i, (peak, current) in enumerate(zip(peaks, currents)):
            b = brm.check_drawdown(peak, current)
            r = rlrm.check_drawdown(peak, current)
            u = unified.check_drawdown(peak, current, 0.15)
            assert b == r == u, (
                f"Scenario {i}: peak={peak:.2f}, current={current:.2f} → "
                f"BRM={b}, RL={r}, Unified={u}"
            )

    def test_var_parity_100_random(self):
        rng = np.random.default_rng(2026_04_16)
        config_b = BacktestingRiskConfig(use_var=True, var_confidence_level=0.95)
        config_r = RLRiskConfig(use_var=True, var_confidence_level=0.95, use_parametric_var=False)
        brm = BacktestingRiskManager(config_b)
        rlrm = RLRiskManager(config_r)
        unified = UnifiedRiskManager(mode="backtest", var_method="historical")

        for i in range(100):
            n = int(rng.integers(15, 101))
            returns = rng.normal(0, 0.01, n)
            b = brm.compute_var(returns, confidence_level=0.95)
            r = rlrm.compute_var(returns)
            u = unified.compute_var(returns, confidence_level=0.95)
            assert b is not None and r is not None and u is not None, (
                f"Scenario {i}: unexpected None (n={n})"
            )
            assert abs(b - r) < 1e-10, f"Scenario {i}: BRM={b} vs RL={r}"
            assert abs(b - u) < 1e-10, f"Scenario {i}: BRM={b} vs Unified={u}"
