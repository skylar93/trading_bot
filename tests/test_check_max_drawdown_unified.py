"""H1: verify check_drawdown handles all call patterns (new API + deprecated shim)."""
import warnings
import pytest
from risk_management import create_risk_manager


class TestCheckDrawdownUnified:
    def test_two_float_args(self):
        """2-arg pattern: check_drawdown(peak, current)."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        # 25% drawdown > 20% threshold → True
        assert rm.check_drawdown(100.0, 75.0) is True
        # 10% drawdown < 20% threshold → False
        assert rm.check_drawdown(100.0, 90.0) is False

    def test_three_args_string_agent(self):
        """3-arg pattern: check_drawdown("agent", peak, current)."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_drawdown("env", 100.0, 75.0) is True
        assert rm.check_drawdown("env", 100.0, 90.0) is False

    def test_agent_id_lookup(self):
        """Agent ID lookup from stored values."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        rm.peak_values["agent1"] = 100.0
        rm.current_values["agent1"] = 75.0
        assert rm.check_drawdown("agent1") is True

    def test_unknown_agent_returns_false(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_drawdown("unknown") is False

    def test_zero_peak_returns_false(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_drawdown(0.0, 50.0) is False


class TestDeprecatedCheckMaxDrawdownShim:
    """Shim check_max_drawdown still works but emits DeprecationWarning."""

    def test_shim_emits_deprecation_warning(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = rm.check_max_drawdown(100.0, 75.0)
        assert result is True
        assert any(issubclass(x.category, DeprecationWarning) for x in w), (
            "check_max_drawdown() should emit DeprecationWarning"
        )

    def test_shim_delegates_correctly(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert rm.check_max_drawdown(100.0, 90.0) is False
            assert rm.check_max_drawdown(100.0, 75.0) is True
