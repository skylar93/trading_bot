"""H1: verify unified check_max_drawdown handles all call patterns."""
import pytest
from risk_management import create_risk_manager


class TestCheckMaxDrawdownUnified:
    def test_two_float_args(self):
        """Legacy 2-arg pattern: check_max_drawdown(peak, current)."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        # 25% drawdown > 20% threshold → True
        assert rm.check_max_drawdown(100.0, 75.0) is True
        # 10% drawdown < 20% threshold → False
        assert rm.check_max_drawdown(100.0, 90.0) is False

    def test_three_args_string_agent(self):
        """3-arg pattern: check_max_drawdown("agent", peak, current)."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_max_drawdown("env", 100.0, 75.0) is True
        assert rm.check_max_drawdown("env", 100.0, 90.0) is False

    def test_agent_id_lookup(self):
        """Agent ID lookup from stored values."""
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        rm.peak_values["agent1"] = 100.0
        rm.current_values["agent1"] = 75.0
        assert rm.check_max_drawdown("agent1") is True

    def test_unknown_agent_returns_false(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_max_drawdown("unknown") is False

    def test_zero_peak_returns_false(self):
        rm = create_risk_manager("rl", {"max_drawdown_pct": 0.20})
        assert rm.check_max_drawdown(0.0, 50.0) is False
