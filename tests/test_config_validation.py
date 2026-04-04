"""Test Pydantic config validators."""
import pytest
from config.schema import AgentConfig, RiskConfig


class TestAgentConfigValidation:
    def test_valid_learning_rate(self):
        cfg = AgentConfig(learning_rate=0.001)
        assert cfg.learning_rate == 0.001

    def test_zero_learning_rate_rejected(self):
        with pytest.raises(Exception):
            AgentConfig(learning_rate=0.0)

    def test_negative_learning_rate_rejected(self):
        with pytest.raises(Exception):
            AgentConfig(learning_rate=-0.01)

    def test_above_one_learning_rate_rejected(self):
        with pytest.raises(Exception):
            AgentConfig(learning_rate=1.5)


class TestRiskConfigCrossValidation:
    def test_valid_thresholds(self):
        cfg = RiskConfig(stop_loss_threshold=0.05, trailing_stop_buffer=0.03, max_drawdown_pct=0.20)
        assert cfg.stop_loss_threshold == 0.05

    def test_stop_loss_ge_drawdown_rejected(self):
        with pytest.raises(Exception):
            RiskConfig(stop_loss_threshold=0.30, max_drawdown_pct=0.20)

    def test_trailing_stop_ge_drawdown_rejected(self):
        with pytest.raises(Exception):
            RiskConfig(trailing_stop_buffer=0.25, max_drawdown_pct=0.20)

    def test_equal_thresholds_rejected(self):
        with pytest.raises(Exception):
            RiskConfig(stop_loss_threshold=0.20, max_drawdown_pct=0.20)
