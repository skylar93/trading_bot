"""Golden tests for VaR, CVaR, trailing stop, portfolio VaR."""
import numpy as np
import pytest


class TestVaRCorrectness:
    def test_parametric_var_positive(self):
        from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
        cfg = RLRiskConfig(use_parametric_var=True, var_confidence_level=0.95)
        rm = RLRiskManager(cfg)
        returns = np.random.RandomState(42).normal(0, 0.02, 1000)
        var = rm.calculate_var(returns)
        assert var > 0, f"VaR should be positive, got {var}"
        assert 0.02 < var < 0.06, f"95% VaR of N(0,0.02) ~0.033, got {var}"

    def test_historical_var_positive(self):
        from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
        cfg = RLRiskConfig(use_parametric_var=False, var_confidence_level=0.95)
        rm = RLRiskManager(cfg)
        returns = np.random.RandomState(42).normal(0, 0.02, 1000)
        var = rm.calculate_var(returns)
        assert var > 0, f"VaR should be positive, got {var}"

    def test_cvar_ge_var(self):
        from risk_management.backtesting_risk_manager import BacktestingRiskManager
        import pandas as pd
        from risk_management.backtesting_risk_manager import BacktestingRiskConfig
        rm = BacktestingRiskManager(BacktestingRiskConfig())
        returns = pd.Series(np.random.RandomState(42).normal(0, 0.02, 1000))
        var = rm.calculate_var(returns, 0.95)
        cvar = rm.calculate_cvar(returns, 0.95)
        assert cvar >= var, f"CVaR ({cvar}) should >= VaR ({var})"


class TestTrailingStopHWM:
    def test_hwm_only_increases(self):
        from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
        rm = RLRiskManager(RLRiskConfig(use_trailing_stop=True))
        rm.update_trailing_stop("BTC", 100.0)
        rm.update_trailing_stop("BTC", 110.0)
        rm.update_trailing_stop("BTC", 105.0)
        assert rm._trailing_hwm["BTC"] == 110.0


class TestPortfolioVaR:
    def test_portfolio_var_reasonable(self):
        from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
        rm = RLRiskManager(RLRiskConfig(use_portfolio_var=True, var_confidence_level=0.95))
        rng = np.random.RandomState(42)
        rm.asset_returns_history = {
            "A": list(rng.normal(0, 0.02, 100)),
            "B": list(rng.normal(0, 0.02, 100)),
        }
        var = rm._calculate_portfolio_var({"A": 1.0, "B": 1.0}, {"A": 100.0, "B": 100.0})
        assert var is not None
        assert 0.01 < var < 0.10, f"Portfolio VaR should be reasonable, got {var}"
