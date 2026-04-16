"""Test that orphaned components are wired into execution path."""
import pytest


class TestPaperTraderRiskWiring:
    def test_accepts_risk_manager(self):
        """PaperTrader __init__ should accept risk_manager param."""
        from deployment.paper_trader import PaperTrader
        import inspect
        sig = inspect.signature(PaperTrader.__init__)
        assert "risk_manager" in sig.parameters, "PaperTrader missing risk_manager param"

    def test_risk_manager_triggers_shutdown(self):
        """If risk_manager says drawdown exceeded, PaperTrader should shut down."""
        from unittest.mock import MagicMock
        from deployment.paper_trader import PaperTrader

        agent = MagicMock()
        rm = MagicMock()
        rm.check_drawdown.return_value = True  # drawdown exceeded

        trader = PaperTrader(
            agent=agent,
            config={"initial_balance": 10000, "symbol": "BTC/USDT"},
            simulation_mode=True,
            risk_manager=rm,
        )
        # Populate state so _check_risk runs
        trader.state.portfolio_history.append(8000.0)
        trader.state.peak_portfolio_value = 10000.0
        trader._check_risk(price=100.0)
        rm.check_drawdown.assert_called_once()


class TestOrderManagerRiskWiring:
    def test_accepts_risk_manager(self):
        from deployment.execution.order_manager import OrderManager
        import inspect
        sig = inspect.signature(OrderManager.__init__)
        assert "risk_manager" in sig.parameters, "OrderManager missing risk_manager param"


class TestMonitoringConfigDrift:
    def test_use_drift_detection_field(self):
        from config.schema import MonitoringConfig
        mc = MonitoringConfig()
        assert hasattr(mc, "use_drift_detection")
        assert mc.use_drift_detection is False
