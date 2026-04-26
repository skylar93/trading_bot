"""I7: PaperTrader → DeploymentDriftDetector wiring tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


def _make_trader(deployment_drift_detector=None, shadow_hours=72):
    """Build a minimal PaperTrader with mocked agent and config."""
    from deployment.paper_trader import PaperTrader

    agent = MagicMock()
    agent.predict.return_value = (0, None)
    config = {
        "paper_trading": {
            "initial_balance": 10_000,
            "trading_fee": 0.001,
        },
        "alerts": {
            "drift": {
                "shadow_mode_hours": shadow_hours,
            }
        },
    }
    trader = PaperTrader(
        agent=agent,
        config=config,
        simulation_mode=True,
        deployment_drift_detector=deployment_drift_detector,
    )
    return trader


def test_dep_drift_auto_created_when_none():
    """I7-a: Passing deployment_drift_detector=None auto-creates one."""
    trader = _make_trader(deployment_drift_detector=None)
    assert trader._dep_drift is not None


def test_dep_drift_injection():
    """I7-a: Explicit injection is stored as-is."""
    from deployment.monitoring.drift_detector import DeploymentDriftDetector
    dd = DeploymentDriftDetector(config={})
    trader = _make_trader(deployment_drift_detector=dd)
    assert trader._dep_drift is dd


def test_drift_during_shadow_no_shutdown():
    """I7-c: Drift during shadow period → dep_drift.report_drift called, no shutdown."""
    import time
    from deployment.monitoring.drift_detector import DeploymentDriftDetector

    # Shadow mode for 1000h to ensure we're still in shadow
    dd = DeploymentDriftDetector(config={"drift": {"shadow_mode_hours": 1000}})
    trader = _make_trader(deployment_drift_detector=dd)
    trader._trigger_shutdown = MagicMock()

    # Inject a training-side drift_detector that always fires
    mock_dd = MagicMock()
    mock_dd.update.return_value = True
    mock_dd.method = "adwin"
    trader.drift_detector = mock_dd

    # Simulate two portfolio values so step_return is computed
    trader.state.portfolio_history.append(10_000.0)
    trader.state.portfolio_history.append(9_000.0)

    trader._check_drift()

    # halt_requested stays False in shadow mode
    assert not dd.halt_requested
    # _trigger_shutdown should NOT have been called
    trader._trigger_shutdown.assert_not_called()


def test_drift_after_shadow_triggers_shutdown():
    """I7-c: Drift after shadow period → halt_requested=True → _trigger_shutdown called."""
    import time
    from deployment.monitoring.drift_detector import DeploymentDriftDetector

    # Shadow mode expired (0 hours = immediate expiry)
    dd = DeploymentDriftDetector(
        config={"drift": {"shadow_mode_hours": 0}},
        _start_time=time.time() - 3600,  # started 1h ago, shadow=0 → already expired
    )
    trader = _make_trader(deployment_drift_detector=dd)
    trader._trigger_shutdown = MagicMock()

    mock_dd = MagicMock()
    mock_dd.update.return_value = True
    mock_dd.method = "adwin"
    trader.drift_detector = mock_dd

    trader.state.portfolio_history.append(10_000.0)
    trader.state.portfolio_history.append(9_000.0)

    trader._check_drift()

    assert dd.halt_requested
    trader._trigger_shutdown.assert_called_once_with(reason="deployment_drift_halt")
