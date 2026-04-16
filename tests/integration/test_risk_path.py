"""
Week 70 — E12: End-to-end risk path integration test.

Scenario: PaperTrader → OrderManager → RiskManager (concrete, no mocks).

Verifies that a real RLRiskManager / BacktestingRiskManager sitting inside
PaperTrader and OrderManager actually blocks orders and triggers shutdown
when drawdown exceeds the configured threshold.
"""
import pytest
from unittest.mock import MagicMock

from risk_management.rl_risk_manager import RLRiskManager, RLRiskConfig
from risk_management.backtesting_risk_manager import BacktestingRiskManager, BacktestingRiskConfig
from risk_management.unified_risk_manager import UnifiedRiskManager
from deployment.execution.order_manager import OrderManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rl_risk_manager(max_drawdown_pct: float = 0.15) -> RLRiskManager:
    config = RLRiskConfig(max_drawdown_pct=max_drawdown_pct, use_var=False)
    return RLRiskManager(config)


def _make_brm(max_drawdown_pct: float = 0.15) -> BacktestingRiskManager:
    config = BacktestingRiskConfig(max_drawdown_pct=max_drawdown_pct)
    return BacktestingRiskManager(config)


def _make_order_manager(risk_manager) -> OrderManager:
    return OrderManager(
        exchange_config={"symbol": "BTC/USDT", "fat_finger_hard_cap": 0.0},
        paper_mode=True,
        risk_manager=risk_manager,
    )


# ---------------------------------------------------------------------------
# E12-A: OrderManager rejects order when drawdown exceeded
# ---------------------------------------------------------------------------

class TestOrderManagerRiskPath:
    """OrderManager must reject a buy order when risk_manager reports drawdown breach."""

    def _submit_buy(self, mgr: OrderManager, peak: float, current: float) -> str:
        """Simulate a position tracker state and submit a buy."""
        tracker = MagicMock()
        tracker.peak_value = peak
        tracker.portfolio_value = current
        mgr._position_tracker = tracker
        return mgr.submit_order("buy", 0.01)

    def _pre_risk_rejected(self, mgr: OrderManager) -> list:
        """Orders rejected by risk check have no submitted_at (set only after risk passes)."""
        return [o for o in mgr._orders.values() if o.status == "failed" and o.submitted_at is None]

    def _passed_risk_check(self, mgr: OrderManager) -> list:
        """Orders that passed the risk check have submitted_at set (even if execution fails)."""
        return [o for o in mgr._orders.values() if o.submitted_at is not None]

    def test_rlrm_rejects_on_drawdown(self):
        """RLRiskManager (2-arg pattern) causes OrderManager to pre-risk-reject the order."""
        rm = _make_rl_risk_manager(max_drawdown_pct=0.15)
        mgr = _make_order_manager(rm)

        # 20% drawdown > 15% threshold → order rejected before execution
        order_id = self._submit_buy(mgr, peak=10_000.0, current=8_000.0)
        assert order_id is not None
        rejected = self._pre_risk_rejected(mgr)
        assert len(rejected) >= 1, "Expected at least one pre-risk-rejected order on drawdown breach"

    def test_rlrm_allows_on_safe_drawdown(self):
        """RLRiskManager allows order when drawdown is within limit — order proceeds past risk check."""
        rm = _make_rl_risk_manager(max_drawdown_pct=0.15)
        mgr = _make_order_manager(rm)

        # 5% drawdown < 15% threshold → risk check passes, order gets submitted_at
        self._submit_buy(mgr, peak=10_000.0, current=9_500.0)
        passed = self._passed_risk_check(mgr)
        assert len(passed) >= 1, "Order should reach execution phase when within drawdown limit"

    def test_brm_rejects_on_drawdown(self):
        """BacktestingRiskManager also causes pre-risk-rejection — same code path."""
        rm = _make_brm(max_drawdown_pct=0.15)
        mgr = _make_order_manager(rm)

        order_id = self._submit_buy(mgr, peak=10_000.0, current=8_000.0)
        assert order_id is not None
        rejected = self._pre_risk_rejected(mgr)
        assert len(rejected) >= 1


# ---------------------------------------------------------------------------
# E12-B: PaperTrader _check_risk triggers shutdown via RiskManager
# ---------------------------------------------------------------------------

class TestPaperTraderRiskPath:
    """PaperTrader must call check_drawdown on the real risk manager and halt."""

    def _make_trader(self, rm):
        from deployment.paper_trader import PaperTrader
        agent = MagicMock()
        agent.predict.return_value = (0.0, None)
        return PaperTrader(
            agent=agent,
            config={"initial_balance": 10_000, "symbol": "BTC/USDT"},
            simulation_mode=True,
            risk_manager=rm,
        )

    def test_shutdown_on_drawdown_breach(self):
        """PaperTrader halts when risk_manager.check_drawdown says breach."""
        rm = _make_rl_risk_manager(max_drawdown_pct=0.15)
        trader = self._make_trader(rm)

        # 20% drawdown
        trader.state.portfolio_history.append(8_000.0)
        trader.state.peak_portfolio_value = 10_000.0

        trader._check_risk(price=100.0)
        assert trader.state.shutdown_triggered, "PaperTrader should shutdown on drawdown breach"

    def test_no_shutdown_within_limit(self):
        """PaperTrader continues when drawdown is within limit."""
        rm = _make_rl_risk_manager(max_drawdown_pct=0.15)
        trader = self._make_trader(rm)

        # 5% drawdown
        trader.state.portfolio_history.append(9_500.0)
        trader.state.peak_portfolio_value = 10_000.0

        trader._check_risk(price=100.0)
        assert not trader.state.shutdown_triggered, "PaperTrader should not shutdown within drawdown limit"


# ---------------------------------------------------------------------------
# E12-C: UnifiedRiskManager math agrees with concrete managers on same scenario
# ---------------------------------------------------------------------------

class TestUnifiedMathAgreement:
    """
    UnifiedRiskManager.check_drawdown must agree with BRM and RL on identical inputs.
    This is the single-path verification promised by E12.
    """

    def test_drawdown_path_agreement(self):
        peak, current = 10_000.0, 8_400.0  # 16% drawdown > 15%
        threshold = 0.15

        unified = UnifiedRiskManager(mode="live")
        rm_rl = _make_rl_risk_manager(max_drawdown_pct=threshold)
        rm_brm = _make_brm(max_drawdown_pct=threshold)

        uni_result = unified.check_drawdown(peak, current, threshold)
        rl_result = rm_rl.check_drawdown(peak, current)
        brm_result = rm_brm.check_drawdown(peak, current)

        assert uni_result is True
        assert rl_result == uni_result, f"RL diverged: rl={rl_result}, unified={uni_result}"
        assert brm_result == uni_result, f"BRM diverged: brm={brm_result}, unified={uni_result}"

    def test_var_path_agreement(self):
        import numpy as np
        rng = np.random.default_rng(70)
        returns = rng.normal(0, 0.01, 50)

        unified = UnifiedRiskManager(mode="backtest", var_method="historical")
        rm_rl = _make_rl_risk_manager()
        rm_brm = _make_brm()

        uni_var = unified.compute_var(returns, confidence_level=0.95)
        rl_var = rm_rl.compute_var(returns)
        brm_var = rm_brm.compute_var(returns, confidence_level=0.95)

        assert uni_var is not None and rl_var is not None and brm_var is not None
        assert abs(rl_var - uni_var) < 1e-10, f"RL VaR diverged: {rl_var} vs {uni_var}"
        assert abs(brm_var - uni_var) < 1e-10, f"BRM VaR diverged: {brm_var} vs {uni_var}"
