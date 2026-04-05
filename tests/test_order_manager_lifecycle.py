"""Test OrderManager order lifecycle and PnL calculation."""
import pytest
from deployment.execution.order_manager import OrderManager


class TestParameterNames:
    """C1: verify live fill uses correct parameter name 'quantity'."""

    def test_live_fill_calls_quantity(self):
        """Ensure _execute_live_order passes 'quantity' not 'qty'."""
        import inspect
        source = inspect.getsource(OrderManager._execute_live_order)
        assert "qty=" not in source, (
            "_execute_live_order still uses 'qty=' instead of 'quantity='"
        )
        assert "quantity=" in source


class TestDailyPnL:
    """C2: verify daily PnL calculation uses entry price."""

    def test_paper_sell_pnl_positive(self):
        om = OrderManager(paper_mode=True)
        # Buy first
        om.submit_order("buy", 1.0, current_price=100.0)
        # Sell at higher price
        pnl_before = om._daily_pnl
        om.submit_order("sell", 1.0, current_price=120.0)
        pnl_after = om._daily_pnl
        profit = pnl_after - pnl_before
        # Profit should be ~20 (minus fees), NOT ~0
        assert profit > 15.0, f"PnL should reflect price gain, got {profit}"

    def test_paper_sell_pnl_negative(self):
        om = OrderManager(paper_mode=True)
        om.submit_order("buy", 1.0, current_price=100.0)
        pnl_before = om._daily_pnl
        om.submit_order("sell", 1.0, current_price=80.0)
        pnl_after = om._daily_pnl
        loss = pnl_after - pnl_before
        assert loss < -15.0, f"PnL should reflect price loss, got {loss}"

    def test_daily_loss_limit_triggers(self):
        om = OrderManager(
            exchange_config={"daily_loss_limit": -10.0},
            paper_mode=True,
        )
        om.submit_order("buy", 1.0, current_price=100.0)
        om.submit_order("sell", 1.0, current_price=50.0)
        assert om._halted, "Trading should be halted after large loss"


class TestSellValidation:
    """H3: verify sell quantity clamping."""

    def test_sell_more_than_position_clamped(self):
        om = OrderManager(paper_mode=True)
        om.submit_order("buy", 1.0, current_price=100.0)
        # Try to sell more than held
        om.submit_order("sell", 5.0, current_price=100.0)
        pos = om._position_tracker.position
        assert pos >= 0, f"Position should not go negative, got {pos}"
