"""
Week 74 — Execution Realism Tests (F12-F16).

Coverage:
  F12 — Order types: limit, stop_loss_limit, take_profit paper simulation
  F13 — Partial fill simulation + live status mapping
  F14 — Cancel-replace + TTL expiry
  F15 — SlippageModel fit/predict
  F16 — FeeModel compute_fee / VIP tier / BNB discount / refresh
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deployment.execution.order_manager import Order, OrderManager, _ORDER_TYPES
from deployment.analysis.slippage_model import SlippageModel, SlippageObservation
from deployment.exchange.fee_model import FeeModel, _BINANCE_VIP_SCHEDULE, _BNB_DISCOUNT_FRACTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_manager(**kwargs) -> OrderManager:
    """Convenience: paper-mode OrderManager with override-able config."""
    cfg = {"initial_cash": 100_000.0, "max_order_size": 10.0}
    cfg.update(kwargs)
    return OrderManager(exchange_config=cfg, paper_mode=True)


def make_manager_with_price(price: float, **kwargs) -> OrderManager:
    m = make_manager(**kwargs)
    m.update_paper_price(price)
    return m


# ---------------------------------------------------------------------------
# F12: Order types
# ---------------------------------------------------------------------------

class TestOrderTypesValidation:
    def test_valid_types_accepted(self):
        for ot in _ORDER_TYPES:
            m = make_manager_with_price(100.0)
            oid = m.submit_order("buy", 0.1, order_type=ot, limit_price=100.0, stop_price=100.0)
            assert m.check_order(oid) in ("filled", "partial", "pending")

    def test_invalid_type_raises(self):
        m = make_manager_with_price(100.0)
        with pytest.raises(ValueError, match="Invalid order_type"):
            m.submit_order("buy", 0.1, order_type="twap")


class TestMarketOrder:
    def test_market_buy_fills_at_current_price(self):
        m = make_manager_with_price(200.0)
        oid = m.submit_order("buy", 0.5, order_type="market", current_price=200.0)
        order = m.get_order(oid)
        assert order.status == "filled"
        assert order.avg_fill_price == pytest.approx(200.0)
        assert order.filled_amount == pytest.approx(0.5)

    def test_market_sell_fills_at_current_price(self):
        m = make_manager_with_price(200.0)
        m.submit_order("buy", 1.0, current_price=200.0)
        oid = m.submit_order("sell", 0.5, order_type="market", current_price=200.0)
        order = m.get_order(oid)
        assert order.status == "filled"
        assert order.avg_fill_price == pytest.approx(200.0)


class TestLimitOrder:
    def test_limit_buy_fills_when_price_at_or_below_limit(self):
        m = make_manager_with_price(95.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=100.0, current_price=95.0)
        assert m.check_order(oid) == "filled"

    def test_limit_buy_stays_pending_when_price_above_limit(self):
        m = make_manager_with_price(110.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=100.0, current_price=110.0)
        assert m.check_order(oid) == "pending"

    def test_limit_buy_fills_on_price_update(self):
        m = make_manager_with_price(110.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=100.0, current_price=110.0)
        assert m.check_order(oid) == "pending"
        m.update_paper_price(98.0)   # price drops below limit
        assert m.check_order(oid) == "filled"

    def test_limit_sell_fills_when_price_at_or_above_limit(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order("sell", 0.5, order_type="limit", limit_price=105.0, current_price=100.0)
        assert m.check_order(oid) == "pending"
        m.update_paper_price(106.0)
        assert m.check_order(oid) == "filled"

    def test_limit_fill_price_equals_limit_price(self):
        m = make_manager_with_price(95.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=100.0, current_price=95.0)
        order = m.get_order(oid)
        assert order.avg_fill_price == pytest.approx(100.0)


class TestStopLossLimit:
    def test_stop_loss_sell_triggers_below_stop(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        # Stop-loss sell: trigger when price drops to stop_price
        oid = m.submit_order(
            "sell", 0.5,
            order_type="stop_loss_limit",
            stop_price=90.0, limit_price=89.0,
            current_price=100.0,
        )
        assert m.check_order(oid) == "pending"   # not triggered yet
        m.update_paper_price(88.0)               # crosses stop
        assert m.check_order(oid) == "filled"

    def test_stop_loss_sell_not_triggered_above_stop(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order(
            "sell", 0.5,
            order_type="stop_loss_limit",
            stop_price=90.0, limit_price=89.0,
            current_price=100.0,
        )
        m.update_paper_price(95.0)
        assert m.check_order(oid) == "pending"

    def test_stop_loss_buy_triggers_above_stop(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order(
            "buy", 0.1,
            order_type="stop_loss_limit",
            stop_price=110.0, limit_price=111.0,
            current_price=100.0,
        )
        assert m.check_order(oid) == "pending"
        m.update_paper_price(112.0)
        assert m.check_order(oid) == "filled"

    def test_fill_price_equals_limit_price(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order(
            "sell", 0.5,
            order_type="stop_loss_limit",
            stop_price=90.0, limit_price=89.5,
            current_price=88.0,   # already triggered
        )
        order = m.get_order(oid)
        assert order.avg_fill_price == pytest.approx(89.5)


class TestTakeProfit:
    def test_take_profit_sell_triggers_above_stop(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order(
            "sell", 0.5,
            order_type="take_profit",
            stop_price=120.0, limit_price=121.0,
            current_price=100.0,
        )
        assert m.check_order(oid) == "pending"
        m.update_paper_price(122.0)
        assert m.check_order(oid) == "filled"

    def test_take_profit_sell_not_triggered_below_stop(self):
        m = make_manager_with_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order(
            "sell", 0.5,
            order_type="take_profit",
            stop_price=120.0, limit_price=121.0,
            current_price=100.0,
        )
        m.update_paper_price(115.0)
        assert m.check_order(oid) == "pending"


# ---------------------------------------------------------------------------
# F13: Partial fills
# ---------------------------------------------------------------------------

class TestPartialFillSimulation:
    def test_full_fill_when_sim_disabled(self):
        m = make_manager_with_price(100.0, partial_fill_sim=False)
        oid = m.submit_order("buy", 1.0, current_price=100.0)
        order = m.get_order(oid)
        assert order.status == "filled"
        assert order.filled_amount == pytest.approx(1.0)

    def test_partial_fill_when_sim_enabled(self):
        np.random.seed(42)
        m = make_manager(partial_fill_sim=True, partial_fill_min_ratio=0.3)
        m.update_paper_price(100.0)
        # With seed 42, draws won't always be 1.0 — run multiple to get a partial
        any_partial = False
        for _ in range(30):
            m2 = make_manager(partial_fill_sim=True, partial_fill_min_ratio=0.3)
            m2.update_paper_price(100.0)
            oid = m2.submit_order("buy", 1.0, current_price=100.0)
            order = m2.get_order(oid)
            if order.status == "partial":
                any_partial = True
                assert order.filled_amount < order.amount
                assert order.filled_amount >= order.amount * 0.3
                break
        assert any_partial, "Expected at least one partial fill in 30 attempts"

    def test_fills_list_populated(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order("buy", 0.5, current_price=100.0)
        order = m.get_order(oid)
        assert len(order.fills) == 1
        fill = order.fills[0]
        assert fill["order_id"] == oid
        assert fill["side"] == "buy"
        assert fill["qty"] == pytest.approx(0.5)
        assert "timestamp" in fill
        assert "fill_id" in fill

    def test_fill_event_audit_logged(self):
        audit = MagicMock()
        m = OrderManager(
            exchange_config={"initial_cash": 100_000.0, "max_order_size": 10.0},
            paper_mode=True,
            audit_logger=audit,
        )
        m.update_paper_price(100.0)
        m.submit_order("buy", 0.5, current_price=100.0)
        # log_order + log_fill (order) + log_fill (fill event) = at least 2 log_fill calls
        assert audit.log_fill.call_count >= 1


class TestLivePartialFillMapping:
    def _make_live_manager(self, result: dict) -> OrderManager:
        # Construct in paper mode then switch to live to bypass ccxt import.
        m = OrderManager(exchange_config={"max_order_size": 10.0}, paper_mode=True)
        m.paper_mode = False
        m._exchange_mode = "live"
        m._exchange = MagicMock()
        m._exchange.create_market_order.return_value = result
        # fetch_order returns same result so _refresh_live_order_status stays consistent
        m._exchange.fetch_order.return_value = result
        m.rate_limiter = MagicMock()
        return m

    def test_closed_maps_to_filled(self):
        m = self._make_live_manager(
            {"id": "x1", "status": "closed", "filled": 1.0, "remaining": 0.0, "average": 100.0, "fee": {}}
        )
        oid = m.submit_order("buy", 1.0, order_type="market")
        assert m.check_order(oid) == "filled"

    def test_open_with_partial_fill_maps_to_partial(self):
        m = self._make_live_manager(
            {"id": "x2", "status": "open", "filled": 0.4, "remaining": 0.6, "average": 100.0, "fee": {}}
        )
        oid = m.submit_order("buy", 1.0, order_type="market")
        assert m.check_order(oid) == "partial"

    def test_open_with_zero_fill_maps_to_pending(self):
        m = self._make_live_manager(
            {"id": "x3", "status": "open", "filled": 0.0, "remaining": 1.0, "average": None, "fee": {}}
        )
        oid = m.submit_order("buy", 1.0, order_type="market")
        assert m.check_order(oid) == "pending"

    def test_canceled_maps_to_cancelled(self):
        m = self._make_live_manager(
            {"id": "x4", "status": "canceled", "filled": 0.0, "remaining": 1.0, "average": None, "fee": {}}
        )
        oid = m.submit_order("buy", 1.0, order_type="market")
        assert m.check_order(oid) == "cancelled"

    def test_partial_fill_event_recorded(self):
        m = self._make_live_manager(
            {"id": "x5", "status": "open", "filled": 0.4, "remaining": 0.6, "average": 100.0, "fee": {"cost": 0.04}}
        )
        oid = m.submit_order("buy", 1.0, order_type="market")
        order = m.get_order(oid)
        assert len(order.fills) == 1
        assert order.fills[0]["is_partial"] is True


# ---------------------------------------------------------------------------
# F14: Cancel-replace / TTL expiry
# ---------------------------------------------------------------------------

class TestCancelReplace:
    def test_cancel_replace_submits_new_order(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=90.0, current_price=100.0)
        assert m.check_order(oid) == "pending"
        new_oid = m.cancel_replace_order(
            oid, side="buy", amount=0.2, order_type="limit", limit_price=88.0, current_price=100.0
        )
        assert m.check_order(oid) == "cancelled"
        assert m.check_order(new_oid) == "pending"
        new_order = m.get_order(new_oid)
        assert new_order.amount == pytest.approx(0.2)
        assert new_order.limit_price == pytest.approx(88.0)

    def test_cancel_replace_raises_on_filled_order(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order("buy", 0.1, current_price=100.0)
        assert m.check_order(oid) == "filled"
        with pytest.raises(RuntimeError, match="cancel_replace_order"):
            m.cancel_replace_order(oid, side="buy", amount=0.1)

    def test_cancel_replace_alerter_triggered_on_failure(self):
        alerter = MagicMock()
        m = make_manager_with_price(100.0, alerter=alerter)
        # Give it a real OrderManager so we can inject alerter
        m._alerter = alerter
        oid = m.submit_order("buy", 0.1, current_price=100.0)  # fills immediately
        assert m.check_order(oid) == "filled"
        with pytest.raises(RuntimeError):
            m.cancel_replace_order(oid, side="buy", amount=0.1)
        alerter.notify_error.assert_called_once()


class TestTTLExpiry:
    def test_pending_order_cancelled_after_ttl(self):
        m = OrderManager(
            exchange_config={
                "initial_cash": 100_000.0,
                "max_order_size": 10.0,
                "order_ttl_sec": 0.1,          # 100ms TTL
                "order_ttl_check_interval_sec": 0.05,
            },
            paper_mode=True,
        )
        m.update_paper_price(100.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=90.0, current_price=100.0)
        assert m.check_order(oid) == "pending"
        time.sleep(0.3)   # wait for expiry worker to run
        assert m.check_order(oid) == "cancelled"
        m.close()

    def test_filled_order_not_cancelled_by_ttl(self):
        m = OrderManager(
            exchange_config={
                "initial_cash": 100_000.0,
                "max_order_size": 10.0,
                "order_ttl_sec": 0.1,
                "order_ttl_check_interval_sec": 0.05,
            },
            paper_mode=True,
        )
        m.update_paper_price(100.0)
        oid = m.submit_order("buy", 0.1, current_price=100.0)   # fills immediately
        assert m.check_order(oid) == "filled"
        time.sleep(0.3)
        assert m.check_order(oid) == "filled"
        m.close()

    def test_per_order_ttl_override(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order(
            "buy", 0.1, order_type="limit", limit_price=90.0,
            current_price=100.0, ttl_sec=60.0
        )
        order = m.get_order(oid)
        assert order.expires_at is not None
        assert order.expires_at > datetime.utcnow()

    def test_ttl_zero_no_expiry(self):
        m = make_manager_with_price(100.0)
        oid = m.submit_order("buy", 0.1, order_type="limit", limit_price=90.0, current_price=100.0)
        order = m.get_order(oid)
        assert order.expires_at is None

    def test_expiry_worker_stops_on_close(self):
        m = OrderManager(
            exchange_config={"initial_cash": 100_000.0, "max_order_size": 10.0, "order_ttl_sec": 60.0},
            paper_mode=True,
        )
        assert m._expiry_thread is not None
        assert m._expiry_thread.is_alive()
        m.close()
        time.sleep(0.1)
        assert not m._expiry_thread.is_alive()

    def test_ttl_audit_logged(self):
        audit = MagicMock()
        m = OrderManager(
            exchange_config={
                "initial_cash": 100_000.0,
                "max_order_size": 10.0,
                "order_ttl_sec": 0.1,
                "order_ttl_check_interval_sec": 0.05,
            },
            paper_mode=True,
            audit_logger=audit,
        )
        m.update_paper_price(100.0)
        m.submit_order("buy", 0.1, order_type="limit", limit_price=90.0, current_price=100.0)
        time.sleep(0.3)
        m.close()
        # Check that order_expired event was logged
        risk_calls = [str(c) for c in audit.log_risk_event.call_args_list]
        assert any("order_expired" in c for c in risk_calls)


# ---------------------------------------------------------------------------
# F15: SlippageModel
# ---------------------------------------------------------------------------

class TestSlippageModel:
    def _make_observations(self, n: int = 50, seed: int = 0) -> list:
        rng = np.random.default_rng(seed)
        obs = []
        for _ in range(n):
            expected = 10_000.0 + rng.uniform(-500, 500)
            slip = rng.uniform(0.0001, 0.002)   # 1-20 bps
            fill = expected * (1 + rng.choice([1, -1]) * slip)
            obs.append(SlippageObservation(
                side=rng.choice(["buy", "sell"]),
                order_size=float(rng.uniform(0.01, 1.0)),
                fill_price=fill,
                expected_price=expected,
                bar_volume=float(rng.uniform(100, 10_000)),
                realized_vol=float(rng.uniform(0.005, 0.05)),
            ))
        return obs

    def test_fit_returns_fitted_true(self):
        model = SlippageModel()
        result = model.fit(self._make_observations(50))
        assert result["fitted"] is True
        assert result["n_samples"] == 50
        assert result["r2"] is not None

    def test_fit_requires_min_observations(self):
        model = SlippageModel(min_observations=10)
        result = model.fit(self._make_observations(5))
        assert result["fitted"] is False

    def test_predict_returns_float_in_range(self):
        model = SlippageModel(max_slippage_frac=0.02)
        model.fit(self._make_observations(50))
        pred = model.predict(volume=5000.0, realized_vol=0.02, side="buy", size=0.1)
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 0.02

    def test_predict_zero_before_fit(self):
        model = SlippageModel()
        assert model.predict(volume=5000.0, realized_vol=0.02, side="buy", size=0.1) == 0.0

    def test_summary_contains_stats(self):
        model = SlippageModel()
        model.fit(self._make_observations(50))
        s = model.summary()
        assert s["fitted"] is True
        assert "mean_slippage_frac" in s
        assert "coefficients" in s
        assert "intercept" in s["coefficients"]

    def test_record_then_fit(self):
        model = SlippageModel()
        for obs in self._make_observations(20):
            model.record(obs)
        result = model.fit()
        assert result["fitted"] is True

    def test_prediction_clipped_to_max(self):
        model = SlippageModel(max_slippage_frac=0.005)
        model.fit(self._make_observations(50))
        # Force very high inputs — prediction should be capped
        pred = model.predict(volume=0.0001, realized_vol=10.0, side="sell", size=1e6)
        assert pred <= 0.005

    def test_slippage_observation_frac(self):
        obs = SlippageObservation(
            side="buy",
            order_size=0.1,
            fill_price=10010.0,
            expected_price=10000.0,
            bar_volume=1000.0,
            realized_vol=0.01,
        )
        assert obs.slippage_frac == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# F16: FeeModel
# ---------------------------------------------------------------------------

class TestFeeModel:
    def test_default_taker_fee(self):
        model = FeeModel()   # VIP0: 10bps taker
        fee = model.compute_fee(quantity=1.0, price=10_000.0, is_maker=False)
        assert fee == pytest.approx(10.0)   # 0.10% × 10,000

    def test_maker_fee_lower_than_taker(self):
        model = FeeModel(vip_tier=1)   # VIP1: maker=9bps, taker=10bps
        taker_fee = model.compute_fee(1.0, 10_000.0, is_maker=False)
        maker_fee = model.compute_fee(1.0, 10_000.0, is_maker=True)
        assert maker_fee < taker_fee

    def test_bnb_discount_applied(self):
        model = FeeModel(bnb_discount=True)
        fee_no_bnb = FeeModel(bnb_discount=False).compute_fee(1.0, 10_000.0, is_maker=False)
        fee_bnb = model.compute_fee(1.0, 10_000.0, is_maker=False)
        expected = fee_no_bnb * (1 - _BNB_DISCOUNT_FRACTION)
        assert fee_bnb == pytest.approx(expected)

    def test_bnb_discount_override_per_order(self):
        model = FeeModel(bnb_discount=False)
        fee_default = model.compute_fee(1.0, 10_000.0, use_bnb=False)
        fee_bnb = model.compute_fee(1.0, 10_000.0, use_bnb=True)
        assert fee_bnb < fee_default

    def test_vip_tier_updates_rates(self):
        model = FeeModel(vip_tier=0)
        assert model._maker_bps == pytest.approx(10.0)
        model.set_vip_tier(3)
        assert model._maker_bps == pytest.approx(_BINANCE_VIP_SCHEDULE[3].maker_bps)

    def test_explicit_override_ignores_tier(self):
        model = FeeModel(maker_bps=5.0, taker_bps=6.0, vip_tier=3)
        model.set_vip_tier(0)   # should not change explicit overrides
        assert model._maker_bps == pytest.approx(5.0)
        assert model._taker_bps == pytest.approx(6.0)

    def test_effective_rate_fraction(self):
        model = FeeModel()   # 10bps taker
        assert model.effective_rate(is_maker=False) == pytest.approx(0.001)

    def test_flat_constructor(self):
        model = FeeModel.flat(rate_bps=5.0)
        assert model._maker_bps == pytest.approx(5.0)
        assert model._taker_bps == pytest.approx(5.0)

    def test_refresh_from_exchange(self):
        model = FeeModel()
        exchange = MagicMock()
        exchange.fetch_trading_fees.return_value = {
            "BTC/USDT": {"maker": 0.0008, "taker": 0.0010}
        }
        result = model.refresh_from_exchange(exchange, symbol="BTC/USDT")
        assert result is True
        assert model._maker_bps == pytest.approx(8.0)
        assert model._taker_bps == pytest.approx(10.0)

    def test_refresh_from_exchange_failure_returns_false(self):
        model = FeeModel()
        exchange = MagicMock()
        exchange.fetch_trading_fees.side_effect = Exception("network error")
        result = model.refresh_from_exchange(exchange)
        assert result is False

    def test_needs_refresh_true_initially(self):
        model = FeeModel(refresh_interval_sec=60.0)
        assert model.needs_refresh() is True

    def test_needs_refresh_false_after_refresh(self):
        model = FeeModel(refresh_interval_sec=3600.0)
        exchange = MagicMock()
        exchange.fetch_trading_fees.return_value = {"BTC/USDT": {"maker": 0.001, "taker": 0.001}}
        model.refresh_from_exchange(exchange)
        assert model.needs_refresh() is False

    def test_summary_dict_structure(self):
        model = FeeModel(vip_tier=1, bnb_discount=True)
        s = model.summary()
        assert "vip_tier" in s
        assert "maker_bps" in s
        assert "taker_bps" in s
        assert "bnb_discount" in s
        assert s["bnb_discount"] is True


# ---------------------------------------------------------------------------
# F16: FeeModel integration with OrderManager (paper mode)
# ---------------------------------------------------------------------------

class TestFeeModelIntegration:
    def test_fee_model_used_in_paper_order(self):
        fee_model = FeeModel.flat(rate_bps=5.0)   # 5bps instead of default 10bps
        m = OrderManager(
            exchange_config={"initial_cash": 100_000.0, "max_order_size": 10.0},
            paper_mode=True,
            fee_model=fee_model,
        )
        m.update_paper_price(1000.0)
        oid = m.submit_order("buy", 1.0, current_price=1000.0)
        order = m.get_order(oid)
        # fee = 1.0 qty × 1000 price × 0.0005 rate = 0.5
        assert order.fee == pytest.approx(1.0 * 1000.0 * 0.0005)

    def test_fallback_to_default_fee_on_model_error(self):
        bad_model = MagicMock()
        bad_model.compute_fee.side_effect = Exception("broken")
        m = OrderManager(
            exchange_config={"initial_cash": 100_000.0, "max_order_size": 10.0},
            paper_mode=True,
            fee_model=bad_model,
        )
        m.update_paper_price(1000.0)
        oid = m.submit_order("buy", 1.0, current_price=1000.0)
        order = m.get_order(oid)
        # Fallback: 0.1% of notional
        assert order.fee == pytest.approx(1.0 * 1000.0 * 0.001)


# ---------------------------------------------------------------------------
# F12 + F14: stop order with TTL
# ---------------------------------------------------------------------------

class TestStopOrderWithTTL:
    def test_stop_order_cancelled_when_ttl_expires_before_trigger(self):
        m = OrderManager(
            exchange_config={
                "initial_cash": 100_000.0,
                "max_order_size": 10.0,
                "order_ttl_sec": 0.1,
                "order_ttl_check_interval_sec": 0.05,
            },
            paper_mode=True,
        )
        m.update_paper_price(100.0)
        m.submit_order("buy", 1.0, current_price=100.0)
        oid = m.submit_order(
            "sell", 0.5,
            order_type="stop_loss_limit",
            stop_price=80.0, limit_price=79.0,
            current_price=100.0,
        )
        assert m.check_order(oid) == "pending"
        time.sleep(0.3)   # TTL expires; stop never triggered
        assert m.check_order(oid) == "cancelled"
        m.close()


# ---------------------------------------------------------------------------
# F13 + F12: partial fill + limit order interaction
# ---------------------------------------------------------------------------

class TestPartialFillLimitOrder:
    def test_partially_filled_limit_stays_alive(self):
        np.random.seed(123)
        # Force a partial fill by patching _draw_partial_fill_ratio
        m = make_manager(partial_fill_sim=True, partial_fill_min_ratio=0.5)
        m.update_paper_price(90.0)

        with patch.object(m, "_draw_partial_fill_ratio", return_value=0.5):
            oid = m.submit_order("buy", 1.0, order_type="limit", limit_price=95.0, current_price=90.0)
        order = m.get_order(oid)
        assert order.status == "partial"
        assert order.filled_amount == pytest.approx(0.5)
