"""
A6 Cost Decomposition tests (Week 90).

Covers:
  - decompose(): algebraic identity (total == sum of 4 axes)
  - decompose(): sell signal_pnl matches (fill - entry) * qty
  - decompose(): buy has zero signal_pnl
  - decompose(): slippage direction (fill > mid → negative for buy)
  - decompose(): fee_pnl and funding_pnl signs
  - CostDecomposer: accumulation and daily summary
  - CostDecomposer: cumulative summary totals
  - CostDecomposer.from_audit_log(): parses fill records from JSONL
  - Regression (A6.5): 4-axis sum matches realized P&L within $0.01
  - verify_decomposition_identity() helper
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from deployment.analysis.cost_decomposition import (
    CostDecomposer,
    FillDecomposition,
    FillRecord,
    decompose,
    verify_decomposition_identity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(hour: int = 12, minute: int = 0) -> datetime:
    return datetime(2026, 4, 28, hour, minute, 0, tzinfo=timezone.utc)


def _sell_fill(
    fill_price: float = 110.0,
    entry_price: float = 100.0,
    qty: float = 1.0,
    mid: float = 109.5,
    fill_id: str = "f_sell",
    ts: datetime = None,
) -> FillRecord:
    return FillRecord(
        fill_id=fill_id,
        timestamp=ts or _ts(),
        side="sell",
        fill_price=fill_price,
        qty=qty,
        entry_price=entry_price,
        mid_at_submit=mid,
    )


def _buy_fill(
    fill_price: float = 100.5,
    qty: float = 1.0,
    mid: float = 100.0,
    fill_id: str = "f_buy",
    ts: datetime = None,
) -> FillRecord:
    return FillRecord(
        fill_id=fill_id,
        timestamp=ts or _ts(10),
        side="buy",
        fill_price=fill_price,
        qty=qty,
        entry_price=0.0,
        mid_at_submit=mid,
    )


# ---------------------------------------------------------------------------
# decompose(): algebraic identity
# ---------------------------------------------------------------------------

class TestDecomposeIdentity:
    def test_sell_4axis_sum_equals_total_pnl(self):
        """total_pnl must equal the algebraic sum of the 4 axes."""
        fill = _sell_fill(fill_price=110.0, entry_price=100.0, qty=2.0, mid=109.0)
        d = decompose(fill, fee_paid=0.22, funding_accrued=0.0)
        assert abs(d.total_pnl - (d.signal_pnl + d.slippage_pnl + d.fee_pnl + d.funding_pnl)) < 1e-9

    def test_buy_4axis_sum_equals_total_pnl(self):
        fill = _buy_fill(fill_price=100.5, qty=1.0, mid=100.0)
        d = decompose(fill, fee_paid=0.10)
        assert abs(d.total_pnl - (d.signal_pnl + d.slippage_pnl + d.fee_pnl + d.funding_pnl)) < 1e-9

    def test_with_funding(self):
        fill = _sell_fill()
        d = decompose(fill, fee_paid=0.11, funding_accrued=0.05)
        assert abs(d.total_pnl - (d.signal_pnl + d.slippage_pnl + d.fee_pnl + d.funding_pnl)) < 1e-9
        assert d.funding_pnl == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# decompose(): sell semantics
# ---------------------------------------------------------------------------

class TestDecomposeSell:
    def test_sell_signal_pnl_is_mid_minus_entry_times_qty(self):
        # signal_pnl = (mid_at_submit - entry_price) * qty
        fill = _sell_fill(fill_price=110.0, entry_price=100.0, qty=1.0, mid=109.0)
        d = decompose(fill, fee_paid=0.0)
        assert d.signal_pnl == pytest.approx((109.0 - 100.0) * 1.0)

    def test_sell_slippage_pnl_is_fill_minus_mid_times_qty(self):
        # slippage_pnl = (fill_price - mid_at_submit) * qty
        fill = _sell_fill(fill_price=110.0, entry_price=100.0, qty=1.0, mid=109.0)
        d = decompose(fill, fee_paid=0.0)
        assert d.slippage_pnl == pytest.approx((110.0 - 109.0) * 1.0)

    def test_sell_total_equals_net_realized_pnl(self):
        """For a sell: total_pnl should equal (fill - entry)*qty - fee."""
        fill = _sell_fill(fill_price=110.0, entry_price=100.0, qty=2.0, mid=109.0)
        fee = 0.22
        d = decompose(fill, fee_paid=fee)
        expected_net = (110.0 - 100.0) * 2.0 - fee
        assert d.total_pnl == pytest.approx(expected_net)

    def test_sell_fee_pnl_is_negative(self):
        fill = _sell_fill()
        d = decompose(fill, fee_paid=1.5)
        assert d.fee_pnl == pytest.approx(-1.5)

    def test_sell_slippage_negative_when_fill_below_mid(self):
        # fill < mid for a sell → you got less than mid → negative slippage
        fill = _sell_fill(fill_price=108.0, mid=109.0)
        d = decompose(fill, fee_paid=0.0)
        assert d.slippage_pnl < 0


# ---------------------------------------------------------------------------
# decompose(): buy semantics
# ---------------------------------------------------------------------------

class TestDecomposeBuy:
    def test_buy_signal_pnl_is_zero(self):
        fill = _buy_fill(fill_price=100.5, mid=100.0)
        d = decompose(fill, fee_paid=0.10)
        assert d.signal_pnl == 0.0

    def test_buy_slippage_negative_when_fill_above_mid(self):
        # fill > mid for a buy → you paid more than mid → negative slippage
        fill = _buy_fill(fill_price=101.0, mid=100.0, qty=1.0)
        d = decompose(fill, fee_paid=0.0)
        assert d.slippage_pnl == pytest.approx(-(101.0 - 100.0) * 1.0)
        assert d.slippage_pnl < 0

    def test_buy_slippage_positive_when_fill_below_mid(self):
        # fill < mid for a buy → you paid less than mid → positive slippage
        fill = _buy_fill(fill_price=99.5, mid=100.0, qty=1.0)
        d = decompose(fill, fee_paid=0.0)
        assert d.slippage_pnl > 0


# ---------------------------------------------------------------------------
# CostDecomposer: accumulation and summaries
# ---------------------------------------------------------------------------

class TestCostDecomposer:
    def test_add_and_cumulative(self):
        cd = CostDecomposer()
        fill1 = _sell_fill(ts=_ts(10))
        fill2 = _buy_fill(ts=_ts(11))
        d1 = cd.add_fill(fill1, fee_paid=0.11)
        d2 = cd.add_fill(fill2, fee_paid=0.10)

        summary = cd.cumulative_summary()
        assert summary.num_fills == 2
        assert summary.total_pnl == pytest.approx(d1.total_pnl + d2.total_pnl)
        assert summary.total_fee_pnl == pytest.approx(-0.11 + -0.10)

    def test_daily_summary_groups_by_date(self):
        cd = CostDecomposer()
        ts_today = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)
        ts_yesterday = datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc)

        cd.add_fill(_sell_fill(ts=ts_today), fee_paid=0.11)
        cd.add_fill(_sell_fill(fill_id="f2", ts=ts_today), fee_paid=0.05)
        cd.add_fill(_sell_fill(fill_id="f3", ts=ts_yesterday), fee_paid=0.08)

        today_s = cd.daily_summary(date(2026, 4, 28))
        yesterday_s = cd.daily_summary(date(2026, 4, 27))

        assert today_s is not None
        assert yesterday_s is not None
        assert today_s.num_fills == 2
        assert yesterday_s.num_fills == 1
        assert today_s.total_fee_pnl == pytest.approx(-0.16)

    def test_daily_summary_returns_none_for_empty_date(self):
        cd = CostDecomposer()
        result = cd.daily_summary(date(2099, 1, 1))
        assert result is None

    def test_all_daily_summaries_sorted(self):
        cd = CostDecomposer()
        for day in [28, 27, 26]:
            ts = datetime(2026, 4, day, 12, 0, tzinfo=timezone.utc)
            cd.add_fill(_sell_fill(fill_id=f"f_{day}", ts=ts), fee_paid=0.01)

        summaries = cd.all_daily_summaries()
        dates = [s.date for s in summaries]
        assert dates == sorted(dates)
        assert len(dates) == 3


# ---------------------------------------------------------------------------
# CostDecomposer.from_audit_log()
# ---------------------------------------------------------------------------

class TestFromAuditLog:
    def _write_audit(self, tmp_path: Path, records: list) -> Path:
        p = tmp_path / "audit.jsonl"
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return p

    def _fill_record(self, side: str, price: float, qty: float, fee: float,
                     pnl: float = 0.0, entry_price: float = 0.0,
                     mid_price: float = None) -> dict:
        return {
            "ts": "2026-04-28T12:00:00Z",
            "type": "fill",
            "payload": {
                "fill_id": f"fill_{side}_{price}",
                "side": side,
                "price": price,
                "mid_price": mid_price if mid_price is not None else price,
                "quantity": qty,
                "fee": fee,
                "pnl": pnl,
                "entry_price": entry_price,
            },
            "hash": "abc",
        }

    def test_loads_fill_records(self, tmp_path):
        path = self._write_audit(tmp_path, [
            self._fill_record("buy", 100.0, 1.0, 0.10),
            self._fill_record("sell", 110.0, 1.0, 0.11, pnl=10.0, entry_price=100.0),
        ])
        cd = CostDecomposer.from_audit_log(path)
        assert len(cd.fills()) == 2

    def test_ignores_non_fill_records(self, tmp_path):
        path = self._write_audit(tmp_path, [
            {"ts": "2026-04-28T12:00:00Z", "type": "risk_event", "payload": {}, "hash": "x"},
            self._fill_record("buy", 100.0, 1.0, 0.10),
        ])
        cd = CostDecomposer.from_audit_log(path)
        assert len(cd.fills()) == 1

    def test_returns_empty_for_missing_file(self, tmp_path):
        cd = CostDecomposer.from_audit_log(tmp_path / "nonexistent.jsonl")
        assert len(cd.fills()) == 0

    def test_sell_total_pnl_from_audit_log(self, tmp_path):
        path = self._write_audit(tmp_path, [
            self._fill_record("sell", 110.0, 2.0, 0.22,
                              pnl=20.0, entry_price=100.0, mid_price=109.0),
        ])
        cd = CostDecomposer.from_audit_log(path)
        fills = cd.fills()
        assert len(fills) == 1
        d = fills[0]
        # total_pnl == (fill - entry) * qty - fee = (110-100)*2 - 0.22 = 19.78
        assert d.total_pnl == pytest.approx(19.78)


# ---------------------------------------------------------------------------
# A6.5 Regression: 4-axis sum == realized P&L (< $0.01)
# ---------------------------------------------------------------------------

class TestRegressionIdentity:
    """4-axis sum must equal (fill_price - entry_price) * qty - fee for sells."""

    @pytest.mark.parametrize("fill_price,entry_price,qty,fee,mid", [
        (110.0, 100.0, 1.0, 0.11, 109.0),
        (50000.0, 45000.0, 0.01, 5.0, 49900.0),
        (1.05, 1.00, 1000.0, 0.50, 1.04),
        (100.0, 100.0, 1.0, 0.10, 100.0),  # breakeven trade
        (90.0, 100.0, 1.0, 0.09, 91.0),    # losing trade
    ])
    def test_sell_regression(self, fill_price, entry_price, qty, fee, mid):
        fill = FillRecord(
            fill_id="r",
            timestamp=_ts(),
            side="sell",
            fill_price=fill_price,
            qty=qty,
            entry_price=entry_price,
            mid_at_submit=mid,
        )
        d = decompose(fill, fee_paid=fee)
        realized_net = (fill_price - entry_price) * qty - fee
        assert verify_decomposition_identity(d, realized_net, tol=0.01), (
            f"Identity failed: 4-axis={d.total_pnl:.6f}, realized={realized_net:.6f}"
        )

    def test_decomposer_cumulative_matches_sum_of_fills(self):
        """CostDecomposer total must equal sum of individual decompose() calls."""
        cd = CostDecomposer()
        fills_data = [
            (_sell_fill(fill_price=110.0, entry_price=100.0, qty=1.0, mid=109.5, fill_id="s1"), 0.11),
            (_buy_fill(fill_price=100.5, qty=1.0, mid=100.0, fill_id="b1"), 0.10),
            (_sell_fill(fill_price=120.0, entry_price=105.0, qty=0.5, mid=119.0, fill_id="s2"), 0.06),
        ]
        expected_total = 0.0
        for fill, fee in fills_data:
            d = decompose(fill, fee_paid=fee)
            expected_total += d.total_pnl
            cd.add(d)

        summary = cd.cumulative_summary()
        assert abs(summary.total_pnl - expected_total) < 1e-9
