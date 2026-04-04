"""Test PositionTracker thread safety and correctness."""
import threading
import pytest
from deployment.execution.position_tracker import PositionTracker


class TestPositionTracker:
    def test_buy_sell_roundtrip(self):
        pt = PositionTracker(initial_cash=10000.0)
        pt.apply_buy(quantity=1.0, price=100.0, fee=0.1)
        assert pt.position == 1.0
        pt.apply_sell(quantity=1.0, price=110.0, fee=0.1)
        assert abs(pt.position) < 1e-9
        assert pt.cash > 10000.0  # profit minus fees

    def test_thread_safety(self):
        pt = PositionTracker(initial_cash=100000.0)
        def buy_loop():
            for _ in range(500):
                pt.apply_buy(quantity=0.01, price=100.0, fee=0.0)
        def sell_loop():
            for _ in range(500):
                pt.apply_sell(quantity=0.01, price=100.0, fee=0.0)
        t1 = threading.Thread(target=buy_loop)
        t2 = threading.Thread(target=sell_loop)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert abs(pt.position) < 0.01, f"Position should be ~0, got {pt.position}"

    def test_snapshot_restore(self):
        pt = PositionTracker(initial_cash=10000.0)
        pt.apply_buy(quantity=1.0, price=100.0, fee=0.1)
        snap = pt.snapshot()
        pt.apply_buy(quantity=1.0, price=200.0, fee=0.1)
        pt.restore(snap)
        assert pt.position == 1.0
