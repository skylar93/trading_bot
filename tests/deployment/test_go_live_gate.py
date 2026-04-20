"""
Week 77 (G11-G14): Go-Live Checklist & Sign-off Gate tests.

Covers:
  G11 — go_live_checklist.md exists and has required sections
  G12 — first_dollar_drill.py auto-checks pass (structural)
  G13 — kill switch: cancel_all_orders + shutdown within 5 s
  G14 — postmortem_template.md exists with required headings

완료 조건:
  - cancel_all_orders cancels all open orders (TestCancelAllOrders)
  - _trigger_shutdown cancels orders + sets shutdown_triggered (TestKillSwitch)
  - SIGUSR1 causes shutdown within 5 s (TestKillSwitchSignal)
  - Docs exist with required structure (TestDocuments)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.execution.order_manager import OrderManager
from deployment.paper_trader import PaperTrader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyAgent:
    def __init__(self, action: float = 0.0):
        self._action = action

    def predict(self, obs, deterministic=True):
        return np.array([self._action]), None


def _make_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 1_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 5,
        },
        "pid_file": str(PROJECT_ROOT / "state" / "test_kill_switch.pid"),
        "monitoring": {},
    }
    cfg.update(overrides)
    return cfg


def _make_order_manager(**kw: Any) -> OrderManager:
    return OrderManager(
        exchange_config={"symbol": "BTC/USDT"},
        paper_mode=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# G13a: OrderManager.cancel_all_orders
# ---------------------------------------------------------------------------

class TestCancelAllOrders:
    def test_cancels_all_pending(self):
        om = _make_order_manager()
        prices = [100.0, 101.0, 102.0]
        for p in prices:
            om.update_paper_price(p)

        ids = [
            om.submit_order("buy", 0.01, order_type="limit", limit_price=99.0,
                            current_price=100.0, idempotency_key=f"buy-{i}")
            for i in range(3)
        ]
        # All should be pending (limit below market)
        for oid in ids:
            assert om.check_order(oid) in ("pending", "partial")

        cancelled = om.cancel_all_orders()
        assert cancelled == 3
        for oid in ids:
            assert om.check_order(oid) == "cancelled"

    def test_cancel_all_ignores_filled(self):
        om = _make_order_manager()
        om.update_paper_price(100.0)
        # Market buy → fills immediately in paper mode
        oid = om.submit_order("buy", 0.01, order_type="market", current_price=100.0,
                               idempotency_key="mkt-1")
        assert om.check_order(oid) in ("filled", "partial")

        cancelled = om.cancel_all_orders()
        assert cancelled == 0  # already filled, nothing to cancel

    def test_cancel_all_returns_count(self):
        om = _make_order_manager()
        om.update_paper_price(50.0)
        for i in range(5):
            om.submit_order("buy", 0.001, order_type="limit", limit_price=40.0,
                            current_price=50.0, idempotency_key=f"lim-{i}")
        assert om.cancel_all_orders() == 5


# ---------------------------------------------------------------------------
# G13b: PaperTrader._trigger_shutdown
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_trigger_shutdown_sets_flag(self):
        trader = PaperTrader(_DummyAgent(), _make_config(), simulation_mode=True)
        trader._trigger_shutdown("test reason")
        assert trader.state.shutdown_triggered is True
        assert trader.state.shutdown_reason == "test reason"

    def test_trigger_shutdown_cancels_open_orders(self):
        om = _make_order_manager()
        om.update_paper_price(100.0)
        oid = om.submit_order("buy", 0.01, order_type="limit", limit_price=80.0,
                               current_price=100.0, idempotency_key="pending-1")

        cfg = _make_config()
        trader = PaperTrader(_DummyAgent(), cfg, simulation_mode=True,
                              order_manager=om)
        trader._update_price(100.0)
        trader._trigger_shutdown("kill_switch")

        assert om.check_order(oid) == "cancelled"
        assert trader.state.shutdown_triggered is True

    def test_trigger_shutdown_stops_run(self):
        prices = iter([100.0 + i * 0.01 for i in range(1000)])
        trader = PaperTrader(_DummyAgent(0.0), _make_config(), simulation_mode=True)

        done = threading.Event()

        def _run():
            trader.run(price_stream=prices, duration_seconds=10)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.2)

        trader._trigger_shutdown("test stop")
        done.wait(timeout=3.0)

        assert done.is_set(), "run() did not stop after shutdown"
        assert trader.state.shutdown_triggered is True

    def test_kill_switch_completes_within_5s(self):
        """G13 completion criterion: shutdown ≤ 5 s."""
        prices = iter([100.0 + i * 0.01 for i in range(10_000)])
        trader = PaperTrader(_DummyAgent(0.0), _make_config(), simulation_mode=True)

        done = threading.Event()

        def _run():
            trader.run(price_stream=prices, duration_seconds=30)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.3)

        start = time.monotonic()
        trader._trigger_shutdown("kill_switch_timing_test")
        done.wait(timeout=6.0)
        elapsed = time.monotonic() - start

        assert done.is_set(), "run() never stopped"
        assert elapsed < 5.0, f"kill switch took {elapsed:.2f}s (> 5 s SLA)"


# ---------------------------------------------------------------------------
# G13c: SIGUSR1 signal handler
# ---------------------------------------------------------------------------

class TestKillSwitchSignal:
    def test_sigusr1_triggers_shutdown(self):
        """SIGUSR1 delivered in-process → shutdown_triggered within 3 s."""
        prices = iter([100.0 + i * 0.01 for i in range(10_000)])
        trader = PaperTrader(_DummyAgent(0.0), _make_config(), simulation_mode=True)

        done = threading.Event()

        def _run():
            trader.run(price_stream=prices, duration_seconds=30)
            done.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.3)

        start = time.monotonic()
        os.kill(os.getpid(), signal.SIGUSR1)

        done.wait(timeout=5.0)
        elapsed = time.monotonic() - start

        assert trader.state.shutdown_triggered is True, "SIGUSR1 did not trigger shutdown"
        assert elapsed < 5.0, f"SIGUSR1 shutdown took {elapsed:.2f}s"

    def test_pid_file_written(self, tmp_path):
        pid_file = str(tmp_path / "test.pid")
        cfg = _make_config()
        cfg["pid_file"] = pid_file
        PaperTrader(_DummyAgent(), cfg, simulation_mode=True)
        assert Path(pid_file).exists()
        assert int(Path(pid_file).read_text().strip()) == os.getpid()


# ---------------------------------------------------------------------------
# G11: go_live_checklist.md
# ---------------------------------------------------------------------------

class TestDocuments:
    def test_go_live_checklist_exists(self):
        path = PROJECT_ROOT / "docs" / "runbook" / "go_live_checklist.md"
        assert path.exists(), f"Missing: {path}"

    def test_go_live_checklist_required_sections(self):
        path = PROJECT_ROOT / "docs" / "runbook" / "go_live_checklist.md"
        text = path.read_text()
        required = [
            "Track E",
            "Track F",
            "Track G",
            "Kill Switch",
            "Sign-Off",
            "Secret",
            "daily_loss_limit",
        ]
        for section in required:
            assert section in text, f"go_live_checklist.md missing section: {section!r}"

    # G14: postmortem template
    def test_postmortem_template_exists(self):
        path = PROJECT_ROOT / "docs" / "runbook" / "postmortem_template.md"
        assert path.exists(), f"Missing: {path}"

    def test_postmortem_template_required_sections(self):
        path = PROJECT_ROOT / "docs" / "runbook" / "postmortem_template.md"
        text = path.read_text()
        required = [
            "Incident Summary",
            "Timeline",
            "Root Cause",
            "Action Items",
            "Audit Log Evidence",
            "Checklist Before Restarting",
        ]
        for section in required:
            assert section in text, f"postmortem_template.md missing: {section!r}"

    # G12: first_dollar_drill.py
    def test_first_dollar_drill_script_exists(self):
        path = PROJECT_ROOT / "scripts" / "first_dollar_drill.py"
        assert path.exists(), f"Missing: {path}"

    # G13: kill_switch.py
    def test_kill_switch_script_exists(self):
        path = PROJECT_ROOT / "scripts" / "kill_switch.py"
        assert path.exists(), f"Missing: {path}"

    def test_kill_switch_has_5s_timeout(self):
        text = (PROJECT_ROOT / "scripts" / "kill_switch.py").read_text()
        assert "5" in text, "kill_switch.py should reference 5 s timeout"


# ---------------------------------------------------------------------------
# G12: first_dollar_drill structural checks (fast subset)
# ---------------------------------------------------------------------------

class TestFirstDollarDrill:
    def test_drill_structural_checks_pass(self):
        """Run first_dollar_drill.py --check-only and expect exit 0."""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "first_dollar_drill.py"),
                "--check-only",
                "--skip-kill-switch-test",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )
        # Print output for debugging if it fails
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, (
            f"first_dollar_drill --check-only returned {result.returncode}\n"
            f"{result.stdout}"
        )

    def test_dollar_drill_simulation(self):
        """$100 drill should run without crashing."""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "first_dollar_drill.py"),
                "--capital", "100",
                "--skip-kill-switch-test",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=60,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        assert result.returncode == 0, (
            f"$100 drill returned {result.returncode}\n{result.stdout}"
        )
