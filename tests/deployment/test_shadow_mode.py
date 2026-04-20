"""
Week 68 (S61): Shadow mode tests for PaperTrader.

Shadow agent observes the same state as the main agent but never submits orders.
Its decisions are recorded to the audit log for post-hoc comparison only.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.paper_trader import PaperTrader
from deployment.audit.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyAgent:
    """Deterministic agent that always returns the same scalar action."""

    def __init__(self, action: float = 0.5) -> None:
        self._action = action
        self.call_count: int = 0

    def predict(self, obs, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        self.call_count += 1
        return np.array([self._action]), None


def _build_config() -> dict:
    return {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.99,
            "window_size": 5,
        },
        "monitoring": {},
    }


def _price_stream(n: int = 30) -> list:
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return prices.tolist()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShadowModeBasic:
    """Shadow agent is called every step but never submits orders."""

    def test_shadow_agent_called_each_step(self):
        main_agent = _DummyAgent(action=0.3)
        shadow_agent = _DummyAgent(action=-0.2)
        config = _build_config()
        prices = _price_stream(30)

        trader = PaperTrader(
            agent=main_agent,
            config=config,
            simulation_mode=True,
            shadow_agent=shadow_agent,
        )
        trader.run(price_stream=iter(prices))

        # Shadow agent should have been called at least once (after window fills)
        assert shadow_agent.call_count > 0

    def test_shadow_agent_does_not_affect_trades(self):
        """Running with vs without a shadow agent produces identical main trades."""
        main_prices = _price_stream(40)

        def _run(with_shadow: bool) -> dict:
            np.random.seed(0)  # ensure identical rng state
            main = _DummyAgent(action=0.4)
            shadow = _DummyAgent(action=-0.9) if with_shadow else None
            t = PaperTrader(
                agent=main,
                config=_build_config(),
                simulation_mode=True,
                shadow_agent=shadow,
            )
            return t.run(price_stream=iter(list(main_prices)))

        report_no_shadow = _run(False)
        report_with_shadow = _run(True)

        assert report_no_shadow["num_trades"] == report_with_shadow["num_trades"]
        assert abs(report_no_shadow["final_portfolio_value"] - report_with_shadow["final_portfolio_value"]) < 1e-6

    def test_no_shadow_agent_no_extra_orders(self):
        """Without a shadow agent, order_manager is only called by main agent logic."""
        main_agent = _DummyAgent(action=0.5)
        order_manager = MagicMock()
        order_manager.submit_order.return_value = "ord-1"
        order_manager.check_order.return_value = None
        order_manager.compute_latency_percentiles.return_value = {"p50": 0, "p95": 0, "p99": 0}

        config = _build_config()
        trader = PaperTrader(
            agent=main_agent,
            config=config,
            simulation_mode=True,
            order_manager=order_manager,
            shadow_agent=None,
        )
        trader.run(price_stream=iter(_price_stream(20)))
        # Any calls come only from main agent
        call_count_no_shadow = order_manager.submit_order.call_count

        # Now with a shadow agent — order_manager calls must NOT increase
        order_manager2 = MagicMock()
        order_manager2.submit_order.return_value = "ord-2"
        order_manager2.check_order.return_value = None
        order_manager2.compute_latency_percentiles.return_value = {"p50": 0, "p95": 0, "p99": 0}

        shadow = _DummyAgent(action=-0.9)
        trader2 = PaperTrader(
            agent=_DummyAgent(action=0.5),
            config=config,
            simulation_mode=True,
            order_manager=order_manager2,
            shadow_agent=shadow,
        )
        trader2.run(price_stream=iter(_price_stream(20)))
        call_count_with_shadow = order_manager2.submit_order.call_count

        assert call_count_with_shadow == call_count_no_shadow


class TestShadowModeAuditLog:
    """Shadow decisions must appear in audit log, tagged as source='shadow'."""

    def test_shadow_decisions_logged_to_audit(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file, fsync=False)

        main_agent = _DummyAgent(action=0.3)
        shadow_agent = _DummyAgent(action=-0.7)
        config = _build_config()

        trader = PaperTrader(
            agent=main_agent,
            config=config,
            simulation_mode=True,
            shadow_agent=shadow_agent,
            audit_logger=audit,
        )
        trader.run(price_stream=iter(_price_stream(30)))
        audit.close()

        # Parse audit log
        records = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        shadow_records = [
            r for r in records
            if r["type"] == "model_decision"
            # G2: source renamed from "shadow" → "canary_observe" / "canary_active"
            and r["payload"].get("source", "").startswith("canary")
        ]
        assert len(shadow_records) > 0, "No canary decisions found in audit log"

    def test_shadow_record_contains_comparison_fields(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file, fsync=False)

        trader = PaperTrader(
            agent=_DummyAgent(action=0.3),
            config=_build_config(),
            simulation_mode=True,
            shadow_agent=_DummyAgent(action=-0.7),
            audit_logger=audit,
        )
        trader.run(price_stream=iter(_price_stream(20)))
        audit.close()

        records = []
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        shadow_recs = [
            r for r in records
            if r["type"] == "model_decision"
            # G2: source renamed from "shadow" → "canary_observe" / "canary_active"
            and r["payload"].get("source", "").startswith("canary")
        ]
        assert len(shadow_recs) > 0
        rec = shadow_recs[0]["payload"]
        # Must contain comparison fields (G2: shadow_action → canary_action)
        assert "main_action" in rec
        assert "canary_action" in rec
        assert "step" in rec
        assert "obs_hash" in rec

    def test_shadow_failure_does_not_crash_trader(self):
        """If shadow agent raises, main trading loop must continue."""
        main_agent = _DummyAgent(action=0.3)

        class _BrokenAgent:
            def predict(self, obs, deterministic=True):
                raise RuntimeError("shadow broken")

        trader = PaperTrader(
            agent=main_agent,
            config=_build_config(),
            simulation_mode=True,
            shadow_agent=_BrokenAgent(),
        )
        report = trader.run(price_stream=iter(_price_stream(25)))
        # Main agent should still have traded normally
        assert report["steps"] > 0

    def test_shadow_audit_log_chain_valid(self, tmp_path):
        """Audit log including shadow entries must pass hash-chain verification."""
        import subprocess
        import sys

        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file, fsync=False)

        trader = PaperTrader(
            agent=_DummyAgent(action=0.4),
            config=_build_config(),
            simulation_mode=True,
            shadow_agent=_DummyAgent(action=-0.2),
            audit_logger=audit,
        )
        trader.run(price_stream=iter(_price_stream(30)))
        audit.close()

        verify_script = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "verify_audit_log.py"
        )
        result = subprocess.run(
            [sys.executable, verify_script, log_file],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Audit chain broken:\n{result.stdout}\n{result.stderr}"
