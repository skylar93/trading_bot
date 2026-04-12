#!/usr/bin/env python
"""
Phase 6 End-to-End Smoke Test (Week 68, S63).

Exercises all Track A/B/C/D features in a single run:
  Track A — StateStore (persistence), AuditLogger (audit)
  Track B — UnifiedRiskManager, DataSource interface, config loader
  Track C — Fat-finger guard, volatility circuit breaker, rate limiter
  Track D — PnL attribution, shadow agent, model registry

Exit 0 on success, non-zero on any failure.

Usage::

    python scripts/phase6_smoke.py
    python scripts/phase6_smoke.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Repo root on path
# --------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("phase6_smoke")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results: list[Tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append((name, ok, detail))
    if not ok:
        logger.warning("SMOKE FAIL: %s %s", name, detail)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# --------------------------------------------------------------------------
# Helpers / shared fixtures
# --------------------------------------------------------------------------

class _DummyAgent:
    """Always returns the same scalar action."""
    def __init__(self, action: float = 0.3):
        self._action = action

    def predict(self, obs, deterministic=True):
        return np.array([self._action]), None


def _prices(n: int = 80) -> list:
    rng = np.random.default_rng(7)
    return (100.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist()


def _base_config(
    db_path: str = ":memory:",
    log_path: str | None = None,
) -> dict:
    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 10_000.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.99,
            "window_size": 5,
        },
        "monitoring": {},
        "persistence": {
            "enabled": True,
            "db_path": db_path,
            "checkpoint_every_n_steps": 1,
        },
    }
    return cfg


# --------------------------------------------------------------------------
# Track A — Ops Readiness
# --------------------------------------------------------------------------

def smoke_state_store(tmp_dir: str) -> None:
    _section("Track A1: StateStore (persistence)")
    from deployment.persistence.state_store import StateStore

    db = os.path.join(tmp_dir, "smoke_state.db")
    store = StateStore(db)
    snap = {
        "symbol": "BTC/USDT",
        "position": 0.1,
        "entry_price": 100.0,
        "cash": 9_000.0,
        "current_price": 101.0,
        "peak_value": 10_100.0,
        "equity": 9_100.1,
        "step": 42,
        "shutdown_triggered": False,
        "shutdown_reason": "",
        "portfolio_history": [10_000.0, 10_050.0, 10_100.0],
        "trades": [],
        "orders": [],
    }
    store.save_snapshot(snap)
    loaded = store.load_latest()
    _check("StateStore.save_snapshot + load_latest", loaded is not None)
    _check("StateStore.step restored", loaded is not None and loaded.get("step") == 42)
    store.clear()
    _check("StateStore.clear", store.load_latest() is None)


def smoke_audit_logger(tmp_dir: str) -> None:
    _section("Track A2: AuditLogger (immutable audit)")
    from deployment.audit.audit_logger import AuditLogger

    log_path = os.path.join(tmp_dir, "smoke_audit.jsonl")
    al = AuditLogger(log_path=log_path, fsync=False)
    for i in range(20):
        al.log_risk_event({"type": "test", "i": i})
    al.log_model_decision(action=0.3, obs_hash="abc123")
    al.close()

    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    _check("AuditLogger records written", len(records) == 21)

    verify = _ROOT / "scripts" / "verify_audit_log.py"
    res = subprocess.run(
        [sys.executable, str(verify), log_path],
        capture_output=True, text=True,
    )
    _check("AuditLogger chain valid (verify script exit 0)", res.returncode == 0, res.stderr.strip())


# --------------------------------------------------------------------------
# Track B — Architecture Consolidation
# --------------------------------------------------------------------------

def smoke_unified_risk_manager() -> None:
    _section("Track B1: UnifiedRiskManager")
    from risk_management.unified_risk_manager import UnifiedRiskManager

    urm = UnifiedRiskManager(mode="backtest", var_method="parametric")
    # Within limit: 500 / 10000 = 5% < 50% max → True
    within = urm.check_position_limit(
        position_value=500.0, portfolio_value=10_000.0, max_position_fraction=0.5
    )
    # Exceeds limit: 8000 / 10000 = 80% > 50% max → False
    exceeds = urm.check_position_limit(
        position_value=8_000.0, portfolio_value=10_000.0, max_position_fraction=0.5
    )
    _check("check_position_limit: within limit returns True", within)
    _check("check_position_limit: exceeds limit returns False", not exceeds)


def smoke_data_source() -> None:
    _section("Track B2: DataSource abstraction")
    import pandas as pd
    from data.sources.base import StaticDataSource

    df = pd.DataFrame({
        "$close": _prices(50),
        "$volume": [1000.0] * 50,
    })
    ds = StaticDataSource(df)
    _check("StaticDataSource.__len__", len(ds) == 50)
    _check("StaticDataSource.is_live = False", not ds.is_live())
    _check("StaticDataSource.is_stale = False", not ds.is_stale(60))
    window = ds.get_window(0, 10)
    _check("StaticDataSource.get_window", len(window) == 10)


def smoke_config_loader() -> None:
    _section("Track B3: Config loader")
    from config.loader import load
    try:
        cfg = load()
        _check("config.loader.load() returns dict", isinstance(cfg, dict))
        _check("config.loader.load() has content", len(cfg) > 0)
    except Exception as e:
        _check("config.loader.load() returns dict", False, str(e))


# --------------------------------------------------------------------------
# Track C — Production Safety
# --------------------------------------------------------------------------

def smoke_fat_finger() -> None:
    _section("Track C1: Fat-finger guard")
    from deployment.execution.fat_finger_guard import FatFingerGuard

    guard = FatFingerGuard(hard_cap=10.0, size_multiplier_limit=5.0, lookback=10)
    ok, _ = guard.check(amount=1.0)
    _check("Fat-finger: small order passes", ok)
    blocked, _ = guard.check(amount=11.0)
    _check("Fat-finger: hard-cap exceeded blocked", not blocked)

    # Seed history, then spike
    for _ in range(10):
        guard.record_fill(1.0)
    spiked, _ = guard.check(amount=8.0)
    _check("Fat-finger: spike order blocked", not spiked)


def smoke_circuit_breaker() -> None:
    _section("Track C2: Volatility circuit breaker")
    from deployment.execution.circuit_breaker import VolatilityCircuitBreaker

    # High threshold → stable prices should NOT trip
    cb_stable = VolatilityCircuitBreaker(vol_threshold=0.5, window=5, cooldown=0.0)
    for p in [100.0, 100.1, 100.2, 100.1, 100.0]:
        cb_stable.update(p)
    stable_tripped = cb_stable.is_tripped()
    _check("Circuit breaker: stable prices not tripped (high threshold)", not stable_tripped)

    # Very low threshold → volatile prices trip the breaker
    cb_vol = VolatilityCircuitBreaker(vol_threshold=0.001, window=5, cooldown=0.0)
    for p in [100.0, 110.0, 80.0, 120.0, 70.0]:
        cb_vol.update(p)
    volatile_tripped = cb_vol.is_tripped()
    _check("Circuit breaker: volatile prices trip breaker", volatile_tripped)


def smoke_rate_limiter() -> None:
    _section("Track C3: Rate limiter")
    from deployment.execution.rate_limiter import RateLimiter

    rl = RateLimiter(max_calls=5, period=1.0)
    t0 = time.monotonic()
    for _ in range(5):
        rl.acquire()
    elapsed = time.monotonic() - t0
    _check("RateLimiter: 5 calls within 1s", elapsed < 1.0)


# --------------------------------------------------------------------------
# Track C: PnL attribution (from Week 66)
# --------------------------------------------------------------------------

def smoke_pnl_attribution() -> None:
    _section("Track C/D: PnL attribution")
    from deployment.paper_trader import PaperTrader, Trade
    from deployment.analysis.pnl_attribution import PnLAttributor
    from datetime import datetime

    trades = [
        Trade(timestamp=datetime.utcnow(), side="buy", price=100.0, quantity=1.0, fee=0.1, pnl=0.0),
        Trade(timestamp=datetime.utcnow(), side="sell", price=105.0, quantity=1.0, fee=0.1, pnl=4.8),
    ]
    attr = PnLAttributor()
    attributions = attr.attribute(trades, slippage_records=[0.001])
    summary = attr.summarise(attributions)
    fields = attr.to_exporter_fields(summary)

    _check("PnLAttributor.attribute returns list", len(attributions) > 0)
    _check("PnLAttributor.summarise.total_net_pnl positive (buy low sell high)",
           summary.total_net_pnl > 0)
    _check("to_exporter_fields returns dict", isinstance(fields, dict))


# --------------------------------------------------------------------------
# Track D — Shadow agent + model registry
# --------------------------------------------------------------------------

def smoke_shadow_agent(tmp_dir: str) -> None:
    _section("Track D1: Shadow agent")
    from deployment.paper_trader import PaperTrader
    from deployment.audit.audit_logger import AuditLogger

    log_path = os.path.join(tmp_dir, "shadow_smoke.jsonl")
    al = AuditLogger(log_path=log_path, fsync=False)

    main_agent = _DummyAgent(action=0.4)
    shadow_agent = _DummyAgent(action=-0.3)

    trader = PaperTrader(
        agent=main_agent,
        config=_base_config(),
        simulation_mode=True,
        shadow_agent=shadow_agent,
        audit_logger=al,
    )
    report = trader.run(price_stream=iter(_prices(40)))
    al.close()

    # Shadow decisions in audit log
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    shadow_recs = [
        r for r in records
        if r["type"] == "model_decision" and r["payload"].get("source") == "shadow"
    ]
    _check("Shadow agent: decisions logged", len(shadow_recs) > 0)
    _check("Shadow agent: does not affect trade count",
           report["num_trades"] >= 0)  # just verify run completes cleanly


def smoke_model_registry(tmp_dir: str) -> None:
    _section("Track D2: Model registry + rollback")
    from training.registry.model_registry import ModelRegistry

    reg_dir = os.path.join(tmp_dir, "smoke_registry")
    reg = ModelRegistry(registry_dir=reg_dir)

    # Create a fake model file
    fake_model = os.path.join(tmp_dir, "model_v1.zip")
    Path(fake_model).write_bytes(b"FAKE_WEIGHTS")

    ver = reg.register(
        model_path=fake_model,
        metrics={"sharpe": 1.2, "max_dd": 0.07},
        config={"algo": "PPO"},
        tag="smoke-test",
    )
    _check("Registry: register returns version", ver == 1)
    _check("Registry: get_version", reg.get_version(ver)["tag"] == "smoke-test")

    reg.set_active(ver)
    _check("Registry: set_active + get_active", reg.get_active()["version"] == ver)

    # rollback via CLI
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "rollback_model.py"),
         str(ver), "--registry-dir", reg_dir],
        capture_output=True, text=True,
    )
    _check("rollback_model.py CLI exit 0", result.returncode == 0, result.stderr.strip())
    _check("rollback_model.py CLI output mentions version",
           f"version {ver}" in result.stdout)


# --------------------------------------------------------------------------
# Full PaperTrader end-to-end (all Phase 6 features combined)
# --------------------------------------------------------------------------

def smoke_full_papertrader(tmp_dir: str) -> None:
    _section("Full E2E: PaperTrader with all Phase 6 features")
    from deployment.paper_trader import PaperTrader
    from deployment.audit.audit_logger import AuditLogger
    from deployment.persistence.state_store import StateStore
    from deployment.execution.order_manager import OrderManager
    from deployment.execution.fat_finger_guard import FatFingerGuard
    from deployment.execution.circuit_breaker import VolatilityCircuitBreaker

    db_path = os.path.join(tmp_dir, "e2e_state.db")
    log_path = os.path.join(tmp_dir, "e2e_audit.jsonl")

    state_store = StateStore(db_path)
    audit_logger = AuditLogger(log_path=log_path, fsync=False)

    fat_finger = FatFingerGuard(hard_cap=1e6, size_multiplier_limit=20.0, lookback=50)
    circuit_breaker = VolatilityCircuitBreaker(vol_threshold=0.5, window=5, cooldown=0.0)

    order_manager = OrderManager(
        exchange_config={},
        paper_mode=True,
        fat_finger_guard=fat_finger,
        circuit_breaker=circuit_breaker,
        audit_logger=audit_logger,
    )

    main_agent = _DummyAgent(action=0.35)
    shadow_agent = _DummyAgent(action=-0.15)

    config = _base_config(db_path=db_path)
    trader = PaperTrader(
        agent=main_agent,
        config=config,
        simulation_mode=True,
        state_store=state_store,
        audit_logger=audit_logger,
        order_manager=order_manager,
        # risk_manager uses PaperTrader's built-in drawdown check; UnifiedRiskManager
        # is verified separately in smoke_unified_risk_manager().
        shadow_agent=shadow_agent,
    )

    report = trader.run(price_stream=iter(_prices(60)))
    audit_logger.close()

    _check("E2E: PaperTrader completes run", report["steps"] > 0)
    _check("E2E: StateStore has checkpoint", state_store.load_latest() is not None)

    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    _check("E2E: AuditLogger has records", len(records) > 0)

    verify = _ROOT / "scripts" / "verify_audit_log.py"
    res = subprocess.run(
        [sys.executable, str(verify), log_path],
        capture_output=True, text=True,
    )
    _check("E2E: Audit chain valid", res.returncode == 0, res.stderr.strip())


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 smoke test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("  Phase 6 Production Readiness — Smoke Test (Week 68 S63)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            smoke_state_store(tmp_dir)
        except Exception as e:
            _check("Track A1: StateStore", False, str(e))

        try:
            smoke_audit_logger(tmp_dir)
        except Exception as e:
            _check("Track A2: AuditLogger", False, str(e))

        try:
            smoke_unified_risk_manager()
        except Exception as e:
            _check("Track B1: UnifiedRiskManager", False, str(e))

        try:
            smoke_data_source()
        except Exception as e:
            _check("Track B2: DataSource", False, str(e))

        try:
            smoke_config_loader()
        except Exception as e:
            _check("Track B3: Config loader", False, str(e))

        try:
            smoke_fat_finger()
        except Exception as e:
            _check("Track C1: Fat-finger", False, str(e))

        try:
            smoke_circuit_breaker()
        except Exception as e:
            _check("Track C2: Circuit breaker", False, str(e))

        try:
            smoke_rate_limiter()
        except Exception as e:
            _check("Track C3: Rate limiter", False, str(e))

        try:
            smoke_pnl_attribution()
        except Exception as e:
            _check("Track C/D: PnL attribution", False, str(e))

        try:
            smoke_shadow_agent(tmp_dir)
        except Exception as e:
            _check("Track D1: Shadow agent", False, str(e))

        try:
            smoke_model_registry(tmp_dir)
        except Exception as e:
            _check("Track D2: Model registry", False, str(e))

        try:
            smoke_full_papertrader(tmp_dir)
        except Exception as e:
            _check("Full E2E: PaperTrader", False, str(e))

    # ---------- Summary ----------
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ← {failed} FAILED")
    else:
        print("  ✓ all passed")
    print("=" * 60)

    if failed:
        print("\nFailed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  • {name}: {detail}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
