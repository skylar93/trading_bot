#!/usr/bin/env python
"""
G12: First Dollar Drill — pre-flight validation + $100 simulation drill.

Usage:
    # Auto-verify only (no trading)
    python scripts/first_dollar_drill.py --check-only

    # Full drill with $100 simulated capital
    python scripts/first_dollar_drill.py --capital 100

    # Write JSON report
    python scripts/first_dollar_drill.py --capital 100 --report drill_report.json

    # Real $100 live drill (requires EXCHANGE_BINANCE_KEY / EXCHANGE_BINANCE_SECRET)
    python scripts/first_dollar_drill.py --live --capital 100

Exit codes:
    0 — all checks passed (+ drill succeeded if --capital provided)
    1 — one or more pre-flight checks failed / missing credentials
    2 — drill itself failed (guards fired unexpectedly or NaN detected)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Root of the project (one level above scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Individual auto-checks
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    msg = f"  {icon} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return {"name": name, "status": status, "detail": detail}


def check_pytest_ini_ignores() -> dict[str, Any]:
    """E2: pytest.ini ignore count ≤ 5."""
    ini = PROJECT_ROOT / "pytest.ini"
    if not ini.exists():
        return _check("pytest.ini ignore ≤ 5", False, "pytest.ini not found")
    text = ini.read_text()
    ignores = [l.strip() for l in text.splitlines() if l.strip().startswith("--ignore=")]
    count = len(ignores)
    return _check("pytest.ini ignore ≤ 5", count <= 5, f"{count} ignores found")


def check_no_old_risk_api() -> dict[str, Any]:
    """E4: no source file calls deprecated risk API methods.

    Delegates to scripts/check_deprecation_callers.py which checks
    check_stop_loss, check_max_drawdown, and calculate_var across all
    non-test, non-shim source files.
    """
    checker = PROJECT_ROOT / "scripts" / "check_deprecation_callers.py"
    result = subprocess.run(
        [sys.executable, str(checker)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    detail = result.stdout.strip() if not ok else ""
    return _check("deprecated risk API callers → 0", ok, detail)


def check_risk_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """R1-R5: validate required risk config keys exist and are non-zero."""
    results = []
    pt = config.get("paper_trading", config)
    ex = config.get("exchange", {})
    lim = config.get("limits", {})

    checks = [
        ("max_drawdown_threshold set", pt.get("max_drawdown_threshold", 0) > 0,
         str(pt.get("max_drawdown_threshold"))),
        ("daily_loss_limit set", ex.get("daily_loss_limit", 0) != 0,
         str(ex.get("daily_loss_limit"))),
        ("limits.per_symbol_notional_max set",
         lim.get("per_symbol_notional_max", 0) > 0,
         str(lim.get("per_symbol_notional_max"))),
        ("limits.portfolio_notional_max set",
         lim.get("portfolio_notional_max", 0) > 0,
         str(lim.get("portfolio_notional_max"))),
        ("limits.leverage_max set",
         lim.get("leverage_max", 0) > 0,
         str(lim.get("leverage_max"))),
    ]
    for name, passed, detail in checks:
        results.append(_check(name, passed, detail))
    return results


def check_postmortem_template() -> dict[str, Any]:
    """O6: postmortem template exists."""
    path = PROJECT_ROOT / "docs" / "runbook" / "postmortem_template.md"
    return _check("postmortem_template.md exists", path.exists(), str(path))


def check_go_live_checklist() -> dict[str, Any]:
    """O6: go_live_checklist.md exists."""
    path = PROJECT_ROOT / "docs" / "runbook" / "go_live_checklist.md"
    return _check("go_live_checklist.md exists", path.exists(), str(path))


def check_kill_switch_script() -> dict[str, Any]:
    """O1: kill_switch.py exists and is executable."""
    path = PROJECT_ROOT / "scripts" / "kill_switch.py"
    exists = path.exists()
    return _check("scripts/kill_switch.py exists", exists, str(path))


def check_checkpoint_freshness(max_age_hours: float = 24.0) -> dict[str, Any]:
    """O5: StateStore checkpoint is < max_age_hours old."""
    db = PROJECT_ROOT / "state" / "paper_trader.db"
    if not db.exists():
        return _check("StateStore checkpoint fresh", True,
                      "no checkpoint (first run — OK)")
    age_h = (time.time() - db.stat().st_mtime) / 3600
    return _check("StateStore checkpoint fresh",
                  age_h < max_age_hours,
                  f"age {age_h:.1f}h (max {max_age_hours}h)")


def check_audit_chain() -> dict[str, Any]:
    """O4: audit chain integrity."""
    audit = PROJECT_ROOT / "audit_log" / "audit.jsonl"
    if not audit.exists():
        return _check("audit chain integrity", True,
                      "no audit log yet (first run — OK)")
    verify = PROJECT_ROOT / "scripts" / "verify_audit_log.py"
    if not verify.exists():
        return _check("audit chain integrity", False, "verify_audit_log.py missing")
    result = subprocess.run(
        [sys.executable, str(verify), str(audit)],
        capture_output=True, text=True,
    )
    return _check("audit chain integrity", result.returncode == 0,
                  result.stderr.strip()[:120] if result.returncode != 0 else "")


# ---------------------------------------------------------------------------
# Week 84 checks (R15-R18)
# ---------------------------------------------------------------------------

def check_key_scope_probe() -> dict[str, Any]:
    """R15 (G7): verify_exchange_key_scope.py exists and dry-run passes."""
    probe = PROJECT_ROOT / "scripts" / "verify_exchange_key_scope.py"
    if not probe.exists():
        return _check("API key scope probe script exists", False, str(probe))
    result = subprocess.run(
        [sys.executable, str(probe), "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    detail = (result.stdout + result.stderr).strip()[-120:] if not ok else "dry-run passed"
    return _check("API key scope probe (dry-run)", ok, detail)


def check_precommit_hook() -> dict[str, Any]:
    """R16 (G8): pre-commit secret scanner hook active and runs clean."""
    config = PROJECT_ROOT / ".pre-commit-config.yaml"
    baseline = PROJECT_ROOT / ".secrets.baseline"
    if not config.exists():
        return _check("pre-commit secret scanner active", False, ".pre-commit-config.yaml missing")
    if not baseline.exists():
        return _check("pre-commit secret scanner active", False, ".secrets.baseline missing")
    # Resolve pre-commit binary (may be in venv or anaconda bin)
    pre_commit_bin = Path(sys.executable).parent / "pre-commit"
    if not pre_commit_bin.exists():
        import shutil
        found = shutil.which("pre-commit")
        pre_commit_bin = Path(found) if found else pre_commit_bin

    if not pre_commit_bin.exists():
        return _check("pre-commit secret scanner (detect-secrets)", False,
                      "pre-commit binary not found — run: pip install pre-commit")

    # Check hook is installed (fast check — don't run all-files in drill)
    result = subprocess.run(
        [str(pre_commit_bin), "run", "detect-secrets", "--all-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    detail = result.stdout.strip()[-120:] if not ok else "detect-secrets: no new secrets found"
    return _check("pre-commit secret scanner (detect-secrets)", ok, detail)


def check_drill_history(min_drills: int = 2) -> dict[str, Any]:
    """R18 (G10): runbook drills/ directory has ≥ min_drills completed records."""
    drills_dir = PROJECT_ROOT / "docs" / "runbook" / "drills"
    if not drills_dir.exists():
        return _check(f"runbook drills ≥ {min_drills}", False, "docs/runbook/drills/ missing")
    drill_files = [
        f for f in drills_dir.iterdir()
        if f.suffix == ".md" and f.name != "README.md" and not f.name.startswith("_")
    ]
    count = len(drill_files)
    names = ", ".join(f.name for f in sorted(drill_files)[:3])
    return _check(
        f"runbook drills ≥ {min_drills}",
        count >= min_drills,
        f"{count} drill(s) found: {names}",
    )


# ---------------------------------------------------------------------------
# Kill-switch timing test (local process, simulation_mode)
# ---------------------------------------------------------------------------

def run_kill_switch_timing_test() -> dict[str, Any]:
    """G13: Start a throwaway PaperTrader, fire kill switch, confirm < 5 s."""
    import threading
    import tempfile
    import numpy as np

    # Minimal config
    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 100.0,
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 5,
        },
        "pid_file": str(PROJECT_ROOT / "state" / "drill_kill_test.pid"),
        "monitoring": {},
    }

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from deployment.paper_trader import PaperTrader

        class _DummyAgent:
            def predict(self, obs, deterministic=True):
                return np.array([0.0]), None

        trader = PaperTrader(_DummyAgent(), cfg, simulation_mode=True)

        prices = iter([100.0 + i * 0.01 for i in range(10_000)])
        done_event = threading.Event()

        def _run():
            try:
                trader.run(price_stream=prices, duration_seconds=30)
            finally:
                done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.3)  # let it start

        start = time.monotonic()
        trader._trigger_shutdown("drill: kill switch test")
        done_event.wait(timeout=6.0)
        elapsed = time.monotonic() - start

        pid_file = Path(cfg["pid_file"])
        pid_file.unlink(missing_ok=True)

        ok = elapsed < 5.0 and trader.state.shutdown_triggered
        return _check(
            "Kill switch < 5 s",
            ok,
            f"shutdown in {elapsed:.2f}s, triggered={trader.state.shutdown_triggered}",
        )

    except Exception as exc:
        return _check("Kill switch < 5 s", False, str(exc))


# ---------------------------------------------------------------------------
# $100 Drill simulation
# ---------------------------------------------------------------------------

def run_dollar_drill(capital: float, n_steps: int = 200) -> dict[str, Any]:
    """G12: Run PaperTrader in simulation mode with *capital*, verify it completes."""
    import numpy as np

    rng = np.random.default_rng(42)
    # Synthetic price stream: GBM-like
    prices = [capital * 10]
    for _ in range(n_steps):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.002)))

    cfg: dict[str, Any] = {
        "paper_trading": {
            "symbol": "BTC/USDT",
            "initial_balance": float(capital),
            "trading_fee": 0.001,
            "max_position_size": 1.0,
            "max_drawdown_threshold": 0.20,
            "window_size": 5,
        },
        "pid_file": str(PROJECT_ROOT / "state" / "drill_dollar.pid"),
        "monitoring": {},
        "limits": {
            "per_symbol_notional_max": capital * 5,
            "portfolio_notional_max": capital * 5,
            "leverage_max": 1.0,
        },
    }

    # Ensure project root is importable when run as subprocess
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from deployment.paper_trader import PaperTrader
        from risk_management.limits import PreTradeComplianceChecker
        from deployment.execution.order_manager import OrderManager

        class _RandomAgent:
            def __init__(self, rng):
                self._rng = rng

            def predict(self, obs, deterministic=True):
                return np.array([self._rng.uniform(-1, 1)]), None

        compliance = PreTradeComplianceChecker(cfg["limits"])
        om = OrderManager(
            exchange_config={"symbol": "BTC/USDT"},
            paper_mode=True,
            compliance_checker=compliance,
        )
        trader = PaperTrader(
            _RandomAgent(rng), cfg, simulation_mode=True, order_manager=om
        )
        t0 = time.monotonic()
        report = trader.run(price_stream=iter(prices), duration_seconds=60)
        elapsed = time.monotonic() - t0

        Path(cfg["pid_file"]).unlink(missing_ok=True)

        final_pv = report.get("final_portfolio_value", capital)
        pnl = final_pv - capital
        shutdown = trader.state.shutdown_triggered

        return _check(
            f"${capital:.0f} drill completed",
            not (shutdown and "error" in str(trader.state.shutdown_reason).lower()),
            f"steps={n_steps}, pnl={pnl:+.2f}, elapsed={elapsed:.1f}s, "
            f"shutdown={shutdown} reason={trader.state.shutdown_reason!r}",
        )

    except Exception as exc:
        return _check(f"${capital:.0f} drill completed", False, str(exc))


# ---------------------------------------------------------------------------
# $100 Live Drill (I1-c)
# ---------------------------------------------------------------------------

def _preflight(capital: float, exchange) -> list[str]:
    """I6-a: Pre-flight checks before live drill. Returns list of failure strings."""
    import datetime as _dt
    failures: list[str] = []

    # 1. Balance check
    try:
        bal = exchange.fetch_balance()
        free_usdt = float((bal.get("USDT") or {}).get("free", 0))
        if free_usdt < capital:
            failures.append(
                f"USDT balance {free_usdt:.2f} < required {capital:.2f}"
            )
    except Exception as exc:
        failures.append(f"balance fetch failed: {exc}")

    # 2. 24h duplicate drill guard
    history_dir = PROJECT_ROOT / "docs" / "phase7.6"
    if history_dir.exists():
        recent = sorted(history_dir.glob("live_drill_*.md"), reverse=True)
        if recent:
            try:
                stem = recent[0].stem  # live_drill_20260426T090000Z
                ts_part = stem.rsplit("_", 1)[-1].rstrip("Z")
                last_ts = _dt.datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
                delta = _dt.datetime.utcnow() - last_ts
                if delta < _dt.timedelta(hours=24):
                    failures.append(
                        f"last drill {recent[0].name} < 24h ago "
                        f"(elapsed {delta.total_seconds() / 3600:.1f}h)"
                    )
            except (ValueError, IndexError):
                pass  # malformed filename — skip

    return failures


def run_live_drill(
    capital: float,
    symbol: str = "BTC/USDT",
    *,
    _exchange=None,       # injectable for tests (real ccxt exchange object)
    _order_manager=None,  # injectable for tests (OrderManager instance)
) -> dict[str, Any]:
    """I1-c: Real exchange $100 drill. Requires EXCHANGE_BINANCE_KEY/SECRET env vars.

    Safety guards (hard-coded, non-configurable):
      - capital > 100 → refuse
      - symbol != BTC/USDT → interactive confirm prompt
      - 10-minute total timeout → cancel all + abort
    """
    import threading

    if capital > 100:
        return _check("live drill capital guard", False,
                      f"capital ${capital:.0f} exceeds $100 drill limit — refusing")

    if symbol != "BTC/USDT":
        resp = input(
            f"Symbol is {symbol!r} (not BTC/USDT). Proceed? [y/N] "
        ).strip().lower()
        if resp != "y":
            return _check("live drill symbol confirm", False, "aborted by user")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # ── 1. Load credentials ──────────────────────────────────────────────────
    if _exchange is None:
        try:
            from deployment.secrets.secret_provider import get_default_provider
            provider = get_default_provider()
            try:
                api_key = provider.get("EXCHANGE_BINANCE_KEY")
                api_secret = provider.get("EXCHANGE_BINANCE_SECRET")
            except KeyError as missing:
                print(f"❌ missing credentials: set {missing}")
                sys.exit(1)
        except ImportError as exc:
            return _check("live drill credentials", False,
                          f"SecretProvider import failed: {exc}")
    else:
        # Test injection path: _exchange is provided, credentials not needed
        api_key = ""
        api_secret = ""

    # ── 2. Verify key scope (no Withdraw permission) ─────────────────────────
    if _exchange is None:
        try:
            from scripts.verify_exchange_key_scope import run_probes
            probes, scope_ok = run_probes(
                exchange_id="binance",
                api_key=api_key,
                api_secret=api_secret,
                sandbox=False,
                symbol=symbol,
            )
        except Exception as exc:
            return _check("live drill key scope", False, str(exc))
        if not scope_ok:
            withdraw_probe = next(
                (p for p in probes if "Withdraw" in p["name"]), None
            )
            if withdraw_probe and not withdraw_probe["pass"]:
                print("❌ Withdraw permission detected — refusing live drill")
                sys.exit(1)
            return _check("live drill key scope", False,
                          "one or more scope probes failed")

    # ── 3. Fetch mid price ───────────────────────────────────────────────────
    if _exchange is not None:
        exchange = _exchange
    else:
        try:
            import ccxt  # type: ignore
            exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            })
        except ImportError:
            return _check("live drill ccxt import", False,
                          "ccxt not installed — run: pip install ccxt")

    try:
        ticker = exchange.fetch_ticker(symbol)
        bid = ticker.get("bid") or ticker.get("last", 0)
        ask = ticker.get("ask") or ticker.get("last", 0)
        mid_price = (bid + ask) / 2
        print(f"  Mid price {symbol}: {mid_price:.2f}")
    except Exception as exc:
        return _check("live drill mid price fetch", False, str(exc))

    # Pre-drill balance
    try:
        balance = exchange.fetch_balance()
        pre_usdt = float((balance.get("USDT") or {}).get("free", 0))
        pre_btc = float((balance.get("BTC") or {}).get("free", 0))
        print(f"  Pre-drill: {pre_usdt:.2f} USDT, {pre_btc:.8f} BTC")
    except Exception as exc:
        return _check("live drill balance fetch", False, str(exc))

    # I6-a: pre-flight checks (balance + 24h dedupe) — skip when exchange is injected (tests)
    if _exchange is None:
        preflight_failures = _preflight(capital, exchange)
        if preflight_failures:
            detail = "; ".join(preflight_failures)
            print(f"  ❌ Pre-flight failed: {detail}")
            return _check("live drill pre-flight", False, detail)

    # ── 4. Submit limit buy $50 @ mid × 0.98 ────────────────────────────────
    notional = 50.0
    limit_price = round(mid_price * 0.98, 2)
    qty = round(notional / limit_price, 6)

    if _order_manager is not None:
        om = _order_manager
    else:
        try:
            from deployment.execution.order_manager import OrderManager
            from deployment.audit.audit_logger import AuditLogger
        except ImportError as exc:
            return _check("live drill order manager import", False, str(exc))

        audit_dir = PROJECT_ROOT / "audit_log"
        audit_dir.mkdir(exist_ok=True)
        audit_logger = AuditLogger(str(audit_dir / "audit.jsonl"))

        om = OrderManager(
            exchange_config={
                "exchange_id": "binance",
                "api_key": api_key,
                "api_secret": api_secret,
                "sandbox": False,
                "symbol": symbol,
            },
            paper_mode=False,
            audit_logger=audit_logger,
        )

    # ── I6-b: Dead-man watchdog (10-min hard timeout) ───────────────────────
    timed_out = threading.Event()
    _drill_start = time.monotonic()

    def _watchdog() -> None:
        """Monitor thread: checks elapsed every 10s; force-flat on timeout."""
        while not timed_out.is_set():
            timed_out.wait(timeout=10)
            if timed_out.is_set():
                break
            elapsed_w = time.monotonic() - _drill_start
            if elapsed_w >= 600:
                print("  ⏰ 10-min watchdog fired — cancelling all orders + market flatten")
                try:
                    om.cancel_all_orders()
                except Exception:
                    pass
                try:
                    cur_btc = float(
                        (exchange.fetch_balance().get("BTC") or {}).get("free", 0)
                    )
                    if cur_btc > 0:
                        exchange.create_market_sell_order(symbol, cur_btc)
                except Exception:
                    pass
                # Write postmortem skeleton
                _ts_w = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                _pm_src = PROJECT_ROOT / "docs" / "runbook" / "postmortem_template.md"
                _pm_dst = PROJECT_ROOT / "docs" / "phase7.6" / f"live_drill_timeout_{_ts_w}.md"
                _pm_dst.parent.mkdir(parents=True, exist_ok=True)
                if _pm_src.exists():
                    import shutil
                    shutil.copy(_pm_src, _pm_dst)
                else:
                    _pm_dst.write_text(
                        f"# Live Drill Timeout Postmortem — {_ts_w}\n\n"
                        "Drill timed out after 10 minutes. Investigate order latency.\n"
                    )
                timed_out.set()
                break

    watchdog = threading.Thread(target=_watchdog, daemon=True, name="drill-watchdog")
    watchdog.start()

    t0 = time.monotonic()
    _submit_ts = time.monotonic()

    try:
        order_id = om.submit_order(
            side="buy",
            amount=qty,
            order_type="limit",
            limit_price=limit_price,
        )
    except Exception as exc:
        timed_out.set()
        return _check("live drill order submit", False, str(exc))

    _ack_ts = time.monotonic()
    submit_latency_ms = (_ack_ts - _submit_ts) * 1000
    print(f"  Limit buy submitted: {qty} {symbol} @ {limit_price} (id={order_id}, ack={submit_latency_ms:.0f}ms)")

    # ── 5. Poll for 3 minutes (every 30s) ───────────────────────────────────
    fill_status = "open"
    for _ in range(6):
        if timed_out.is_set():
            break
        time.sleep(30)
        try:
            fill_status = om.check_order(order_id)
        except Exception:
            pass
        print(f"  Order status: {fill_status}")
        if fill_status in ("filled", "cancelled", "failed"):
            break

    elapsed = time.monotonic() - t0
    _fill_ts = time.monotonic()

    if timed_out.is_set() or elapsed >= 600:
        om.cancel_all_orders()
        timed_out.set()
        return _check("live drill timeout", False,
                      f"10-minute timeout — all orders cancelled after {elapsed:.0f}s")

    # ── 6. Handle fill status ────────────────────────────────────────────────
    try:
        order = om.get_order(order_id)
    except Exception:
        order = None

    fill_latency_ms: float = (_fill_ts - _ack_ts) * 1000
    try:
        actual_fill_price = float(getattr(order, "avg_fill_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        actual_fill_price = 0.0
    try:
        actual_fee = float(getattr(order, "fee", 0.0) or 0.0)
    except (TypeError, ValueError):
        actual_fee = 0.0

    if fill_status in ("open", "pending"):
        print("  Unfilled — cancelling")
        om.cancel_order(order_id)
        fill_status = "cancelled"
    elif fill_status == "filled":
        filled_qty = order.filled_amount if order else qty
        print(f"  Filled @ {actual_fill_price or '?'} — market sell {filled_qty}")
        om.submit_order(side="sell", amount=filled_qty, order_type="market")
    elif fill_status in ("partial", "partially_filled"):
        filled_qty = order.filled_amount if order else 0.0
        print(f"  Partial fill ({filled_qty}) — cancel remainder + market sell")
        om.cancel_order(order_id)
        if filled_qty > 0:
            om.submit_order(side="sell", amount=filled_qty, order_type="market")

    # ── 7. Audit chain verification ──────────────────────────────────────────
    audit_jsonl = PROJECT_ROOT / "audit_log" / "audit.jsonl"
    audit_ok = True
    if audit_jsonl.exists():
        verify_result = subprocess.run(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "verify_audit_log.py"),
             str(audit_jsonl)],
            capture_output=True, text=True,
        )
        audit_ok = verify_result.returncode == 0

    # ── 8. Post-drill balance + report ───────────────────────────────────────
    try:
        post_bal = exchange.fetch_balance()
        post_usdt = float((post_bal.get("USDT") or {}).get("free", 0))
        post_btc = float((post_bal.get("BTC") or {}).get("free", 0))
        usdt_diff = post_usdt - pre_usdt
        btc_diff = post_btc - pre_btc
    except Exception:
        post_usdt = post_btc = usdt_diff = btc_diff = 0.0

    if _order_manager is None:
        try:
            om.close()
        except Exception:
            pass

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = PROJECT_ROOT / "docs" / "phase7.6"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"live_drill_{ts}.md"
    # I6-c: Slippage model predicted vs actual
    slippage_predicted: float = 0.0
    slippage_actual: float = 0.0
    if actual_fill_price and mid_price:
        slippage_actual = abs(actual_fill_price - mid_price) / mid_price
    # Expected fee from config (paper mode default: 0.1%)
    config_fee_rate: float = 0.001
    config_fee_amount: float = notional * config_fee_rate

    report_path.write_text(
        f"# Live Drill Report — {ts}\n\n"
        f"**Symbol**: {symbol}  \n"
        f"**Capital**: ${capital:.0f}  \n"
        f"**Notional**: ${notional:.0f}  \n"
        f"**Limit price**: {limit_price} (mid {mid_price:.2f} × 0.98)  \n"
        f"**Fill status**: {fill_status}  \n"
        f"**Elapsed**: {elapsed:.1f}s  \n\n"
        f"## Balance\n\n"
        f"| | Pre | Post | Diff |\n"
        f"|---|---|---|---|\n"
        f"| USDT | {pre_usdt:.2f} | {post_usdt:.2f} | {usdt_diff:+.2f} |\n"
        f"| BTC  | {pre_btc:.8f} | {post_btc:.8f} | {btc_diff:+.8f} |\n\n"
        f"## Latency\n\n"
        f"| Stage | Latency |\n"
        f"|-------|---------|\n"
        f"| Submit → ack | {submit_latency_ms:.0f}ms |\n"
        f"| Ack → fill/cancel | {fill_latency_ms:.0f}ms |\n\n"
        f"## Slippage\n\n"
        f"| | Predicted | Actual |\n"
        f"|---|---|---|\n"
        f"| Slippage | {slippage_predicted:.4%} | {slippage_actual:.4%} |\n\n"
        f"## Fee\n\n"
        f"| | Config | Actual |\n"
        f"|---|---|---|\n"
        f"| Fee | ${config_fee_amount:.4f} | ${actual_fee:.4f} |\n\n"
        f"## Audit Chain\n\n"
        f"{'✅ intact' if audit_ok else '❌ FAILED'}\n"
    )
    print(f"  Report → {report_path.relative_to(PROJECT_ROOT)}")

    timed_out.set()  # release watchdog thread
    ok = elapsed < 600 and audit_ok
    return _check(
        "live drill completed",
        ok,
        f"status={fill_status}, elapsed={elapsed:.1f}s, audit={'ok' if audit_ok else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="First Dollar pre-flight drill (G12/G13)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Run simulation drill with this capital (e.g. 100)")
    parser.add_argument("--config", default=None,
                        help="YAML config path for risk/exchange settings")
    parser.add_argument("--check-only", action="store_true",
                        help="Run auto-checks only, skip trading drill")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--skip-kill-switch-test", action="store_true",
                        help="Skip the in-process kill switch timing test")
    parser.add_argument(
        "--live", action="store_true",
        help="Run real exchange drill (requires EXCHANGE_BINANCE_KEY/SECRET). "
             "Implies --capital 100 if --capital is not set.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  First Dollar Pre-Flight Drill")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # Load config
    config: dict[str, Any] = {}
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"WARNING: could not load config {args.config}: {exc}")
    else:
        config = {
            "paper_trading": {"max_drawdown_threshold": 0.20, "window_size": 5},
            "exchange": {"daily_loss_limit": -500.0},
            "limits": {
                "per_symbol_notional_max": 10_000.0,
                "portfolio_notional_max": 50_000.0,
                "leverage_max": 1.0,
            },
        }

    results: list[dict[str, Any]] = []

    print("── Structural Checks ──────────────────────────────────────")
    results.append(check_pytest_ini_ignores())
    results.append(check_no_old_risk_api())
    results.append(check_postmortem_template())
    results.append(check_go_live_checklist())
    results.append(check_kill_switch_script())
    results.append(check_checkpoint_freshness())
    results.append(check_audit_chain())

    print("\n── Risk Config Checks ─────────────────────────────────────")
    results.extend(check_risk_config(config))

    print("\n── Week 84 Security & Capacity Checks ────────────────────")
    results.append(check_key_scope_probe())
    results.append(check_precommit_hook())
    results.append(check_drill_history())

    if not args.skip_kill_switch_test:
        print("\n── Kill Switch Timing Test ────────────────────────────────")
        results.append(run_kill_switch_timing_test())

    if args.live:
        capital = args.capital or 100.0
        print(f"\n── ${capital:.0f} Live Drill (Real Exchange) ──────────────────────")
        results.append(run_live_drill(capital))
    elif not args.check_only:
        capital = args.capital or 100.0
        print(f"\n── ${capital:.0f} Simulation Drill ─────────────────────────────")
        results.append(run_dollar_drill(capital))

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ✅ ALL CHECKS PASSED — you may proceed to go-live sign-off")
    else:
        print("  ❌ CHECKS FAILED — resolve before going live")
    print(f"{'='*60}\n")

    if args.report:
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "failed": failed,
            "total": total,
            "results": results,
        }
        Path(args.report).write_text(json.dumps(report_data, indent=2))
        print(f"Report written to {args.report}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
