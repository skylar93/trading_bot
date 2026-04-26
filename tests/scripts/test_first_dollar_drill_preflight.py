"""I6-a: _preflight function tests."""
from __future__ import annotations

import pathlib
import sys
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_exchange(usdt_balance: float = 200.0) -> MagicMock:
    ex = MagicMock()
    ex.fetch_balance.return_value = {"USDT": {"free": usdt_balance}}
    return ex


def test_preflight_passes_when_balance_sufficient():
    from scripts.first_dollar_drill import _preflight
    exchange = _make_exchange(usdt_balance=200.0)
    failures = _preflight(capital=100.0, exchange=exchange)
    # Only the 24h-dedupe could fail if a drill was run recently;
    # balance check should pass.
    balance_failures = [f for f in failures if "balance" in f.lower()]
    assert not balance_failures


def test_preflight_fails_when_balance_insufficient():
    from scripts.first_dollar_drill import _preflight
    exchange = _make_exchange(usdt_balance=50.0)
    failures = _preflight(capital=100.0, exchange=exchange)
    assert any("balance" in f.lower() for f in failures)


def test_preflight_fails_on_24h_dedupe(tmp_path, monkeypatch):
    """If a drill was run < 24h ago, preflight rejects."""
    from scripts import first_dollar_drill as fdd
    import scripts.first_dollar_drill as fdd_mod

    # Create a fake recent drill report
    docs_dir = tmp_path / "docs" / "phase7.6"
    docs_dir.mkdir(parents=True)
    recent_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    (docs_dir / f"live_drill_{recent_ts}Z.md").write_text("# drill")

    monkeypatch.setattr(fdd_mod, "PROJECT_ROOT", tmp_path)

    exchange = _make_exchange(usdt_balance=200.0)
    failures = fdd_mod._preflight(capital=100.0, exchange=exchange)
    assert any("24h" in f.lower() or "ago" in f.lower() for f in failures)


def test_preflight_passes_when_no_recent_drill(tmp_path, monkeypatch):
    """No recent drill → no 24h failure."""
    from scripts import first_dollar_drill as fdd_mod

    # Empty docs dir — no previous drill
    docs_dir = tmp_path / "docs" / "phase7.6"
    docs_dir.mkdir(parents=True)

    monkeypatch.setattr(fdd_mod, "PROJECT_ROOT", tmp_path)

    exchange = _make_exchange(usdt_balance=200.0)
    failures = fdd_mod._preflight(capital=100.0, exchange=exchange)
    assert not any("24h" in f.lower() for f in failures)
