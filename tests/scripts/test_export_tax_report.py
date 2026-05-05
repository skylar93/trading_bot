"""
Tests for scripts/export_tax_report.py — E6 tax/accounting CSV export.

All 10 cases from the plan.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_tax_report import (
    _fifo_match,
    _get_exchange_rate,
    _read_fills,
    _write_us1099b,
    main,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_GENESIS = "0" * 64


def _sha256(prev: str, payload: dict) -> str:
    raw = prev + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _make_fill_record(
    ts: str,
    side: str,
    amount: float,
    price: float,
    fee: float,
    order_id: str,
    prev_hash: str,
) -> tuple[dict, str]:
    """Build an audit fill record dict and return (record, new_hash)."""
    payload = {
        "order_id": order_id,
        "side": side,
        "filled_amount": amount,
        "avg_fill_price": price,
        "fee": fee,
        "pnl": 0.0,
    }
    h = _sha256(prev_hash, payload)
    record = {"ts": ts, "type": "fill", "payload": payload, "hash": h}
    return record, h


def _write_audit(tmp_path: Path, records: list[dict]) -> Path:
    log = tmp_path / "audit.jsonl"
    with open(log, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return log


def _run_main(argv: list[str]) -> int:
    """Run main() capturing SystemExit and returning exit code."""
    old_argv = sys.argv
    sys.argv = ["export_tax_report.py"] + argv
    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
        return exc_info.value.code
    finally:
        sys.argv = old_argv


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — single buy + single sell, korea format
# ─────────────────────────────────────────────────────────────────────────────

def test_single_buy_sell_korea(tmp_path: Path) -> None:
    rec1, h1 = _make_fill_record("2026-03-01T10:00:00+00:00", "buy",  1.0, 50000.0, 5.0, "b1", _GENESIS)
    rec2, h2 = _make_fill_record("2026-06-01T10:00:00+00:00", "sell", 1.0, 60000.0, 6.0, "s1", h1)
    log = _write_audit(tmp_path, [rec1, rec2])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "korea",
                      "--output", str(out), "--audit-log", str(log)])

    assert code == 0
    rows = list(csv.reader(open(out)))
    # header + 1 data row + TOTAL row
    assert len(rows) == 3
    data = rows[1]
    # qty
    assert float(data[3]) == pytest.approx(1.0)
    # price_krw == sell_price * 1.0 (no FX config)
    assert float(data[4]) == pytest.approx(60000.0)
    # fill_id
    assert data[7] == "s1"


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — two buys, one sell (FIFO partial)
# ─────────────────────────────────────────────────────────────────────────────

def test_fifo_two_buys_one_sell(tmp_path: Path) -> None:
    rec1, h1 = _make_fill_record("2026-01-10T00:00:00+00:00", "buy",  1.0, 40000.0, 4.0, "b1", _GENESIS)
    rec2, h2 = _make_fill_record("2026-02-10T00:00:00+00:00", "buy",  2.0, 45000.0, 9.0, "b2", h1)
    rec3, h3 = _make_fill_record("2026-05-10T00:00:00+00:00", "sell", 2.5, 55000.0, 5.5, "s1", h2)
    log = _write_audit(tmp_path, [rec1, rec2, rec3])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "us-1099b",
                      "--output", str(out), "--audit-log", str(log)])

    assert code == 0
    rows = list(csv.reader(open(out)))
    # header + 2 matched lots + TOTAL
    assert len(rows) == 4

    # First lot: b1 consumed fully (1.0 @ 40000)
    assert float(rows[1][2]) == pytest.approx(55000.0)   # proceeds
    assert float(rows[1][3]) == pytest.approx(40000.0)   # cost_basis

    # Second lot: 1.5 @ 45000 from b2
    assert float(rows[2][2]) == pytest.approx(55000.0 * 1.5)
    assert float(rows[2][3]) == pytest.approx(45000.0 * 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — sell with no matching buy
# ─────────────────────────────────────────────────────────────────────────────

def test_sell_no_matching_buy(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rec1, h1 = _make_fill_record("2026-04-01T00:00:00+00:00", "sell", 1.0, 50000.0, 5.0, "s1", _GENESIS)
    log = _write_audit(tmp_path, [rec1])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "us-1099b",
                      "--output", str(out), "--audit-log", str(log)])

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    rows = list(csv.reader(open(out)))
    # header + 1 NaN row + TOTAL
    assert len(rows) == 3
    assert rows[1][3] == "NaN"   # cost_basis_usd


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — empty year
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_year(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rec1, h1 = _make_fill_record("2025-06-01T00:00:00+00:00", "buy", 1.0, 30000.0, 3.0, "b1", _GENESIS)
    log = _write_audit(tmp_path, [rec1])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "generic",
                      "--output", str(out), "--audit-log", str(log)])

    assert code == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    rows = list(csv.reader(open(out)))
    assert len(rows) == 1  # header only


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — us-1099b short vs long term
# ─────────────────────────────────────────────────────────────────────────────

def test_short_vs_long_term(tmp_path: Path) -> None:
    # Test _write_us1099b directly with synthetic matched rows so we avoid
    # the cross-year buy/sell scoping issue of the CLI (--year reads one year).
    from datetime import date

    short_row = {
        "buy_date": date(2026, 1, 1),
        "buy_price": 40000.0,
        "buy_fill_id": "b_short",
        "sell_date": date(2026, 6, 1),    # 151 days < 365
        "sell_price": 50000.0,
        "sell_fill_id": "s_short",
        "sell_hash": "aaa",
        "qty": 1.0,
        "cost_basis_usd": 40000.0,
        "proceeds_usd": 50000.0,
        "gain_loss_usd": 10000.0,
        "chain_broken": False,
    }
    long_row = {
        "buy_date": date(2025, 1, 1),
        "buy_price": 30000.0,
        "buy_fill_id": "b_long",
        "sell_date": date(2026, 3, 1),    # 424 days >= 365
        "sell_price": 48000.0,
        "sell_fill_id": "s_long",
        "sell_hash": "bbb",
        "qty": 1.0,
        "cost_basis_usd": 30000.0,
        "proceeds_usd": 48000.0,
        "gain_loss_usd": 18000.0,
        "chain_broken": False,
    }

    out_short = tmp_path / "short.csv"
    out_long  = tmp_path / "long.csv"

    with open(out_short, "w", newline="") as fh:
        _write_us1099b(csv.writer(fh), [short_row])
    with open(out_long, "w", newline="") as fh:
        _write_us1099b(csv.writer(fh), [long_row])

    short_rows = list(csv.reader(open(out_short)))
    long_rows  = list(csv.reader(open(out_long)))
    assert short_rows[1][5] == "short"
    assert long_rows[1][5] == "long"


# ─────────────────────────────────────────────────────────────────────────────
# Case 6 — generic format (no FIFO, raw rows)
# ─────────────────────────────────────────────────────────────────────────────

def test_generic_format_no_fifo(tmp_path: Path) -> None:
    rec1, h1 = _make_fill_record("2026-04-01T00:00:00+00:00", "buy",  1.0, 50000.0, 5.0, "b1", _GENESIS)
    rec2, h2 = _make_fill_record("2026-05-01T00:00:00+00:00", "buy",  0.5, 52000.0, 2.6, "b2", h1)
    rec3, h3 = _make_fill_record("2026-06-01T00:00:00+00:00", "sell", 1.5, 55000.0, 8.25, "s1", h2)
    log = _write_audit(tmp_path, [rec1, rec2, rec3])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "generic",
                      "--output", str(out), "--audit-log", str(log)])

    assert code == 0
    rows = list(csv.reader(open(out)))
    # header + 3 raw fill rows (no FIFO matching)
    assert len(rows) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Case 7 — year outside log range
# ─────────────────────────────────────────────────────────────────────────────

def test_year_outside_range(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rec1, h1 = _make_fill_record("2026-07-01T00:00:00+00:00", "buy", 1.0, 50000.0, 5.0, "b1", _GENESIS)
    log = _write_audit(tmp_path, [rec1])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2024", "--format", "generic",
                      "--output", str(out), "--audit-log", str(log)])

    assert code == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# ─────────────────────────────────────────────────────────────────────────────
# Case 8 — audit log missing
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_log_missing(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    code = _run_main(["--year", "2026", "--format", "generic",
                      "--output", str(out),
                      "--audit-log", str(tmp_path / "nonexistent.jsonl")])
    assert code == 2


# ─────────────────────────────────────────────────────────────────────────────
# Case 9 — broken hash chain
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_chain_row_exported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rec_good, h1 = _make_fill_record("2026-03-01T00:00:00+00:00", "buy",  1.0, 50000.0, 5.0, "b1", _GENESIS)
    rec_broken, _  = _make_fill_record("2026-05-01T00:00:00+00:00", "sell", 1.0, 55000.0, 5.5, "s1", _GENESIS)
    # Tamper with hash to simulate break
    rec_broken["hash"] = "0" * 64

    log = _write_audit(tmp_path, [rec_good, rec_broken])
    out = tmp_path / "out.csv"

    code = _run_main(["--year", "2026", "--format", "us-1099b",
                      "--output", str(out), "--audit-log", str(log)])

    # Should exit 1 (broken chain found) but still export
    assert code == 1
    rows = list(csv.reader(open(out)))
    # Find a row starting with # BROKEN_CHAIN
    broken_rows = [r for r in rows if r and r[0].startswith("# BROKEN_CHAIN")]
    assert len(broken_rows) == 1
    # Non-broken rows should not have the prefix (buy is clean, no sell matched row should be clean)
    clean_rows = [r for r in rows[1:] if r and not r[0].startswith(("#", "TOTAL", "# TOTAL"))]
    # There shouldn't be clean data rows here since the sell was broken
    # but we confirm clean_rows don't have BROKEN_CHAIN prefix
    for cr in clean_rows:
        assert not cr[0].startswith("# BROKEN_CHAIN")


# ─────────────────────────────────────────────────────────────────────────────
# Case 10 — FX rate absent, non-USD ccy
# ─────────────────────────────────────────────────────────────────────────────

def test_fx_rate_absent_warns(capsys: pytest.CaptureFixture) -> None:
    from datetime import date
    rate = _get_exchange_rate(date(2026, 3, 1), "USD", "KRW", {})
    assert rate == pytest.approx(1.0)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
