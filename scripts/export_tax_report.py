#!/usr/bin/env python
"""
E6: Tax / accounting CSV export.

Reads fills from the audit log and writes a jurisdiction-formatted CSV.

Usage:
    python scripts/export_tax_report.py \\
        --year 2026 \\
        --format korea \\
        --output reports/tax/2026_korea.csv \\
        [--audit-log audit_log/audit.jsonl]

Formats:
  korea      양도세 신고용 — FIFO lot matching, KRW columns
  us-1099b   US Schedule D / Form 1099-B — FIFO, USD columns
  generic    All raw fill fields, no FIFO, no jurisdictional logic

Exit codes:
  0  success (including empty year)
  1  partial failure (broken hash chain rows exported with warning prefix)
  2  fatal — audit log missing or unreadable

See docs/phase8/tax_export_caveats.md for known gaps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_GENESIS_HASH = "0" * 64
_DEFAULT_AUDIT_LOG = PROJECT_ROOT / "audit_log" / "audit.jsonl"
_FX_RATES_CONFIG = PROJECT_ROOT / "config" / "fx_rates.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# FX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_fx_rates() -> Dict[str, Any]:
    """Load config/fx_rates.yaml if present. Returns {} if absent."""
    if not _FX_RATES_CONFIG.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(_FX_RATES_CONFIG) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        # PyYAML not installed — try a naive line-based parse
        rates: Dict[str, Any] = {}
        try:
            with open(_FX_RATES_CONFIG) as fh:
                for line in fh:
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        k, _, v = line.partition(":")
                        try:
                            rates[k.strip()] = float(v.strip())
                        except ValueError:
                            pass
        except OSError:
            pass
        return rates


def _get_exchange_rate(
    fill_date: date,
    from_ccy: str,
    to_ccy: str,
    fx_rates: Dict[str, Any],
) -> float:
    """Return exchange rate from_ccy→to_ccy for fill_date.

    Lookup order:
      1. fx_rates["{from_ccy}_{to_ccy}_{YYYY-MM-DD}"]
      2. fx_rates["{from_ccy}_{to_ccy}"]
      3. Falls back to 1.0 with a stderr WARNING.
    """
    if from_ccy == to_ccy:
        return 1.0
    dated_key = f"{from_ccy}_{to_ccy}_{fill_date.isoformat()}"
    generic_key = f"{from_ccy}_{to_ccy}"
    if dated_key in fx_rates:
        return float(fx_rates[dated_key])
    if generic_key in fx_rates:
        return float(fx_rates[generic_key])
    print(
        f"WARNING: no FX rate for {from_ccy}→{to_ccy} on {fill_date}; "
        "using 1.0. Add rate to config/fx_rates.yaml for accurate output.",
        file=sys.stderr,
    )
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Audit log reader
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(prev_hash: str, payload: Dict[str, Any]) -> str:
    raw = prev_hash + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_fills(
    audit_log_path: str,
    year: int,
) -> List[Dict[str, Any]]:
    """Read all fill records for the given year.

    Returns a list of dicts with an extra ``_audit_hash`` key and
    ``_chain_broken`` bool.  Raises SystemExit(2) if file is unreadable.
    """
    path = Path(audit_log_path)
    if not path.exists():
        print(f"ERROR: audit log not found: {path}", file=sys.stderr)
        sys.exit(2)

    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    fills: List[Dict[str, Any]] = []
    prev_hash = _GENESIS_HASH

    try:
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                record = json.loads(raw_line)
                stored_hash = record.get("hash", "")
                payload = record.get("payload", {})
                expected_hash = _sha256(prev_hash, payload)
                chain_broken = stored_hash != expected_hash
                prev_hash = stored_hash  # advance chain regardless

                if record.get("type") != "fill":
                    continue

                # Parse timestamp
                ts_str = record.get("ts", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if not (year_start <= ts < year_end):
                    continue

                row = dict(payload)
                row["_ts"] = ts
                row["_audit_hash"] = stored_hash
                row["_chain_broken"] = chain_broken
                fills.append(row)
    except OSError as exc:
        print(f"ERROR: cannot read audit log: {exc}", file=sys.stderr)
        sys.exit(2)

    return fills


# ─────────────────────────────────────────────────────────────────────────────
# FIFO lot matching
# ─────────────────────────────────────────────────────────────────────────────

class _Lot:
    __slots__ = ("qty", "price", "date", "fill_id", "hash")

    def __init__(self, qty: float, price: float, dt: date, fill_id: str, h: str) -> None:
        self.qty = qty
        self.price = price
        self.date = dt
        self.fill_id = fill_id
        self.hash = h


def _fifo_match(fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match sells to buys via FIFO.  Returns list of matched-row dicts.

    Each matched row has:
      buy_date, buy_price, buy_fill_id
      sell_date, sell_price, sell_fill_id, sell_hash
      qty, cost_basis_usd, proceeds_usd, gain_loss_usd
      chain_broken (bool)
    Sells with no matching buy get cost_basis_usd=NaN.
    """
    lots: Deque[_Lot] = deque()
    matched: List[Dict[str, Any]] = []

    for fill in fills:
        side = (fill.get("side") or "").lower()
        qty = float(fill.get("filled_amount") or fill.get("amount") or 0.0)
        price = float(fill.get("avg_fill_price") or fill.get("limit_price") or 0.0)
        fill_id = str(fill.get("order_id") or fill.get("exchange_order_id") or "")
        ts: datetime = fill["_ts"]
        fill_date = ts.date()
        h = fill["_audit_hash"]
        broken = fill["_chain_broken"]

        if qty <= 0:
            continue

        if side == "buy":
            lots.append(_Lot(qty, price, fill_date, fill_id, h))
        elif side == "sell":
            remaining = qty
            while remaining > 0 and lots:
                lot = lots[0]
                consumed = min(lot.qty, remaining)
                proceeds = price * consumed
                cost = lot.price * consumed
                matched.append({
                    "buy_date": lot.date,
                    "buy_price": lot.price,
                    "buy_fill_id": lot.fill_id,
                    "sell_date": fill_date,
                    "sell_price": price,
                    "sell_fill_id": fill_id,
                    "sell_hash": h,
                    "qty": consumed,
                    "cost_basis_usd": cost,
                    "proceeds_usd": proceeds,
                    "gain_loss_usd": proceeds - cost,
                    "chain_broken": broken,
                })
                lot.qty -= consumed
                remaining -= consumed
                if lot.qty <= 0:
                    lots.popleft()

            if remaining > 0:
                # Sell with no matching buy for the residual qty
                print(
                    f"WARNING: sell fill_id={fill_id!r} on {fill_date} has "
                    f"{remaining:.8f} units with no matching buy lot in scope.",
                    file=sys.stderr,
                )
                matched.append({
                    "buy_date": None,
                    "buy_price": float("nan"),
                    "buy_fill_id": None,
                    "sell_date": fill_date,
                    "sell_price": price,
                    "sell_fill_id": fill_id,
                    "sell_hash": h,
                    "qty": remaining,
                    "cost_basis_usd": float("nan"),
                    "proceeds_usd": price * remaining,
                    "gain_loss_usd": float("nan"),
                    "chain_broken": broken,
                })

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Format writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_korea(
    writer: csv.writer,
    matched: List[Dict[str, Any]],
    fx_rates: Dict[str, Any],
) -> bool:
    """Write korea-format rows.  Returns True if any chain-broken rows written."""
    header = [
        "date", "symbol", "side", "qty",
        "price_krw", "fee_krw", "exchange_rate_used",
        "fill_id", "audit_hash", "holding_period_days",
    ]
    writer.writerow(header)
    any_broken = False
    total_gain = 0.0
    for row in matched:
        import math
        cost_basis = row["cost_basis_usd"]
        proceeds = row["proceeds_usd"]
        sell_date = row["sell_date"]
        buy_date = row["buy_date"]

        rate = _get_exchange_rate(sell_date, "USD", "KRW", fx_rates)
        price_krw = row["sell_price"] * rate
        # fee not available from lot-matched row — mark as 0 (see caveats)
        fee_krw = 0.0
        holding_days = (
            (sell_date - buy_date).days if buy_date is not None else ""
        )
        gain_loss_krw = (
            row["gain_loss_usd"] * rate if not math.isnan(row["gain_loss_usd"]) else "NaN"
        )
        cost_krw = (
            cost_basis * rate if not math.isnan(cost_basis) else "NaN"
        )

        data_row = [
            sell_date.isoformat(),
            "",  # symbol gap — see caveats
            "sell",
            f"{row['qty']:.8f}",
            f"{price_krw:.2f}",
            f"{fee_krw:.2f}",
            f"{rate:.4f}",
            row["sell_fill_id"],
            row["sell_hash"],
            holding_days,
        ]
        if row["chain_broken"]:
            any_broken = True
            writer.writerow([f"# BROKEN_CHAIN: {row['sell_hash']}"] + data_row)
        else:
            writer.writerow(data_row)
        if not math.isnan(row["gain_loss_usd"]):
            total_gain += row["gain_loss_usd"] * rate

    writer.writerow(["# TOTAL", "", "", "", "", "", "", "", "", f"{total_gain:.2f}"])
    return any_broken


def _write_us1099b(
    writer: csv.writer,
    matched: List[Dict[str, Any]],
) -> bool:
    """Write us-1099b-format rows.  Returns True if any chain-broken rows."""
    header = [
        "date_acquired", "date_sold",
        "proceeds_usd", "cost_basis_usd", "gain_loss_usd",
        "short_or_long_term",
        "fill_id", "audit_hash",
    ]
    writer.writerow(header)
    any_broken = False
    import math
    total_gain = 0.0
    for row in matched:
        buy_date = row["buy_date"]
        sell_date = row["sell_date"]
        if buy_date is not None:
            holding_days = (sell_date - buy_date).days
            term = "short" if holding_days < 365 else "long"
        else:
            term = "short"  # conservative default when lot is unknown

        gain = row["gain_loss_usd"]
        data_row = [
            buy_date.isoformat() if buy_date else "NaN",
            sell_date.isoformat(),
            f"{row['proceeds_usd']:.8f}",
            "NaN" if math.isnan(row["cost_basis_usd"]) else f"{row['cost_basis_usd']:.8f}",
            "NaN" if math.isnan(gain) else f"{gain:.8f}",
            term,
            row["sell_fill_id"],
            row["sell_hash"],
        ]
        if row["chain_broken"]:
            any_broken = True
            writer.writerow([f"# BROKEN_CHAIN: {row['sell_hash']}"] + data_row)
        else:
            writer.writerow(data_row)
        if not math.isnan(gain):
            total_gain += gain

    writer.writerow(["# TOTAL", "", f"{total_gain:.8f}", "", "", "", "", ""])
    return any_broken


def _write_generic(
    writer: csv.writer,
    fills: List[Dict[str, Any]],
) -> bool:
    """Write all raw fill fields.  Returns True if any chain-broken rows."""
    if not fills:
        # Still write a minimal header
        writer.writerow(["ts", "side", "filled_amount", "avg_fill_price", "fee", "pnl",
                         "order_id", "audit_hash"])
        return False

    # Collect all keys (excluding internal _ keys)
    all_keys = []
    seen = set()
    for fill in fills:
        for k in fill:
            if not k.startswith("_") and k not in seen:
                all_keys.append(k)
                seen.add(k)
    # Add our synthetic columns at the end
    all_keys += ["audit_hash"]
    writer.writerow(all_keys)

    any_broken = False
    for fill in fills:
        row = [fill.get(k, "") for k in all_keys[:-1]] + [fill["_audit_hash"]]
        if fill["_chain_broken"]:
            any_broken = True
            writer.writerow([f"# BROKEN_CHAIN"] + row)
        else:
            writer.writerow(row)

    return any_broken


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export tax/accounting CSV from audit log fills."
    )
    parser.add_argument("--year", type=int, required=True, help="Calendar year (UTC)")
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["korea", "us-1099b", "generic"],
        required=True,
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--audit-log",
        default=str(_DEFAULT_AUDIT_LOG),
        help=f"Path to audit.jsonl (default: {_DEFAULT_AUDIT_LOG})",
    )
    args = parser.parse_args()

    fills = _read_fills(args.audit_log, args.year)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not fills:
        print(
            f"WARNING: no fill records found for year {args.year} in {args.audit_log}",
            file=sys.stderr,
        )
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if args.fmt == "korea":
                w.writerow(["date", "symbol", "side", "qty", "price_krw", "fee_krw",
                             "exchange_rate_used", "fill_id", "audit_hash", "holding_period_days"])
            elif args.fmt == "us-1099b":
                w.writerow(["date_acquired", "date_sold", "proceeds_usd", "cost_basis_usd",
                             "gain_loss_usd", "short_or_long_term", "fill_id", "audit_hash"])
            else:
                w.writerow(["ts", "side", "filled_amount", "avg_fill_price", "fee", "pnl",
                             "order_id", "audit_hash"])
        sys.exit(0)

    any_broken = False
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if args.fmt == "generic":
            any_broken = _write_generic(writer, fills)
        else:
            matched = _fifo_match(fills)
            fx_rates = _load_fx_rates()
            if args.fmt == "korea":
                any_broken = _write_korea(writer, matched, fx_rates)
            else:
                any_broken = _write_us1099b(writer, matched)

    print(f"Wrote {output_path}", file=sys.stderr)
    sys.exit(1 if any_broken else 0)


if __name__ == "__main__":
    main()
