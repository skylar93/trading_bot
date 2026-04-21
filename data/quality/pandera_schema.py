"""
OHLCV Pandera schema — Week 79 (H6).

Replaces the ad-hoc checks in DataQualityGate with a declarative
pandera DataFrameSchema that is composable, reportable, and portable.

Usage::

    from data.quality.pandera_schema import OHLCV_SCHEMA, validate_ohlcv

    # Raises pandera.errors.SchemaError on first failure (default)
    clean_df = OHLCV_SCHEMA.validate(df)

    # Collect all errors without raising
    issues = validate_ohlcv(df)          # → list[str], empty means clean

The schema enforces four invariants that map directly to the H6 plan:
- not_null      : no NaN in price/volume columns
- positive      : $open/$high/$low/$close/$volume > 0
- monotonic_ts  : DatetimeIndex (or 'timestamp' column) strictly increasing
- no_gap        : consecutive bar intervals within 3× the median interval
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import DataFrameSchema, Column, Check, Index
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    pa = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Custom checks
# ---------------------------------------------------------------------------

def _check_positive(series: pd.Series) -> pd.Series:
    """Element-wise: value > 0 (NaN treated as failure upstream by not_nullable)."""
    return series > 0


def _check_no_inf(series: pd.Series) -> pd.Series:
    """Element-wise: reject ±inf."""
    return ~series.isin([np.inf, -np.inf])


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

def _build_schema() -> "pa.DataFrameSchema | None":
    if not HAS_PANDERA:
        return None

    return pa.DataFrameSchema(
        columns={
            "$open": pa.Column(
                float,
                checks=[
                    pa.Check(_check_no_inf, element_wise=True, error="$open contains ±inf"),
                    pa.Check(_check_positive, element_wise=True, error="$open must be > 0"),
                ],
                nullable=False,
            ),
            "$high": pa.Column(
                float,
                checks=[
                    pa.Check(_check_no_inf, element_wise=True, error="$high contains ±inf"),
                    pa.Check(_check_positive, element_wise=True, error="$high must be > 0"),
                ],
                nullable=False,
            ),
            "$low": pa.Column(
                float,
                checks=[
                    pa.Check(_check_no_inf, element_wise=True, error="$low contains ±inf"),
                    pa.Check(_check_positive, element_wise=True, error="$low must be > 0"),
                ],
                nullable=False,
            ),
            "$close": pa.Column(
                float,
                checks=[
                    pa.Check(_check_no_inf, element_wise=True, error="$close contains ±inf"),
                    pa.Check(_check_positive, element_wise=True, error="$close must be > 0"),
                ],
                nullable=False,
            ),
            "$volume": pa.Column(
                float,
                checks=[
                    pa.Check(_check_no_inf, element_wise=True, error="$volume contains ±inf"),
                    pa.Check(_check_positive, element_wise=True, error="$volume must be > 0"),
                ],
                nullable=False,
            ),
        },
        coerce=False,
        strict=False,  # allow extra columns
    )


OHLCV_SCHEMA: "pa.DataFrameSchema | None" = _build_schema()

# ---------------------------------------------------------------------------
# Monotonic-timestamp check (dataframe-level, not column-level)
# ---------------------------------------------------------------------------

_TIME_COL_NAMES = {"timestamp", "time", "date", "datetime"}


def _get_time_series(df: pd.DataFrame) -> "pd.Series | None":
    """Return the datetime axis as a Series, or None if unavailable."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)
    for col in df.columns:
        if col.lower() in _TIME_COL_NAMES:
            try:
                return pd.to_datetime(df[col]).reset_index(drop=True)
            except Exception:
                pass
    return None


def _check_monotonic_ts(df: pd.DataFrame) -> List[str]:
    """Return list of error messages for non-monotonic timestamps."""
    errors: List[str] = []
    ts = _get_time_series(df)
    if ts is None or len(ts) < 2:
        return errors
    diffs = ts.diff().dropna()
    non_positive = diffs[diffs <= pd.Timedelta(0)]
    for idx, diff in non_positive.items():
        errors.append(
            f"[monotonic_ts] row={int(idx)}: timestamp not strictly increasing "
            f"(diff={diff})"
        )
    return errors


def _check_no_gap(df: pd.DataFrame, max_gap_multiplier: float = 3.0) -> List[str]:
    """Return list of error messages for time gaps > max_gap_multiplier × median bar."""
    errors: List[str] = []
    ts = _get_time_series(df)
    if ts is None or len(ts) < 2:
        return errors
    diffs = ts.diff().dropna()
    if len(diffs) == 0:
        return errors
    median_diff = diffs.median()
    if median_diff.total_seconds() <= 0:
        return errors
    threshold = median_diff * max_gap_multiplier
    for idx, diff in diffs.items():
        if diff > threshold:
            errors.append(
                f"[no_gap] row={int(idx)}: gap={diff} > threshold={threshold} "
                f"(median_bar={median_diff})"
            )
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ohlcv(
    df: pd.DataFrame,
    max_gap_multiplier: float = 3.0,
) -> List[str]:
    """
    Validate *df* against the OHLCV schema.

    Returns
    -------
    list[str]
        Human-readable error messages.  Empty list means the data is clean.
    """
    errors: List[str] = []

    if HAS_PANDERA and OHLCV_SCHEMA is not None:
        try:
            OHLCV_SCHEMA.validate(df, lazy=True)
        except Exception as exc:
            # pandera SchemaErrors have a .schema_errors attribute
            if hasattr(exc, "schema_errors"):
                for e in exc.schema_errors:
                    errors.append(f"[pandera] {e.get('check', '')} — {e.get('failure_cases', '')}")
            else:
                errors.append(f"[pandera] {exc}")
    else:
        # Fallback without pandera: basic numpy checks
        errors.extend(_fallback_validate(df))

    errors.extend(_check_monotonic_ts(df))
    errors.extend(_check_no_gap(df, max_gap_multiplier))
    return errors


def _fallback_validate(df: pd.DataFrame) -> List[str]:
    """Pure-numpy fallback when pandera is not installed."""
    price_cols = ["$open", "$high", "$low", "$close", "$volume"]
    errors: List[str] = []
    for col in price_cols:
        if col not in df.columns:
            continue
        s = df[col]
        null_mask = s.isna()
        if null_mask.any():
            rows = df.index[null_mask].tolist()[:5]
            errors.append(f"[not_null] {col}: NaN at rows {rows}")
        inf_mask = s.isin([np.inf, -np.inf])
        if inf_mask.any():
            rows = df.index[inf_mask].tolist()[:5]
            errors.append(f"[no_inf] {col}: ±inf at rows {rows}")
        pos_mask = s <= 0
        if pos_mask.any():
            rows = df.index[pos_mask].tolist()[:5]
            errors.append(f"[positive] {col}: value ≤ 0 at rows {rows}")
    return errors
