"""Unit tests for scripts/aggregate_wf_results.py.

Tests cover:
- Log parsing (RESULT block extraction, FoldResult field parsing)
- Edge cases (missing block, truncated log, nan/inf values)
- apply_criterion logic (criterion 1, criterion 2, stop)
- CLI argument handling
"""

import math
import textwrap
from pathlib import Path

import pytest

# Script-under-test lives in scripts/, not a package — import via sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from aggregate_wf_results import (
    ParsedFold,
    VariantResult,
    apply_criterion,
    main,
    parse_log,
    _parse_fold_range,
    _FOLD_RE,
    print_maxdd_table,
    print_fixed_start_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fold_repr(
    fold_idx: int = 0,
    train_size: int = 600,
    test_size: int = 120,
    is_sharpe: float = 0.5,
    oos_sharpe: float = 0.3,
    oos_max_drawdown: float = 0.05,
    oos_total_return: float = 0.02,
    oos_sharpe_random: float = 0.25,
    oos_total_return_random: float = 0.018,
    oos_trade_count_mean: float = 2.0,
    oos_trade_count_random_mean: float = 2.5,
    metrics: str = "{}",
) -> str:
    return (
        f"FoldResult(fold_idx={fold_idx}, train_size={train_size}, "
        f"test_size={test_size}, is_sharpe={is_sharpe}, oos_sharpe={oos_sharpe}, "
        f"oos_max_drawdown={oos_max_drawdown}, oos_total_return={oos_total_return}, "
        f"oos_sharpe_random={oos_sharpe_random}, "
        f"oos_total_return_random={oos_total_return_random}, "
        f"oos_trade_count_mean={oos_trade_count_mean}, "
        f"oos_trade_count_random_mean={oos_trade_count_random_mean}, "
        f"metrics={metrics})"
    )


def _make_result_block(n_folds: int = 3, **fold_overrides) -> str:
    """Build a minimal log snippet with a RESULT block."""
    folds = [_fold_repr(fold_idx=i, **fold_overrides) for i in range(n_folds)]
    fold_list = ", ".join(folds)
    return f"=== RESULT ===\nWalkForwardResult(folds=[{fold_list}])\n"


def _write_log(tmp_path: Path, content: str, filename: str = "run.log") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_log: happy path
# ---------------------------------------------------------------------------

class TestParseLogHappyPath:
    def test_basic_three_folds(self, tmp_path):
        log = _write_log(tmp_path, _make_result_block(n_folds=3))
        folds = parse_log(log)
        assert len(folds) == 3
        assert [f.fold_idx for f in folds] == [0, 1, 2]

    def test_field_values_round_trip(self, tmp_path):
        content = _make_result_block(
            n_folds=1,
            is_sharpe=1.23,
            oos_sharpe=-0.45,
            oos_max_drawdown=0.07,
            oos_total_return=-0.03,
            oos_sharpe_random=0.11,
            oos_total_return_random=0.009,
            oos_trade_count_mean=3.5,
            oos_trade_count_random_mean=4.2,
        )
        log = _write_log(tmp_path, content)
        (fold,) = parse_log(log)
        assert fold.fold_idx == 0
        assert abs(fold.is_sharpe - 1.23) < 1e-9
        assert abs(fold.oos_sharpe - (-0.45)) < 1e-9
        assert abs(fold.oos_total_return_random - 0.009) < 1e-9
        assert abs(fold.oos_trade_count_random_mean - 4.2) < 1e-9

    def test_twelve_folds(self, tmp_path):
        log = _write_log(tmp_path, _make_result_block(n_folds=12))
        folds = parse_log(log)
        assert len(folds) == 12

    def test_folds_sorted_by_index(self, tmp_path):
        """Folds may appear in any order in the repr; parser must sort."""
        folds_repr = [_fold_repr(fold_idx=i) for i in [2, 0, 1]]
        block = "=== RESULT ===\nWalkForwardResult(folds=[" + ", ".join(folds_repr) + "])\n"
        log = _write_log(tmp_path, block)
        folds = parse_log(log)
        assert [f.fold_idx for f in folds] == [0, 1, 2]

    def test_scientific_notation(self, tmp_path):
        content = _make_result_block(n_folds=1, oos_total_return_random=1.23e-04)
        log = _write_log(tmp_path, content)
        (fold,) = parse_log(log)
        assert abs(fold.oos_total_return_random - 1.23e-04) < 1e-12

    def test_negative_scientific_notation(self, tmp_path):
        content = _make_result_block(n_folds=1, oos_total_return_random=-3.5e-03)
        log = _write_log(tmp_path, content)
        (fold,) = parse_log(log)
        assert abs(fold.oos_total_return_random - (-3.5e-03)) < 1e-12

    def test_nan_values(self, tmp_path):
        content = _make_result_block(n_folds=1, oos_trade_count_random_mean=float("nan"))
        # float("nan") renders as 'nan' in repr
        log = _write_log(tmp_path, content.replace("oos_trade_count_random_mean=nan", "oos_trade_count_random_mean=nan"))
        folds = parse_log(log)
        assert math.isnan(folds[0].oos_trade_count_random_mean)

    def test_last_result_block_used(self, tmp_path):
        """If log contains two RESULT blocks (re-run), take the last."""
        block1 = _make_result_block(n_folds=1, oos_total_return_random=0.01)
        block2 = _make_result_block(n_folds=2, oos_total_return_random=0.05)
        log = _write_log(tmp_path, "preamble\n" + block1 + "more output\n" + block2)
        folds = parse_log(log)
        assert len(folds) == 2
        for f in folds:
            assert abs(f.oos_total_return_random - 0.05) < 1e-9

    def test_log_with_preamble_lines(self, tmp_path):
        preamble = "Config: base.yaml + override from config/futures_maker.yaml\n"
        preamble += "Effective env: cost_model=futures_maker, trading_fee=0.00018\n"
        log = _write_log(tmp_path, preamble + _make_result_block(n_folds=3))
        folds = parse_log(log)
        assert len(folds) == 3


# ---------------------------------------------------------------------------
# parse_log: error cases
# ---------------------------------------------------------------------------

class TestParseLogEncodings:
    """PowerShell on Windows defaults to UTF-16 LE for `*>` redirects;
    parser must handle that and other BOM variants transparently.
    """

    def test_utf16_le_log_parses(self, tmp_path):
        block = _make_result_block(n_folds=2)
        text = "log start\n" + block + "\n"
        path = tmp_path / "B0_utf16le.log"
        path.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        folds = parse_log(path)
        assert len(folds) == 2

    def test_utf16_be_log_parses(self, tmp_path):
        block = _make_result_block(n_folds=2)
        text = "log start\n" + block + "\n"
        path = tmp_path / "B0_utf16be.log"
        path.write_bytes(b"\xfe\xff" + text.encode("utf-16-be"))
        folds = parse_log(path)
        assert len(folds) == 2

    def test_utf8_with_bom_log_parses(self, tmp_path):
        block = _make_result_block(n_folds=2)
        text = "log start\n" + block + "\n"
        path = tmp_path / "B0_utf8bom.log"
        path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
        folds = parse_log(path)
        assert len(folds) == 2


class TestParseLogErrors:
    def test_missing_result_block_raises(self, tmp_path):
        log = _write_log(tmp_path, "Training complete. No result block here.\n")
        with pytest.raises(ValueError, match="RESULT"):
            parse_log(log)

    def test_zero_folds_after_marker_raises(self, tmp_path):
        log = _write_log(tmp_path, "=== RESULT ===\nWalkForwardResult(folds=[])\n")
        with pytest.raises(ValueError, match="0 FoldResults"):
            parse_log(log)

    def test_truncated_log_raises(self, tmp_path):
        """Log ends right after the marker, before any FoldResult."""
        log = _write_log(tmp_path, "=== RESULT ===\nWalkForward")
        with pytest.raises(ValueError):
            parse_log(log)


# ---------------------------------------------------------------------------
# VariantResult aggregates
# ---------------------------------------------------------------------------

def _make_variant(name: str, returns_random: list, trades_random: list = None) -> VariantResult:
    if trades_random is None:
        trades_random = [3.0] * len(returns_random)
    folds = [
        ParsedFold(
            fold_idx=i,
            is_sharpe=0.0,
            oos_sharpe=0.0,
            oos_max_drawdown=0.0,
            oos_total_return=r,
            oos_sharpe_random=0.0,
            oos_total_return_random=r,
            oos_trade_count_mean=t,
            oos_trade_count_random_mean=t,
        )
        for i, (r, t) in enumerate(zip(returns_random, trades_random))
    ]
    return VariantResult(name=name, folds=folds)


class TestVariantAggregates:
    def test_all_mean_random(self):
        v = _make_variant("X", [0.01, -0.02, 0.03])
        assert abs(v.all_mean_random() - (0.01 - 0.02 + 0.03) / 3) < 1e-9

    def test_bull_mean_random(self):
        v = _make_variant("X", [0.05, 0.04, -0.03, -0.01])
        bull_mean = v.bull_mean_random([0, 1])
        assert abs(bull_mean - 0.045) < 1e-9

    def test_bear_mean_random(self):
        v = _make_variant("X", [0.05, 0.04, -0.03, -0.01])
        bear_mean = v.bear_mean_random([2, 3])
        assert abs(bear_mean - (-0.02)) < 1e-9

    def test_folds_positive(self):
        v = _make_variant("X", [0.01, -0.02, 0.03, -0.01, 0.02])
        assert v.folds_positive() == 3

    def test_trades_per_ep(self):
        v = _make_variant("X", [0.0] * 3, trades_random=[2.0, 4.0, 6.0])
        assert abs(v.trades_per_ep() - 4.0) < 1e-9

    def test_nan_excluded_from_mean(self):
        v = _make_variant("X", [0.04, float("nan"), -0.02])
        mean = v.all_mean_random()
        assert abs(mean - (0.04 - 0.02) / 2) < 1e-9


# ---------------------------------------------------------------------------
# apply_criterion
# ---------------------------------------------------------------------------

BULL = list(range(0, 5))
BEAR = list(range(5, 12))


def _twelve_folds(returns_random: list, trades: float) -> VariantResult:
    assert len(returns_random) == 12
    return _make_variant("?", returns_random, [trades] * 12)


class TestApplyCriterion:
    def test_criterion1_wins(self):
        b0 = _make_variant("B0", [0.02] * 12, [2.0] * 12)
        b2 = _make_variant("B2", [0.03] * 12, [6.0] * 12)  # trades>5, bull >= B0
        variants = [b0, b2]
        winner, criterion = apply_criterion(variants, BULL, BEAR, "B0")
        assert winner == "B2"
        assert "Criterion 1" in criterion

    def test_criterion1_b2_preferred_over_b3(self):
        """Tie-break: B2 (config-only) preferred over B3."""
        b0 = _make_variant("B0", [0.01] * 12, [2.0] * 12)
        b2 = _make_variant("B2", [0.01] * 12, [6.0] * 12)
        b3 = _make_variant("B3", [0.01] * 12, [6.0] * 12)
        variants = [b0, b2, b3]
        winner, _ = apply_criterion(variants, BULL, BEAR, "B0")
        assert winner == "B2"

    def test_criterion2_activates_when_criterion1_fails(self):
        """trades<=5 for all, but one has better bear than B0."""
        b0_returns = [-0.05] * 12
        b1_returns = [-0.05] * 12
        b1_returns[5] = -0.03  # slightly better in one bear fold
        b0 = _make_variant("B0", b0_returns, [2.0] * 12)
        b1 = _make_variant("B1", b1_returns, [4.0] * 12)  # trades>3
        variants = [b0, b1]
        winner, criterion = apply_criterion(variants, BULL, BEAR, "B0")
        assert winner == "B1"
        assert "Criterion 2" in criterion

    def test_stop_when_no_variant_qualifies(self):
        b0 = _make_variant("B0", [-0.03] * 12, [2.0] * 12)
        b1 = _make_variant("B1", [-0.04] * 12, [2.0] * 12)  # bear worse than B0
        variants = [b0, b1]
        winner, criterion = apply_criterion(variants, BULL, BEAR, "B0")
        assert winner is None
        assert "STOP" in criterion

    def test_b0_excluded_from_selection(self):
        """B0 should never be selected as the winner."""
        b0 = _make_variant("B0", [0.05] * 12, [8.0] * 12)
        variants = [b0]
        winner, criterion = apply_criterion(variants, BULL, BEAR, "B0")
        assert winner is None

    def test_criterion1_bull_check_vs_b0(self):
        """B2 has high trades but bull mean < B0 bull → falls to criterion 2."""
        b0_rets = [0.05] * 5 + [-0.03] * 7  # bull=5%, bear=-3%
        b2_rets = [0.01] * 5 + [-0.01] * 7  # bull=1% < b0_bull
        b0 = _make_variant("B0", b0_rets, [2.0] * 12)
        b2 = _make_variant("B2", b2_rets, [6.0] * 12)  # trades>5 but bull < B0
        variants = [b0, b2]
        winner, criterion = apply_criterion(variants, BULL, BEAR, "B0")
        # Criterion 1 requires bull >= B0 bull (0.05); b2 bull=0.01 < 0.05 → fail criterion 1
        # Criterion 2: trades>3 AND bear > B0 bear (-0.03): b2 bear=-0.01 > -0.03 → passes
        assert winner == "B2"
        assert "Criterion 2" in criterion


# ---------------------------------------------------------------------------
# _parse_fold_range
# ---------------------------------------------------------------------------

class TestParseFoldRange:
    def test_range_notation(self):
        assert _parse_fold_range("0-4") == [0, 1, 2, 3, 4]

    def test_comma_notation(self):
        assert _parse_fold_range("0,1,2") == [0, 1, 2]

    def test_single_value(self):
        assert _parse_fold_range("3") == [3]

    def test_bull_folds_default(self):
        assert _parse_fold_range("0,2,5,8,11") == [0, 2, 5, 8, 11]

    def test_bear_folds_default(self):
        assert _parse_fold_range("1,3,4,6,7,9,10") == [1, 3, 4, 6, 7, 9, 10]

    def test_bull_bear_defaults_are_disjoint_and_cover_twelve_folds(self):
        bull = _parse_fold_range("0,2,5,8,11")
        bear = _parse_fold_range("1,3,4,6,7,9,10")
        assert set(bull) & set(bear) == set()
        assert set(bull) | set(bear) == set(range(12))


# ---------------------------------------------------------------------------
# CLI integration (uses tmp_path, writes real log files)
# ---------------------------------------------------------------------------

class TestCLI:
    def _write_variant_log(self, tmp_path: Path, name: str, returns: list, trades: float) -> Path:
        folds_repr = [
            _fold_repr(
                fold_idx=i,
                oos_total_return_random=r,
                oos_trade_count_random_mean=trades,
            )
            for i, r in enumerate(returns)
        ]
        block = "=== RESULT ===\nWalkForwardResult(folds=[" + ", ".join(folds_repr) + "])\n"
        return _write_log(tmp_path, block, filename=f"{name}.log")

    def test_basic_table_output(self, tmp_path, capsys):
        b0 = self._write_variant_log(tmp_path, "B0", [-0.01] * 12, 2.0)
        b2 = self._write_variant_log(tmp_path, "B2", [0.01] * 12, 6.0)
        rc = main([
            "--logs", str(b0), str(b2),
            "--variant-names", "B0", "B2",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "B0" in out
        assert "B2" in out

    def test_apply_criterion_output(self, tmp_path, capsys):
        b0 = self._write_variant_log(tmp_path, "B0", [0.01] * 12, 2.0)
        b2 = self._write_variant_log(tmp_path, "B2", [0.02] * 12, 6.0)
        rc = main([
            "--logs", str(b0), str(b2),
            "--variant-names", "B0", "B2",
            "--apply-criterion",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Winner" in out or "Criterion" in out

    def test_missing_log_returns_nonzero(self, tmp_path, capsys):
        b0 = self._write_variant_log(tmp_path, "B0", [0.01] * 3, 2.0)
        missing = tmp_path / "ghost.log"
        rc = main([
            "--logs", str(b0), str(missing),
            "--variant-names", "B0", "GHOST",
        ])
        assert rc > 0  # at least one error

    def test_log_dir_flag(self, tmp_path, capsys):
        self._write_variant_log(tmp_path, "B0", [0.0] * 3, 2.0)
        self._write_variant_log(tmp_path, "B2", [0.0] * 3, 5.0)
        rc = main(["--log-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "B0" in out

    def test_variant_name_count_mismatch(self, tmp_path, capsys):
        b0 = self._write_variant_log(tmp_path, "B0", [0.0] * 3, 2.0)
        rc = main([
            "--logs", str(b0),
            "--variant-names", "B0", "extra",
        ])
        assert rc != 0

    def test_detail_flag(self, tmp_path, capsys):
        b0 = self._write_variant_log(tmp_path, "B0", [0.01, -0.02, 0.03], 3.0)
        rc = main([
            "--logs", str(b0),
            "--variant-names", "B0",
            "--detail", "B0",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Per-fold detail" in out


# ---------------------------------------------------------------------------
# MaxDD aggregation — synthetic capital path exercise
# ---------------------------------------------------------------------------

def _make_variant_with_drawdowns(name: str, drawdowns: list) -> VariantResult:
    """Build a VariantResult where each fold has the given oos_max_drawdown value."""
    folds = [
        ParsedFold(
            fold_idx=i,
            is_sharpe=0.0,
            oos_sharpe=0.0,
            oos_max_drawdown=dd,
            oos_total_return=0.0,
            oos_sharpe_random=0.0,
            oos_total_return_random=0.0,
            oos_trade_count_mean=2.0,
            oos_trade_count_random_mean=2.0,
        )
        for i, dd in enumerate(drawdowns)
    ]
    return VariantResult(name=name, folds=folds)


class TestMaxDDStats:
    """Synthetic capital path: 12 folds with known drawdown values.

    Drawdowns represent peak-to-trough fractions (e.g. 0.10 = 10%).
    We verify mean, median, and p95 for all / bull / bear regimes.
    """

    # 12-fold drawdowns (indices 0-11); bull = 0,2,5,8,11 (5 folds), bear = rest (7 folds)
    _DD = [0.05, 0.20, 0.08, 0.15, 0.25, 0.10, 0.18, 0.22, 0.12, 0.30, 0.06, 0.04]
    _BULL = [0, 2, 5, 8, 11]   # dd: 0.05, 0.08, 0.10, 0.12, 0.04
    _BEAR = [1, 3, 4, 6, 7, 9, 10]  # dd: 0.20, 0.15, 0.25, 0.18, 0.22, 0.30, 0.06

    def _v(self):
        return _make_variant_with_drawdowns("X", self._DD)

    def test_all_mean(self):
        mean, _, _ = self._v().maxdd_stats()
        assert abs(mean - sum(self._DD) / 12) < 1e-9

    def test_all_median(self):
        _, median, _ = self._v().maxdd_stats()
        sorted_dd = sorted(self._DD)
        expected = (sorted_dd[5] + sorted_dd[6]) / 2  # n=12, even
        assert abs(median - expected) < 1e-9

    def test_all_p95(self):
        import math as _math
        _, _, p95 = self._v().maxdd_stats()
        sorted_dd = sorted(self._DD)
        idx = min(int(_math.ceil(0.95 * 12)) - 1, 11)  # ceil(11.4)-1 = 11
        assert abs(p95 - sorted_dd[idx]) < 1e-9

    def test_bull_mean(self):
        bull_dd = [self._DD[i] for i in self._BULL]
        mean, _, _ = self._v().maxdd_stats(self._BULL)
        assert abs(mean - sum(bull_dd) / len(bull_dd)) < 1e-9

    def test_bear_mean(self):
        bear_dd = [self._DD[i] for i in self._BEAR]
        mean, _, _ = self._v().maxdd_stats(self._BEAR)
        assert abs(mean - sum(bear_dd) / len(bear_dd)) < 1e-9

    def test_empty_indices_returns_nan(self):
        mean, median, p95 = self._v().maxdd_stats([])
        assert math.isnan(mean) and math.isnan(median) and math.isnan(p95)

    def test_nan_drawdown_excluded(self):
        dd = [0.10, float("nan"), 0.20]
        v = _make_variant_with_drawdowns("Y", dd)
        mean, _, _ = v.maxdd_stats()
        assert abs(mean - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# Phase 8-Gamma G1: gate metric fields + backward-compat
# ---------------------------------------------------------------------------

class TestGateMetricFields:
    def test_parse_fold_with_gate_fields(self):
        """Parser correctly captures new gate metric fields."""
        text = (
            "FoldResult(fold_idx=0, train_size=100, test_size=20, "
            "is_sharpe=0.5, oos_sharpe=0.3, oos_max_drawdown=0.05, "
            "oos_total_return=0.01, oos_sharpe_random=0.2, "
            "oos_total_return_random=0.005, oos_trade_count_mean=2.0, "
            "oos_trade_count_random_mean=2.5, "
            "oos_mean_gate_fires_per_episode=12.3, "
            "oos_mean_gate_active_fraction=0.15, "
            "metrics={})"
        )
        matches = list(_FOLD_RE.finditer(text))
        assert len(matches) == 1
        g = matches[0].groupdict()
        assert abs(float(g["gate_fires"]) - 12.3) < 1e-9
        assert abs(float(g["gate_frac"]) - 0.15) < 1e-9

    def test_maxdd_table_prints_without_error(self, capsys):
        """print_maxdd_table runs without error and emits expected column headers."""
        v = _make_variant("G2", [0.01] * 12)
        # Override drawdowns to known values so we can check output
        for i, f in enumerate(v.folds):
            object.__setattr__(f, "oos_max_drawdown", (i + 1) * 0.01)
        print_maxdd_table([v], bull_indices=list(range(5)), bear_indices=list(range(5, 12)))
        out = capsys.readouterr().out
        assert "MaxDD" in out
        assert "G2" in out

    def test_parse_fold_without_gate_fields_backward_compat(self):
        """Old logs without gate fields still parse (backward-compat)."""
        text = (
            "FoldResult(fold_idx=0, train_size=100, test_size=20, "
            "is_sharpe=0.5, oos_sharpe=0.3, oos_max_drawdown=0.05, "
            "oos_total_return=0.01, oos_sharpe_random=0.2, "
            "oos_total_return_random=0.005, oos_trade_count_mean=2.0, "
            "oos_trade_count_random_mean=2.5, metrics={})"
        )
        matches = list(_FOLD_RE.finditer(text))
        assert len(matches) == 1
        g = matches[0].groupdict()
        assert g.get("gate_fires") is None
        assert g.get("gate_frac") is None


# ---------------------------------------------------------------------------
# Fixed-start stats (oos_total_return field)
# ---------------------------------------------------------------------------

def _make_variant_fixed(name: str, fixed_returns: list) -> VariantResult:
    """Build a VariantResult with distinct fixed-start (oos_total_return) values.

    oos_total_return_random is deliberately zeroed so tests confirm the
    fixed-start methods read from the correct field.
    """
    folds = [
        ParsedFold(
            fold_idx=i,
            is_sharpe=0.0,
            oos_sharpe=0.0,
            oos_max_drawdown=0.0,
            oos_total_return=r,
            oos_sharpe_random=0.0,
            oos_total_return_random=0.0,  # distinct from fixed — ensures correct field is read
            oos_trade_count_mean=2.0,
            oos_trade_count_random_mean=2.0,
        )
        for i, r in enumerate(fixed_returns)
    ]
    return VariantResult(name=name, folds=folds)


class TestFixedStartStats:
    def test_all_mean(self):
        v = _make_variant_fixed("X", [0.01, -0.02, 0.03])
        assert abs(v.all_mean() - (0.01 - 0.02 + 0.03) / 3) < 1e-9

    def test_bull_mean(self):
        v = _make_variant_fixed("X", [0.05, 0.04, -0.03, -0.01])
        assert abs(v.bull_mean([0, 1]) - 0.045) < 1e-9

    def test_bear_mean(self):
        v = _make_variant_fixed("X", [0.05, 0.04, -0.03, -0.01])
        assert abs(v.bear_mean([2, 3]) - (-0.02)) < 1e-9

    def test_folds_positive_fixed(self):
        v = _make_variant_fixed("X", [0.01, -0.02, 0.03, -0.01, 0.02])
        assert v.folds_positive_fixed() == 3


class TestFixedStartTable:
    def test_header_and_variant_rows_present(self, capsys):
        g2 = _make_variant_fixed("G2", [0.01, -0.02, 0.03] * 4)
        b0 = _make_variant_fixed("B0", [-0.01] * 12)
        print_fixed_start_table(
            [g2, b0],
            bull_indices=[0, 2, 4, 6, 8],
            bear_indices=[1, 3, 5, 7, 9, 10, 11],
        )
        out = capsys.readouterr().out
        assert "Fixed-start" in out
        assert "G2" in out
        assert "B0" in out
