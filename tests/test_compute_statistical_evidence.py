"""Unit and smoke tests for scripts/compute_statistical_evidence.py."""

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# Script-under-test lives in scripts/, not a package — import via sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compute_statistical_evidence import (
    _compute,
    _returns_from_csv,
    _returns_from_log,
    bootstrap_ci,
    dsr,
    main,
    net_sharpe,
    permutation_p,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _fold_repr(
    fold_idx: int = 0,
    oos_total_return: float = 0.02,
    oos_total_return_random: float = 0.018,
) -> str:
    return (
        f"FoldResult(fold_idx={fold_idx}, train_size=600, test_size=120, "
        f"is_sharpe=0.5, oos_sharpe=0.3, oos_max_drawdown=0.05, "
        f"oos_total_return={oos_total_return}, oos_sharpe_random=0.25, "
        f"oos_total_return_random={oos_total_return_random}, "
        f"oos_trade_count_mean=2.0, oos_trade_count_random_mean=2.5, metrics={{}})"
    )


def _make_log(folds_returns: list[tuple[float, float]], tmp_path: Path) -> Path:
    """Write a minimal WF log with given (fixed, random) returns per fold."""
    reprs = [
        _fold_repr(fold_idx=i, oos_total_return=f, oos_total_return_random=r)
        for i, (f, r) in enumerate(folds_returns)
    ]
    content = "=== RESULT ===\nWalkForwardResult(folds=[" + ", ".join(reprs) + "])\n"
    p = tmp_path / "run.log"
    p.write_text(content, encoding="utf-8")
    return p


def _make_csv(folds_returns: list[tuple[float, float]], tmp_path: Path) -> Path:
    """Write a per-fold OOS return CSV."""
    p = tmp_path / "returns.csv"
    with open(p, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fold_idx", "oos_total_return", "oos_total_return_random"])
        writer.writeheader()
        for i, (f, r) in enumerate(folds_returns):
            writer.writerow({"fold_idx": i, "oos_total_return": f, "oos_total_return_random": r})
    return p


# ---------------------------------------------------------------------------
# Unit: net_sharpe
# ---------------------------------------------------------------------------

class TestNetSharpe:
    def test_all_positive_constant(self):
        # All returns identical → std = 0 → nan
        r = np.ones(12) * 0.02
        assert math.isnan(net_sharpe(r))

    def test_positive_with_variance(self):
        # All positive returns → sharpe should be positive
        r = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03] * 2, dtype=float)
        sr = net_sharpe(r)
        assert sr > 0.0

    def test_all_negative(self):
        r = np.array([-0.02] * 5 + [-0.01] * 7, dtype=float)
        sr = net_sharpe(r)
        assert sr < 0.0

    def test_single_observation(self):
        assert math.isnan(net_sharpe(np.array([0.05])))

    def test_known_value(self):
        # mean=0.02, std=0.01 (ddof=1), N=12 → sharpe = 0.02/0.01*sqrt(12)
        # Build returns where mean and std match (approximately)
        r = np.array([0.02 + 0.01 * (i - 5.5) / 3.0 for i in range(12)])
        sr = net_sharpe(r)
        expected = r.mean() / r.std(ddof=1) * math.sqrt(12)
        assert abs(sr - expected) < 1e-9


# ---------------------------------------------------------------------------
# Unit: dsr
# ---------------------------------------------------------------------------

class TestDSR:
    def test_strongly_positive_returns_positive_dsr(self):
        # Strong positive returns should sit above the sign-flip null
        r = np.array([0.05] * 6 + [0.03] * 6, dtype=float)
        d = dsr(r, n_permutations=1000, rng=np.random.default_rng(0))
        assert d > 0.0

    def test_strongly_negative_returns_negative_dsr(self):
        r = np.array([-0.05] * 6 + [-0.03] * 6, dtype=float)
        d = dsr(r, n_permutations=1000, rng=np.random.default_rng(0))
        assert d < 0.0

    def test_constant_returns_nan(self):
        r = np.ones(12) * 0.01
        assert math.isnan(dsr(r))

    def test_reproducible_with_seed(self):
        r = np.array([0.01 * i for i in range(1, 13)], dtype=float)
        d1 = dsr(r, n_permutations=200, rng=np.random.default_rng(99))
        d2 = dsr(r, n_permutations=200, rng=np.random.default_rng(99))
        assert d1 == d2


# ---------------------------------------------------------------------------
# Unit: bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_positive_ci_lower_positive(self):
        r = np.array([0.02, 0.03, 0.025, 0.01, 0.02, 0.015] * 2, dtype=float)
        _, lower, upper = bootstrap_ci(r, n_bootstrap=2000, rng=np.random.default_rng(7))
        assert lower > 0.0
        assert upper > lower

    def test_all_negative_ci_upper_negative(self):
        r = np.array([-0.02, -0.03, -0.01] * 4, dtype=float)
        _, lower, upper = bootstrap_ci(r, n_bootstrap=2000, rng=np.random.default_rng(7))
        assert upper < 0.0

    def test_mean_matches_sample_mean(self):
        r = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.01] * 2, dtype=float)
        mean, _, _ = bootstrap_ci(r, n_bootstrap=500, rng=np.random.default_rng(0))
        assert abs(mean - float(r.mean())) < 1e-12

    def test_ci_contains_mean(self):
        r = np.array([0.01 * i for i in range(1, 13)], dtype=float)
        mean, lower, upper = bootstrap_ci(r, n_bootstrap=2000, rng=np.random.default_rng(5))
        assert lower <= mean <= upper


# ---------------------------------------------------------------------------
# Unit: permutation_p
# ---------------------------------------------------------------------------

class TestPermutationP:
    def test_strongly_positive_small_p(self):
        # All-positive returns → almost no sign-flip permutation will match
        r = np.array([0.04, 0.05, 0.03, 0.06, 0.04, 0.05] * 2, dtype=float)
        p = permutation_p(r, n_permutations=1000, rng=np.random.default_rng(42))
        assert p < 0.05

    def test_zero_mean_p_near_half(self):
        # Symmetric returns around 0 → p ≈ 0.5
        r = np.array([-0.02, 0.02, -0.01, 0.01, -0.03, 0.03] * 2, dtype=float)
        p = permutation_p(r, n_permutations=2000, rng=np.random.default_rng(42))
        assert 0.3 < p < 0.7

    def test_output_in_range(self):
        r = np.array([0.01 * i for i in range(1, 13)], dtype=float)
        p = permutation_p(r, n_permutations=500, rng=np.random.default_rng(0))
        assert 0.0 < p <= 1.0

    def test_formula_denominator(self):
        # With n_permutations=99 and count=0, p = 1/100 = 0.01
        r = np.ones(12) * 1.0  # large constant → no permutation can match
        p = permutation_p(r, n_permutations=99, rng=np.random.default_rng(0))
        assert p == pytest.approx(1 / 100, abs=1e-9)


# ---------------------------------------------------------------------------
# Smoke: full pipeline on synthetic log
# ---------------------------------------------------------------------------

class TestSmokePipeline:
    """End-to-end smoke test using a small synthetic log and CSV."""

    _RETURNS = [
        (0.020, 0.022),
        (0.015, 0.018),
        (-0.005, -0.003),
        (0.030, 0.028),
        (0.010, 0.012),
        (0.025, 0.020),
        (0.018, 0.017),
        (-0.008, -0.006),
        (0.022, 0.024),
        (0.014, 0.016),
        (0.028, 0.026),
        (0.005, 0.007),
    ]

    def test_log_and_csv_agree(self, tmp_path):
        log_dir = tmp_path / "log"
        csv_dir = tmp_path / "csv"
        log_dir.mkdir()
        csv_dir.mkdir()

        log_path = _make_log(self._RETURNS, log_dir)
        csv_path = _make_csv(self._RETURNS, csv_dir)

        fixed_log, random_log = _returns_from_log(log_path)
        fixed_csv, random_csv = _returns_from_csv(csv_path)

        np.testing.assert_allclose(fixed_log, fixed_csv, rtol=1e-9)
        np.testing.assert_allclose(random_log, random_csv, rtol=1e-9)

    def test_all_positive_returns_pass_gate_thresholds(self, tmp_path):
        """With clearly positive fold returns (with variance), all gate criteria should pass."""
        # Deliberately vary returns so std > 0 while keeping all values positive
        pos = [(0.02 + 0.005 * (i % 3), 0.02 + 0.005 * (i % 3)) for i in range(12)]
        log = _make_log(pos, tmp_path)
        _, random = _returns_from_log(log)

        result = _compute(random, seed=42, n_perm=1000, n_boot=1000)

        assert result["net_sharpe"] > 0.5, f"net_sharpe={result['net_sharpe']}"
        assert result["dsr"] > 0.0, f"dsr={result['dsr']}"
        assert result["bootstrap_ci_lower"] > 0.0, f"ci_lower={result['bootstrap_ci_lower']}"
        assert result["permutation_p"] < 0.05, f"perm_p={result['permutation_p']}"

    def test_cli_log_input(self, tmp_path):
        log = _make_log(self._RETURNS, tmp_path)
        out_path = tmp_path / "out.json"
        rc = main(["--log", str(log), "--output", str(out_path), "--seed", "42"])
        assert rc == 0
        data = json.loads(out_path.read_text())
        assert "random_start" in data
        assert "fixed_start" in data
        for section in ("random_start", "fixed_start"):
            assert "net_sharpe" in data[section]
            assert "dsr" in data[section]
            assert "bootstrap_ci_lower" in data[section]
            assert "permutation_p" in data[section]
            assert data[section]["n_folds"] == 12

    def test_cli_csv_input(self, tmp_path):
        csv_path = _make_csv(self._RETURNS, tmp_path)
        rc = main(["--returns-csv", str(csv_path), "--seed", "42"])
        assert rc == 0

    def test_mixed_returns_sensible_output(self, tmp_path):
        """Mixed-sign returns: CI should span zero, p should be moderate."""
        log = _make_log(self._RETURNS, tmp_path)
        _, random = _returns_from_log(log)
        result = _compute(random, seed=42, n_perm=1000, n_boot=1000)

        assert result["bootstrap_ci_lower"] < result["bootstrap_ci_upper"]
        assert 0.0 < result["permutation_p"] <= 1.0
        assert result["n_folds"] == 12
