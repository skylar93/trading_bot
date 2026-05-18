"""Compute statistical evidence metrics for live gate (B2-B5).

Reads a walk-forward log or per-fold OOS return CSV and emits:
  net_sharpe, dsr, bootstrap_ci_lower, permutation_p

These fields are consumed by deployment/governance/live_signal_gate.py.

Usage
-----
# From a walk-forward log:
python scripts/compute_statistical_evidence.py --log logs/G2_1M.log

# From a per-fold CSV (fold_idx, oos_total_return, oos_total_return_random):
python scripts/compute_statistical_evidence.py --returns-csv results.csv

# Write JSON to file:
python scripts/compute_statistical_evidence.py --log logs/G2_1M.log --output evidence.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import log parser from aggregate_wf_results (same scripts/ dir)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_wf_results import parse_log  # noqa: E402


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray) -> float:
    n = len(returns)
    if n < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    # Guard against zero / near-zero std (identical returns → infinite Sharpe → undefined)
    if std < 1e-12:
        return float("nan")
    return float(returns.mean() / std * math.sqrt(n))


def net_sharpe(returns: np.ndarray) -> float:
    """Sample Sharpe of fold returns: mean/std * sqrt(N)."""
    return _sharpe(returns)


def dsr(
    returns: np.ndarray,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Deflated Sharpe: z-score of observed Sharpe vs sign-flip null.

    Permutes return signs to build H0 distribution; returns how many
    std-devs the observed Sharpe sits above the null mean.
    """
    sr_obs = _sharpe(returns)
    if math.isnan(sr_obs):
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(42)

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(returns)))
    permuted_returns = signs * returns[np.newaxis, :]
    permuted_srs = np.array([_sharpe(row) for row in permuted_returns])
    valid = permuted_srs[~np.isnan(permuted_srs)]
    if len(valid) < 2:
        return float("nan")

    std = float(valid.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float((sr_obs - float(valid.mean())) / std)


def bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI of mean return (with replacement).

    Returns (mean, lower_2.5%, upper_97.5%).
    """
    n = len(returns)
    if rng is None:
        rng = np.random.default_rng(42)

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = returns[idx].mean(axis=1)
    return (
        float(returns.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def block_bootstrap_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_size: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Circular block bootstrap 95% CI for time-series data.

    Preserves autocorrelation between adjacent folds (walk-forward folds
    share training data → adjacent returns are correlated). Block size ≈
    sqrt(n) is a standard heuristic; default 4 for n=12.

    Returns (mean, lower_2.5%, upper_97.5%).
    """
    n = len(returns)
    if rng is None:
        rng = np.random.default_rng(42)

    n_blocks = math.ceil(n / block_size)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        sample = []
        for s in starts:
            for j in range(block_size):
                sample.append(returns[(s + j) % n])
        boot_means[i] = np.mean(sample[:n])

    return (
        float(returns.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def permutation_p(
    returns: np.ndarray,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sign-flip permutation p-value for H0: no positive edge.

    p = (count[permuted_mean >= observed_mean] + 1) / (n + 1)
    """
    obs_mean = float(returns.mean())
    if rng is None:
        rng = np.random.default_rng(42)

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, len(returns)))
    permuted_means = (signs * returns[np.newaxis, :]).mean(axis=1)
    count = int((permuted_means >= obs_mean).sum())
    return (count + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _returns_from_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fixed_start_returns, random_start_returns) from a WF log."""
    folds = parse_log(path)
    fixed = np.array([f.oos_total_return for f in folds], dtype=float)
    random = np.array([f.oos_total_return_random for f in folds], dtype=float)
    return fixed, random


def _returns_from_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fixed_start_returns, random_start_returns) from a CSV.

    Expected columns: fold_idx, oos_total_return, oos_total_return_random
    """
    fixed: List[float] = []
    random: List[float] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fixed.append(float(row["oos_total_return"]))
            random.append(float(row["oos_total_return_random"]))
    return np.array(fixed, dtype=float), np.array(random, dtype=float)


def _compute(
    returns: np.ndarray,
    seed: int,
    n_perm: int,
    n_boot: int,
    bootstrap_method: str = "iid",
    block_size: int = 4,
) -> Dict:
    rng_dsr = np.random.default_rng(seed)
    rng_boot = np.random.default_rng(seed + 1)
    rng_perm = np.random.default_rng(seed + 2)

    sr = net_sharpe(returns)
    d = dsr(returns, n_permutations=n_perm, rng=rng_dsr)

    if bootstrap_method == "block":
        b_mean, b_lower, b_upper = block_bootstrap_ci(
            returns, n_bootstrap=n_boot, block_size=block_size, rng=rng_boot
        )
    else:
        b_mean, b_lower, b_upper = bootstrap_ci(returns, n_bootstrap=n_boot, rng=rng_boot)

    p = permutation_p(returns, n_permutations=n_perm, rng=rng_perm)

    return {
        "net_sharpe": round(sr, 6),
        "dsr": round(d, 6),
        "bootstrap_method": bootstrap_method,
        "bootstrap_block_size": block_size if bootstrap_method == "block" else None,
        "bootstrap_ci_mean": round(b_mean, 6),
        "bootstrap_ci_lower": round(b_lower, 6),
        "bootstrap_ci_upper": round(b_upper, 6),
        "permutation_p": round(p, 6),
        "n_folds": len(returns),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute statistical evidence metrics for live gate (B2-B5)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", metavar="FILE", help="walk-forward log file (run_wf.py output)")
    src.add_argument("--returns-csv", metavar="FILE", help="per-fold CSV with fold_idx, oos_total_return, oos_total_return_random")
    parser.add_argument("--output", metavar="FILE", help="write JSON to this path (also printed to stdout)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--n-permutations", type=int, default=1000, help="permutation count for DSR and p-value (default: 1000)")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="bootstrap resample count (default: 10000)")
    parser.add_argument("--bootstrap-method", choices=["iid", "block"], default="iid",
                        help="bootstrap method: 'iid' (standard) or 'block' (circular, corrects for fold autocorrelation)")
    parser.add_argument("--block-size", type=int, default=4,
                        help="block size for block bootstrap (default: 4 ≈ sqrt(12))")
    args = parser.parse_args(argv)

    try:
        if args.log:
            fixed, random = _returns_from_log(Path(args.log))
        else:
            fixed, random = _returns_from_csv(Path(args.returns_csv))
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    result = {
        "random_start": _compute(
            random, args.seed, args.n_permutations, args.n_bootstrap,
            bootstrap_method=args.bootstrap_method, block_size=args.block_size,
        ),
        "fixed_start": _compute(
            fixed, args.seed, args.n_permutations, args.n_bootstrap,
            bootstrap_method=args.bootstrap_method, block_size=args.block_size,
        ),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload)
        print(f"\nWritten to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
