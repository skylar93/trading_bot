"""Aggregate walk-forward results from multiple run_wf.py log files.

Parses the ``=== RESULT ===`` block (WalkForwardResult dataclass repr) that
run_wf.py writes at the end of each log, then outputs a Phase 8-Beta §2.5
selection-criterion comparison table.

Usage examples
--------------
# Compare four ablation logs and apply the winner criterion:
python scripts/aggregate_wf_results.py \\
    --logs logs/phase8_beta/B0_baseline.log \\
           logs/phase8_beta/B1_inactivity.log \\
           logs/phase8_beta/B2_sharpe_weight.log \\
           logs/phase8_beta/B3_sharpe_clip.log \\
    --variant-names B0 B1 B2 B3 \\
    --apply-criterion

# All logs in a directory, auto-name from filenames:
python scripts/aggregate_wf_results.py \\
    --log-dir logs/phase8_beta --apply-criterion
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedFold:
    fold_idx: int
    is_sharpe: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_total_return: float
    oos_sharpe_random: float
    oos_total_return_random: float
    oos_trade_count_mean: float
    oos_trade_count_random_mean: float
    # Phase 8-Gamma G1 diagnostic — NaN when absent from older logs
    oos_mean_gate_fires_per_episode: float = float("nan")
    oos_mean_gate_active_fraction: float = float("nan")


@dataclass
class VariantResult:
    name: str
    folds: List[ParsedFold] = field(default_factory=list)

    # Aggregate properties (computed lazily)
    def _fold_values(self, attr: str, indices: Optional[List[int]] = None) -> List[float]:
        src = self.folds if indices is None else [f for f in self.folds if f.fold_idx in indices]
        vals = [getattr(f, attr) for f in src]
        return [v for v in vals if not math.isnan(v)]

    def all_mean_random(self) -> float:
        vals = self._fold_values("oos_total_return_random")
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def bull_mean_random(self, bull_indices: List[int]) -> float:
        vals = self._fold_values("oos_total_return_random", bull_indices)
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def bear_mean_random(self, bear_indices: List[int]) -> float:
        vals = self._fold_values("oos_total_return_random", bear_indices)
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def folds_positive(self) -> int:
        return sum(1 for f in self.folds if f.oos_total_return_random > 0)

    def trades_per_ep(self) -> float:
        vals = self._fold_values("oos_trade_count_random_mean")
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def gate_fires_per_ep_mean(self) -> float:
        vals = self._fold_values("oos_mean_gate_fires_per_episode")
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def gate_active_fraction_mean(self) -> float:
        vals = self._fold_values("oos_mean_gate_active_fraction")
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def n_folds(self) -> int:
        return len(self.folds)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Float token: handles nan, inf, -inf, scientific notation, plain decimals
_FLOAT = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))"

# Regex to capture each FoldResult's named numeric fields from a dataclass repr.
# oos_returns has repr=False so it's absent; metrics={...} is last and skipped.
_FOLD_RE = re.compile(
    r"FoldResult\("
    r"fold_idx=(?P<fold_idx>\d+),\s*"
    r"train_size=\d+,\s*"
    r"test_size=\d+,\s*"
    r"is_sharpe=(?P<is_sharpe>" + _FLOAT + r"),\s*"
    r"oos_sharpe=(?P<oos_sharpe>" + _FLOAT + r"),\s*"
    r"oos_max_drawdown=(?P<oos_max_drawdown>" + _FLOAT + r"),\s*"
    r"oos_total_return=(?P<oos_total_return>" + _FLOAT + r"),\s*"
    r"oos_sharpe_random=(?P<oos_sharpe_random>" + _FLOAT + r"),\s*"
    r"oos_total_return_random=(?P<oos_total_return_random>" + _FLOAT + r"),\s*"
    r"oos_trade_count_mean=(?P<oos_trade_count_mean>" + _FLOAT + r"),\s*"
    r"oos_trade_count_random_mean=(?P<oos_trade_count_random_mean>" + _FLOAT + r"),\s*"
    # Optional Phase 8-Gamma G1 fields — present in newer logs, absent in older
    r"(?:oos_mean_gate_fires_per_episode=(?P<gate_fires>" + _FLOAT + r"),\s*"
    r"oos_mean_gate_active_fraction=(?P<gate_frac>" + _FLOAT + r"),\s*)?"
    r"metrics=\{[^}]*\}"
    r"\)"
)


def _parse_float(s: str) -> float:
    s = s.strip()
    if s == "nan":
        return float("nan")
    if s in ("inf", "+inf"):
        return float("inf")
    if s == "-inf":
        return float("-inf")
    return float(s)


def _read_log_text(path: Path) -> str:
    """Read a log file with BOM auto-detection.

    PowerShell on Windows redirects (`*>` / `Out-File`) default to UTF-16 LE
    with BOM (\\xff\\xfe), which UTF-8 decoding garbles. We sniff the first
    bytes and pick the right codec; fall back to UTF-8 with replacement.
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le", errors="replace").lstrip("﻿")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be", errors="replace").lstrip("﻿")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path) -> List[ParsedFold]:
    """Extract all FoldResults from a run_wf.py log file.

    The function scans for ``=== RESULT ===`` and then matches every
    FoldResult(...) in the text that follows.

    Raises ValueError if the RESULT block is absent or no folds are found.
    """
    text = _read_log_text(path)

    marker = "=== RESULT ==="
    idx = text.rfind(marker)  # take the last occurrence if log was re-run
    if idx == -1:
        raise ValueError(f"{path}: '=== RESULT ===' block not found — log may be incomplete")

    result_text = text[idx:]
    folds: List[ParsedFold] = []

    for m in _FOLD_RE.finditer(result_text):
        g = m.groupdict()
        gate_fires = _parse_float(g["gate_fires"]) if g.get("gate_fires") is not None else float("nan")
        gate_frac = _parse_float(g["gate_frac"]) if g.get("gate_frac") is not None else float("nan")
        folds.append(ParsedFold(
            fold_idx=int(g["fold_idx"]),
            is_sharpe=_parse_float(g["is_sharpe"]),
            oos_sharpe=_parse_float(g["oos_sharpe"]),
            oos_max_drawdown=_parse_float(g["oos_max_drawdown"]),
            oos_total_return=_parse_float(g["oos_total_return"]),
            oos_sharpe_random=_parse_float(g["oos_sharpe_random"]),
            oos_total_return_random=_parse_float(g["oos_total_return_random"]),
            oos_trade_count_mean=_parse_float(g["oos_trade_count_mean"]),
            oos_trade_count_random_mean=_parse_float(g["oos_trade_count_random_mean"]),
            oos_mean_gate_fires_per_episode=gate_fires,
            oos_mean_gate_active_fraction=gate_frac,
        ))

    if not folds:
        raise ValueError(
            f"{path}: Parsed 0 FoldResults after '=== RESULT ===' — "
            "repr format may have changed or run was aborted mid-fold"
        )

    return sorted(folds, key=lambda f: f.fold_idx)


# ---------------------------------------------------------------------------
# Selection criterion (Phase 8-Beta §2.5)
# ---------------------------------------------------------------------------

CRITERION_1 = "Criterion 1 (trades>5 AND bull>=B0 bull)"
CRITERION_2 = "Criterion 2 (trades>3 AND bear>B0 bear)"
CRITERION_NONE = "STOP — no variant passes criterion 2"


def apply_criterion(
    variants: List[VariantResult],
    bull_indices: List[int],
    bear_indices: List[int],
    b0_name: str = "B0",
) -> Tuple[Optional[str], str]:
    """Return (winning_variant_name, criterion_label) per §2.5.

    Tie-break: prefer simpler variant (B2 over B3, then lower index).
    Returns (None, CRITERION_NONE) if no variant passes.
    """
    b0 = next((v for v in variants if v.name == b0_name), None)
    b0_bull = b0.bull_mean_random(bull_indices) if b0 else float("nan")
    b0_bear = b0.bear_mean_random(bear_indices) if b0 else float("nan")

    # Exclude B0 from selection (it's the baseline reference)
    candidates = [v for v in variants if v.name != b0_name]

    # Criterion 1: trades/ep > 5 AND bull mean >= B0 bull mean
    c1_hits = [
        v for v in candidates
        if v.trades_per_ep() > 5.0 and
        (math.isnan(b0_bull) or v.bull_mean_random(bull_indices) >= b0_bull)
    ]
    if c1_hits:
        # Tie-break: prefer B2 > B3 > others, then first in list order
        def _c1_rank(v: VariantResult) -> int:
            order = {"B2": 0, "B3": 1}
            return order.get(v.name, 2 + candidates.index(v))
        winner = min(c1_hits, key=_c1_rank)
        return winner.name, CRITERION_1

    # Criterion 2: trades/ep > 3 AND bear mean strictly better than B0
    c2_hits = [
        v for v in candidates
        if v.trades_per_ep() > 3.0 and
        (math.isnan(b0_bear) or v.bear_mean_random(bear_indices) > b0_bear)
    ]
    if c2_hits:
        winner = max(c2_hits, key=lambda v: v.bear_mean_random(bear_indices))
        return winner.name, CRITERION_2

    return None, CRITERION_NONE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "  n/a "
    return f"{v * 100:+.{decimals}f}%"


def _f2(v: float) -> str:
    if math.isnan(v):
        return " n/a"
    return f"{v:.2f}"


def _gate_fmt(v: float) -> str:
    if math.isnan(v):
        return "n/a"
    return f"{v:.2f}"


def print_table(
    variants: List[VariantResult],
    bull_indices: List[int],
    bear_indices: List[int],
) -> None:
    header = (
        f"{'Variant':<12} {'All mean(rnd)':>14} {'Bull mean':>10} "
        f"{'Bear mean':>10} {'Folds+':>7} {'Trades/ep':>10} "
        f"{'GateFires/ep':>13} {'GateActiveFrac':>15}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for v in variants:
        print(
            f"{v.name:<12} "
            f"{_pct(v.all_mean_random()):>14} "
            f"{_pct(v.bull_mean_random(bull_indices)):>10} "
            f"{_pct(v.bear_mean_random(bear_indices)):>10} "
            f"{v.folds_positive():>5}/{v.n_folds():<2} "
            f"{_f2(v.trades_per_ep()):>10} "
            f"{_gate_fmt(v.gate_fires_per_ep_mean()):>13} "
            f"{_gate_fmt(v.gate_active_fraction_mean()):>15}"
        )
    print(sep)
    # Warn if a G1/gamma variant has dormant gate
    for v in variants:
        name_lower = v.name.lower()
        if "g1" in name_lower or "gamma" in name_lower or "regime" in name_lower:
            gf = v.gate_fires_per_ep_mean()
            if math.isnan(gf):
                print(
                    f"WARNING: variant '{v.name}' has no gate metrics. "
                    "Run with logger.INFO + diagnostic-polish PR.",
                    file=sys.stderr,
                )
            elif gf < 1.0:
                print(
                    f"WARNING: variant '{v.name}' shows mean gate_fires/ep = {gf:.2f} < 1.",
                    file=sys.stderr,
                )
                print(
                    "  → Detector dormant or evaluator not capturing info. "
                    "Check run_wf log for 'argmax==BEAR' counts.",
                    file=sys.stderr,
                )


def print_fold_detail(variant: VariantResult) -> None:
    print(f"\nPer-fold detail — {variant.name}")
    hdr = f"{'Fold':>5} {'OOS ret(rnd)':>13} {'OOS Sharpe(rnd)':>16} {'Trades/ep(rnd)':>15}"
    print(hdr)
    print("-" * len(hdr))
    for f in variant.folds:
        print(
            f"{f.fold_idx:>5} "
            f"{_pct(f.oos_total_return_random):>13} "
            f"{_f2(f.oos_sharpe_random):>16} "
            f"{_f2(f.oos_trade_count_random_mean):>15}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_fold_range(s: str) -> List[int]:
    """Parse '0-4' or '0,1,2,3,4' into a sorted list of fold indices."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 8-Beta walk-forward log files into a comparison table."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--logs", nargs="+", metavar="FILE",
        help="One or more run_wf.py log files to compare.",
    )
    src.add_argument(
        "--log-dir", metavar="DIR",
        help="Directory containing *.log files — all will be parsed.",
    )
    parser.add_argument(
        "--variant-names", nargs="+", metavar="NAME",
        help="Friendly names for each log (same order as --logs). "
             "Defaults to the log filename stem.",
    )
    parser.add_argument(
        "--bull-folds", default="0,2,5,8,11", metavar="RANGE",
        help="0-indexed fold indices classified as bull regime "
             "(default: '0,2,5,8,11'). Accepts 'lo-hi' or comma list.",
    )
    parser.add_argument(
        "--bear-folds", default="1,3,4,6,7,9,10", metavar="RANGE",
        help="0-indexed fold indices classified as bear regime "
             "(default: '1,3,4,6,7,9,10').",
    )
    parser.add_argument(
        "--apply-criterion", action="store_true",
        help="Automatically apply the §2.5 selection criterion and print winner.",
    )
    parser.add_argument(
        "--b0-name", default="B0",
        help="Name of the baseline variant used for criterion comparison (default: B0).",
    )
    parser.add_argument(
        "--detail", metavar="VARIANT",
        help="Print per-fold detail for this variant name (e.g. --detail B2).",
    )
    args = parser.parse_args(argv)

    # Collect log paths
    if args.logs:
        log_paths = [Path(p) for p in args.logs]
    else:
        log_paths = sorted(Path(args.log_dir).glob("*.log"))
        if not log_paths:
            print(f"No *.log files found in {args.log_dir}", file=sys.stderr)
            return 1

    # Determine variant names
    names: List[str]
    if args.variant_names:
        if len(args.variant_names) != len(log_paths):
            print(
                f"--variant-names has {len(args.variant_names)} entries "
                f"but {len(log_paths)} log files were found",
                file=sys.stderr,
            )
            return 1
        names = args.variant_names
    else:
        names = [p.stem for p in log_paths]

    bull_indices = _parse_fold_range(args.bull_folds)
    bear_indices = _parse_fold_range(args.bear_folds)

    # Parse logs
    variants: List[VariantResult] = []
    errors = 0
    for path, name in zip(log_paths, names):
        try:
            folds = parse_log(path)
            variants.append(VariantResult(name=name, folds=folds))
            print(f"  Loaded {name}: {len(folds)} folds from {path}")
        except (ValueError, OSError) as exc:
            print(f"  ERROR loading {name} ({path}): {exc}", file=sys.stderr)
            errors += 1

    if not variants:
        print("No variants successfully loaded — nothing to compare.", file=sys.stderr)
        return 1

    print()
    print(f"Bull folds (0-indexed): {bull_indices}")
    print(f"Bear folds (0-indexed): {bear_indices}")
    print()

    print_table(variants, bull_indices, bear_indices)

    if args.detail:
        target = next((v for v in variants if v.name == args.detail), None)
        if target is None:
            print(f"--detail: variant '{args.detail}' not found in loaded variants", file=sys.stderr)
        else:
            print_fold_detail(target)

    if args.apply_criterion:
        print()
        winner, criterion = apply_criterion(variants, bull_indices, bear_indices, args.b0_name)
        print(f"Selection criterion: {criterion}")
        if winner:
            print(f"  → Winner: {winner}  (selected for Stage 2)")
        else:
            print("  → No winner. Consider Stage 1.5 (combined B4/B5 variants).")

    return errors  # 0 = clean, >0 = some logs failed


if __name__ == "__main__":
    sys.exit(main())
