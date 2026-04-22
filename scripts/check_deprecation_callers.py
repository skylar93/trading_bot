#!/usr/bin/env python3
"""CI gate: verify no source file calls deprecated risk API methods.

Forbidden symbols (public deprecated wrappers):
  - check_stop_loss     → use check_trailing_stop()
  - check_max_drawdown  → use check_drawdown()
  - calculate_var       → use compute_var()

Files that are EXCLUDED (define or test the deprecated shims):
  - risk_management/risk_manager_base.py    (defines the shim)
  - risk_management/backtesting_risk_manager.py (defines the shim)
  - risk_management/rl_risk_manager.py      (defines the shim)
  - tests/                                  (may test the deprecated API directly)
  - scripts/check_deprecation_callers.py    (this file)

Exit 0  →  clean
Exit 1  →  callers found (CI fail)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_SYMBOLS = [
    "check_stop_loss",
    "check_max_drawdown",
    "calculate_var",
]

EXCLUDED_PATHS = [
    "risk_management/risk_manager_base.py",
    "risk_management/backtesting_risk_manager.py",
    "risk_management/rl_risk_manager.py",
    "tests/",
    "scripts/",   # scripts reference symbol names as strings for enforcement checks
    ".claude/",
    "__pycache__/",
]


def _is_excluded(path: str) -> bool:
    for excl in EXCLUDED_PATHS:
        if excl in path:
            return True
    return False


def _search(symbol: str) -> list[str]:
    """Return list of non-excluded files containing *symbol* as a whole word."""
    pattern = rf"\b{symbol}\b"
    try:
        result = subprocess.run(
            ["rg", "-l", pattern, str(PROJECT_ROOT), "--glob", "*.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
    except FileNotFoundError:
        result = subprocess.run(
            ["grep", "-rlw", symbol, str(PROJECT_ROOT), "--include=*.py"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()

    return [l for l in lines if l and not _is_excluded(l)]


def main() -> int:
    violations: dict[str, list[str]] = {}

    for symbol in FORBIDDEN_SYMBOLS:
        hits = _search(symbol)
        if hits:
            violations[symbol] = hits

    if not violations:
        print("check_deprecation_callers: OK — no deprecated API callers found")
        return 0

    print("check_deprecation_callers: FAIL — deprecated API callers found:")
    for symbol, files in violations.items():
        print(f"  {symbol}:")
        for f in files:
            print(f"    {f}")
    print()
    print("Replace deprecated callers:")
    print("  check_stop_loss()    → check_trailing_stop()")
    print("  check_max_drawdown() → check_drawdown()")
    print("  calculate_var()      → compute_var()")
    return 1


if __name__ == "__main__":
    sys.exit(main())
