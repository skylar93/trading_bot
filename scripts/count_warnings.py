"""
E17: Count filtered pytest warnings and assert < 500 target.

Usage:
    python scripts/count_warnings.py            # assert count < 500
    python scripts/count_warnings.py --no-fail  # report only, no assertion
    python scripts/count_warnings.py --target 200
"""

import subprocess
import sys
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = 500


def run_pytest() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout + result.stderr


def extract_counts(output: str) -> dict:
    """Pull passed/skipped/warning/failed counts from pytest summary line."""
    # e.g. "1807 passed, 40 skipped, 37 warnings in 77.61s"
    summary_re = re.compile(
        r"(\d+)\s+passed"
        r"(?:,\s*(\d+)\s+skipped)?"
        r"(?:,\s*(\d+)\s+(?:warning|warnings))?"
        r"(?:,\s*(\d+)\s+failed)?"
    )
    m = summary_re.search(output)
    if not m:
        return {}
    return {
        "passed": int(m.group(1) or 0),
        "skipped": int(m.group(2) or 0),
        "warnings": int(m.group(3) or 0),
        "failed": int(m.group(4) or 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Count filtered pytest warnings")
    parser.add_argument("--no-fail", action="store_true", help="Report only, don't exit non-zero")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Max allowed warnings (default: {DEFAULT_TARGET})")
    args = parser.parse_args()

    print("Running pytest (this may take a moment)...", file=sys.stderr)
    output = run_pytest()
    counts = extract_counts(output)

    if not counts:
        print("ERROR: Could not parse pytest output.", file=sys.stderr)
        print(output[-2000:])
        sys.exit(2)

    warnings = counts.get("warnings", 0)
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)

    print(f"\n{'='*50}")
    print(f"  pytest results")
    print(f"{'='*50}")
    print(f"  passed   : {passed}")
    print(f"  skipped  : {skipped}")
    print(f"  failed   : {failed}")
    print(f"  warnings : {warnings}  (target < {args.target})")
    print(f"{'='*50}")

    ok = warnings < args.target
    if ok:
        print(f"  PASS: {warnings} < {args.target}")
    else:
        print(f"  FAIL: {warnings} >= {args.target}  (reduce warnings to meet target)")

    if not args.no_fail and not ok:
        sys.exit(1)

    return warnings


if __name__ == "__main__":
    main()
