"""
E13: pytest warning report parser and classifier.

Usage:
    python scripts/analyze_warnings.py
    python scripts/analyze_warnings.py --json
    python scripts/analyze_warnings.py --top 20

Runs pytest with warning capture enabled and classifies each warning by:
  - category (RuntimeWarning, DeprecationWarning, etc.)
  - source (our code vs third-party library)
  - root cause bucket (divide/invalid-value, empty-slice, dof-zero, etc.)
  - verdict (normal-edge-case | bug)
"""

import subprocess
import sys
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Classify warnings by file path
OWN_CODE_PREFIX = str(ROOT)
THIRD_PARTY_PREFIXES = (
    "site-packages",
    "/anaconda3/",
    "/lib/python",
)

# Map warning message patterns → root cause bucket
BUCKETS = [
    (r"invalid value encountered in (divide|true_divide|scalar divide)", "divide-invalid"),
    (r"divide by zero encountered", "divide-zero"),
    (r"Mean of empty slice", "empty-slice"),
    (r"Degrees of freedom <= 0", "dof-zero"),
    (r"overflow encountered", "overflow"),
    (r"underflow encountered", "underflow"),
    (r"invalid value encountered in (multiply|subtract|add|log|sqrt|power)", "numeric-op"),
    (r"NaN values", "nan-values"),
]

# Paths in our own code that are *intentional* edge-case guards
INTENTIONAL_PATHS = [
    # add more as we annotate them
]


def _bucket(msg: str) -> str:
    for pattern, label in BUCKETS:
        if re.search(pattern, msg, re.IGNORECASE):
            return label
    return "other"


def _is_own_code(path: str) -> bool:
    if any(tp in path for tp in THIRD_PARTY_PREFIXES):
        return False
    return OWN_CODE_PREFIX in path or path.startswith("envs/") or path.startswith("deployment/")


def _verdict(path: str, bucket: str) -> str:
    if any(p in path for p in INTENTIONAL_PATHS):
        return "intentional"
    if not _is_own_code(path):
        return "third-party"
    # Our code, unguarded path
    if bucket in ("divide-invalid", "divide-zero", "overflow", "nan-values"):
        return "bug"
    if bucket in ("empty-slice", "dof-zero"):
        return "edge-case"
    return "unknown"


def run_pytest_with_warnings() -> str:
    """Run pytest and capture the warnings summary."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-q", "--tb=no",
            "-W", "all",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout + result.stderr


def parse_warnings(output: str) -> list[dict]:
    """Parse pytest warning lines into structured records."""
    # warnings section starts after "warnings summary" header
    in_warnings = False
    records = []

    # Pattern: "  path/to/file.py:lineno: CategoryName: message text"
    warn_re = re.compile(
        r"^\s+(?P<path>[^:]+):(?P<line>\d+):\s+(?P<category>\w+Warning):\s+(?P<msg>.+)$"
    )
    for raw_line in output.splitlines():
        if "warnings summary" in raw_line.lower():
            in_warnings = True
            continue
        if in_warnings and raw_line.startswith("="):
            in_warnings = False
            continue
        if not in_warnings:
            continue

        m = warn_re.match(raw_line)
        if not m:
            continue

        path = m.group("path").strip()
        category = m.group("category")
        msg = m.group("msg").strip()
        bucket = _bucket(msg)
        verdict = _verdict(path, bucket)

        records.append(
            {
                "path": path,
                "line": int(m.group("line")),
                "category": category,
                "message": msg,
                "bucket": bucket,
                "verdict": verdict,
                "own_code": _is_own_code(path),
            }
        )

    return records


def summarize(records: list[dict]) -> dict:
    total = len(records)
    by_bucket: dict[str, int] = defaultdict(int)
    by_verdict: dict[str, int] = defaultdict(int)
    by_file: dict[str, int] = defaultdict(int)
    bugs: list[dict] = []

    for r in records:
        by_bucket[r["bucket"]] += 1
        by_verdict[r["verdict"]] += 1
        by_file[r["path"]] += 1
        if r["verdict"] == "bug":
            bugs.append(r)

    return {
        "total": total,
        "by_bucket": dict(sorted(by_bucket.items(), key=lambda x: -x[1])),
        "by_verdict": dict(sorted(by_verdict.items(), key=lambda x: -x[1])),
        "top_files": dict(sorted(by_file.items(), key=lambda x: -x[1])[:10]),
        "bugs": bugs,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze pytest warnings")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--top", type=int, default=10, help="Top-N files to show")
    args = parser.parse_args()

    print("Running pytest (this may take a moment)...", file=sys.stderr)
    output = run_pytest_with_warnings()

    records = parse_warnings(output)
    summary = summarize(records)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"\n{'='*60}")
    print(f"Total warnings: {summary['total']}")
    print(f"{'='*60}")

    print("\n--- By Bucket ---")
    for bucket, count in summary["by_bucket"].items():
        print(f"  {bucket:<30} {count:>6}")

    print("\n--- By Verdict ---")
    for verdict, count in summary["by_verdict"].items():
        marker = " <-- ACTION NEEDED" if verdict == "bug" else ""
        print(f"  {verdict:<20} {count:>6}{marker}")

    print(f"\n--- Top {args.top} Files ---")
    for path, count in list(summary["top_files"].items())[: args.top]:
        own = "[OWN]" if _is_own_code(path) else "[lib]"
        print(f"  {own} {count:>5}  {path}")

    if summary["bugs"]:
        print(f"\n--- BUG WARNINGS ({len(summary['bugs'])}) --- (require fix)")
        seen = set()
        for b in summary["bugs"]:
            key = f"{b['path']}:{b['line']}"
            if key not in seen:
                seen.add(key)
                print(f"  {b['path']}:{b['line']}")
                print(f"    [{b['bucket']}] {b['message']}")
    else:
        print("\nNo bug-verdict warnings found.")

    print(f"\n{'='*60}")
    own_bugs = [r for r in records if r["verdict"] == "bug"]
    print(f"Action needed: {len(own_bugs)} bug-verdict warnings in own code")


if __name__ == "__main__":
    main()
