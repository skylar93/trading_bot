#!/usr/bin/env python
"""I12-c: Verify bash code blocks in week85 runbook docs with `bash -n`.

Usage:
    python scripts/verify_doc_commands.py          # checks both week85 docs
    python scripts/verify_doc_commands.py path/to/doc.md ...  # custom files

Exit code: 0 if all blocks pass, 1 if any fail.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DOCS = [
    "docs/phase7/week85_72h.md",
    "docs/phase7/week85_first_dollar.md",
]

_BASH_BLOCK = re.compile(r"```bash\n(.*?)\n```", re.DOTALL)


def verify_file(path: Path) -> list[str]:
    """Return list of failure messages (empty = all passed)."""
    if not path.exists():
        return [f"{path}: file not found"]

    text = path.read_text()
    blocks = _BASH_BLOCK.findall(text)
    failures = []
    for i, block in enumerate(blocks):
        result = subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            failures.append(f"{path} block {i}: {stderr or 'syntax error'}")
    return failures


def main(paths: list[Path]) -> int:
    all_failures: list[str] = []
    for p in paths:
        failures = verify_file(p)
        for f in failures:
            print(f"❌ {f}")
        all_failures.extend(failures)

    if not all_failures:
        print(f"✅ all bash blocks passed ({len(paths)} file(s) checked)")
        return 0

    print(f"\n{len(all_failures)} block(s) failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        doc_paths = [Path(a) for a in sys.argv[1:]]
    else:
        doc_paths = [PROJECT_ROOT / d for d in DEFAULT_DOCS]

    sys.exit(main(doc_paths))
