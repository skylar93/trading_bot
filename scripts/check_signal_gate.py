#!/usr/bin/env python
"""[A0.5] Operator wrapper — live-readiness signal gate check.

Standalone script: runs LiveSignalGate and exits with:
    0 — all checks passed (live mode authorized)
    2 — gate failed (thresholds not met / evidence stale)
    1 — error (missing file, bad config, import failure)

Usage:
    python scripts/check_signal_gate.py
    python scripts/check_signal_gate.py --config config/deployment.yaml
    python scripts/check_signal_gate.py --evidence-pack /path/to/evidence.md
    python scripts/check_signal_gate.py --verbose

This script is referenced by go_live_checklist.md Track Z (Z1).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.governance.live_signal_gate import _cli_main

if __name__ == "__main__":
    sys.exit(_cli_main())
