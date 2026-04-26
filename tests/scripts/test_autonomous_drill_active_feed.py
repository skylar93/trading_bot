"""I9-b: active_feed field in drill snapshots."""
from __future__ import annotations

import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_snapshot_contains_active_feed(tmp_path):
    """_write_snapshot includes 'active_feed' key."""
    from scripts.autonomous_72h_drill import AutonomousDrill

    config = {
        "duration_hours": 0.001,
        "log_dir": str(tmp_path / "logs"),
        "docs_dir": str(tmp_path / "docs"),
        "feed": "gbm",
        "tick_interval": 0.0,
    }
    drill = AutonomousDrill(config)
    # Simulate feed selection (normally happens in _run_feed thread)
    drill._select_feed()
    drill._write_snapshot()

    snap_path = tmp_path / "logs" / "drill_snapshots.jsonl"
    assert snap_path.exists()
    snap = json.loads(snap_path.read_text().strip())
    assert "active_feed" in snap
    assert snap["active_feed"] == "gbm"


def test_active_feed_updates_on_fallback(tmp_path):
    """_on_feed_fallback updates _active_feed and logs incident."""
    from scripts.autonomous_72h_drill import AutonomousDrill

    config = {
        "duration_hours": 0.001,
        "log_dir": str(tmp_path / "logs"),
        "docs_dir": str(tmp_path / "docs"),
        "feed": "ws",
        "tick_interval": 0.0,
    }
    drill = AutonomousDrill(config)
    drill._active_feed = "ws"

    drill._on_feed_fallback("ws_test_fail")

    assert drill._active_feed in ("csv", "gbm")
    assert any(
        i["type"] == "feed_fallback" for i in drill._stats.incidents
    )
