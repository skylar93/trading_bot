"""I5: setup_testnet.py --dry-run tests."""
from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_dry_run_exits_zero():
    """--dry-run completes without error (exit code 0)."""
    from scripts.setup_testnet import main
    rc = main(["--dry-run"])
    assert rc == 0


def test_dry_run_updates_checklist(tmp_path, monkeypatch):
    """--dry-run skips external calls but still patches the checklist."""
    import re
    from scripts import setup_testnet as wiz

    # Create a minimal go_live_checklist.md in tmp_path
    checklist = tmp_path / "go_live_checklist.md"
    checklist.write_text(
        "| F1 | something | manual |\n"
        "| F2 | something | manual |\n"
        "| S3 | something | manual |\n"
        "| O7 | something | manual |\n"
    )
    monkeypatch.setattr(wiz, "_CHECKLIST", checklist)

    rc = wiz.run_wizard(dry_run=True)
    assert rc == 0

    content = checklist.read_text()
    assert "[auto-wizard]" in content


def test_keychain_set_subprocess_args(monkeypatch):
    """I5-b: KeychainSecretProvider.set calls 'security add-generic-password'."""
    import subprocess
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    from deployment.secrets.secret_provider import KeychainSecretProvider
    kp = KeychainSecretProvider.__new__(KeychainSecretProvider)

    kp.set("MY_KEY", "MY_VALUE")

    assert any("add-generic-password" in str(c) for c in calls)
    assert any("MY_KEY" in str(c) for c in calls)
