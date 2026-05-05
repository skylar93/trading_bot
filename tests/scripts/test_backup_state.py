"""Unit tests for scripts/backup_state.py and scripts/cleanup_state_backups.py."""

import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Make scripts/ importable without installing
SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import backup_state as bs
import cleanup_state_backups as cs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(path: Path, wal: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(v TEXT)")
    conn.execute("INSERT INTO t VALUES('hello')")
    conn.commit()
    conn.close()


def _read_value(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    val = conn.execute("SELECT v FROM t").fetchone()[0]
    conn.close()
    return val


# ---------------------------------------------------------------------------
# backup_state tests
# ---------------------------------------------------------------------------

class TestBackupDb:
    def test_copies_wal_db(self, tmp_path):
        src = tmp_path / "state" / "paper_trader.db"
        _make_db(src)
        dest_dir = tmp_path / "dest"

        ok = bs._backup_db(src, dest_dir)

        assert ok
        dest = dest_dir / "paper_trader.db"
        assert dest.exists()
        assert _read_value(dest) == "hello"

    def test_new_path_creates_empty_backup(self, tmp_path):
        # sqlite3.connect on a non-existent path creates an empty DB — backup should succeed
        ok = bs._backup_db(tmp_path / "nonexistent.db", tmp_path / "dest")
        assert ok
        assert (tmp_path / "dest" / "nonexistent.db").exists()

    def test_corrupt_src_returns_false(self, tmp_path):
        src = tmp_path / "bad.db"
        src.write_bytes(b"not a valid sqlite database at all!!!")
        dest_dir = tmp_path / "dest"
        ok = bs._backup_db(src, dest_dir)
        assert not ok
        assert not list(dest_dir.glob("*.db"))


class TestBackup:
    def test_backs_up_all_dbs(self, tmp_path):
        state_dir = tmp_path / "state"
        for name in ("paper_trader.db", "other.db"):
            _make_db(state_dir / name)
        backup_root = tmp_path / "backups"

        backed = bs.backup(state_dir, backup_root, dry_run=False)

        assert len(backed) == 2
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        dest_dir = backup_root / today
        assert (dest_dir / "paper_trader.db").exists()
        assert (dest_dir / "other.db").exists()

    def test_dry_run_writes_nothing(self, tmp_path):
        state_dir = tmp_path / "state"
        _make_db(state_dir / "paper_trader.db")
        backup_root = tmp_path / "backups"

        backed = bs.backup(state_dir, backup_root, dry_run=True)

        assert backed  # returned the list
        assert not backup_root.exists()  # no actual write

    def test_empty_state_dir_returns_empty(self, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        backed = bs.backup(state_dir, tmp_path / "backups", dry_run=False)
        assert backed == []

    def test_main_exit_zero(self, tmp_path):
        state_dir = tmp_path / "state"
        _make_db(state_dir / "paper_trader.db")
        backup_root = tmp_path / "backups"

        rc = bs.main([
            "--state-dir", str(state_dir),
            "--backup-root", str(backup_root),
            "--keep-days", "0",
        ])
        assert rc == 0

    def test_main_missing_state_dir_returns_2(self, tmp_path):
        rc = bs.main([
            "--state-dir", str(tmp_path / "ghost"),
            "--backup-root", str(tmp_path / "backups"),
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# cleanup_state_backups tests
# ---------------------------------------------------------------------------

class TestCleanup:
    def _make_folder(self, root: Path, days_ago: int) -> Path:
        date_str = (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        folder = root / date_str
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "dummy.db").write_text("x")
        return folder

    def test_removes_old_folders(self, tmp_path):
        root = tmp_path / "backups"
        old = self._make_folder(root, days_ago=8)
        recent = self._make_folder(root, days_ago=2)

        removed = cs.cleanup(root, keep_days=7, dry_run=False)

        assert old in removed
        assert not old.exists()
        assert recent.exists()

    def test_dry_run_does_not_delete(self, tmp_path):
        root = tmp_path / "backups"
        old = self._make_folder(root, days_ago=10)

        removed = cs.cleanup(root, keep_days=7, dry_run=True)

        assert old in removed
        assert old.exists()  # not actually deleted

    def test_skips_non_date_folders(self, tmp_path):
        root = tmp_path / "backups"
        misc = root / "misc_folder"
        misc.mkdir(parents=True)

        removed = cs.cleanup(root, keep_days=0, dry_run=False)

        assert misc not in removed
        assert misc.exists()

    def test_nonexistent_root_returns_empty(self, tmp_path):
        removed = cs.cleanup(tmp_path / "ghost", keep_days=7, dry_run=False)
        assert removed == []

    def test_keeps_within_window(self, tmp_path):
        root = tmp_path / "backups"
        self._make_folder(root, days_ago=3)
        self._make_folder(root, days_ago=6)

        removed = cs.cleanup(root, keep_days=7, dry_run=False)

        assert removed == []
        assert len(list(root.iterdir())) == 2
