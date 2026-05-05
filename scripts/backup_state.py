"""
Backup state/*.db files using SQLite's online backup API (WAL-safe).

Usage:
    python scripts/backup_state.py [--state-dir STATE_DIR] [--backup-root BACKUP_ROOT]
                                   [--keep-days N] [--dry-run]

Destinations (in priority order):
    1. --backup-root  CLI flag
    2. TRADINGBOT_BACKUP_ROOT  env var
    3. ~/Library/Mobile Documents/com~apple~CloudDocs/tradingbot_backups  (iCloud, if mounted)
    4. state/backups  (local fallback)

Calls cleanup_state_backups.py logic after a successful backup.
Exit code: 0 = success, 1 = partial failure (some DBs failed), 2 = fatal.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("backup_state")

_ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "tradingbot_backups"


def _resolve_backup_root(cli_override: str | None) -> Path:
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("TRADINGBOT_BACKUP_ROOT")
    if env:
        return Path(env)
    if _ICLOUD.parent.exists():
        return _ICLOUD
    return Path("state/backups")


def _backup_db(src: Path, dest_dir: Path) -> bool:
    """Copy src into dest_dir/<src.name> via sqlite3 online backup API."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        src_conn = sqlite3.connect(str(src), check_same_thread=False)
        dst_conn = sqlite3.connect(str(dest))
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        size_kb = dest.stat().st_size // 1024
        log.info("backed up %s → %s (%d KB)", src, dest, size_kb)
        return True
    except Exception as exc:
        log.error("failed to back up %s: %s", src, exc)
        dest.unlink(missing_ok=True)
        return False


def backup(state_dir: Path, backup_root: Path, dry_run: bool) -> list[Path]:
    """Back up all *.db files in state_dir. Returns list of successfully backed-up sources."""
    dbs = sorted(state_dir.glob("*.db"))
    if not dbs:
        log.warning("no *.db files found in %s — nothing to back up", state_dir)
        return []

    tag = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    dest_dir = backup_root / tag

    if dry_run:
        log.info("[dry-run] would back up %d DB(s) → %s", len(dbs), dest_dir)
        return dbs

    log.info("backing up %d DB(s) to %s", len(dbs), dest_dir)
    ok = []
    for db in dbs:
        if _backup_db(db, dest_dir):
            ok.append(db)
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WAL-safe SQLite state backup")
    p.add_argument("--state-dir", default="state", help="directory containing *.db files")
    p.add_argument("--backup-root", default=None, help="root for backup folders (overrides env + iCloud)")
    p.add_argument("--keep-days", type=int, default=7, help="days of backups to keep (0 = no cleanup)")
    p.add_argument("--dry-run", action="store_true", help="show what would happen without writing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_dir = Path(args.state_dir)
    backup_root = _resolve_backup_root(args.backup_root)

    if not state_dir.exists():
        log.error("state-dir %s does not exist", state_dir)
        return 2

    log.info("backup_root resolved to: %s", backup_root)

    backed_up = backup(state_dir, backup_root, args.dry_run)

    # In-process cleanup so this script is self-contained
    if args.keep_days > 0 and not args.dry_run:
        from cleanup_state_backups import cleanup  # local import; same scripts/ dir
        removed = cleanup(backup_root, keep_days=args.keep_days, dry_run=False)
        if removed:
            log.info("cleaned up %d old backup folder(s)", len(removed))

    dbs_in_dir = list(state_dir.glob("*.db"))
    if dbs_in_dir and not backed_up:
        return 1  # found DBs but none backed up
    return 0


if __name__ == "__main__":
    sys.exit(main())
