"""
Remove state backup folders older than --keep-days days.

Folder naming convention: <backup-root>/YYYY-MM-DD/
Any subfolder whose name does not parse as a date is left untouched.

Usage:
    python scripts/cleanup_state_backups.py [--backup-root DIR] [--keep-days N] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def cleanup(backup_root: Path, keep_days: int, dry_run: bool) -> list[Path]:
    """Delete date-stamped subfolders older than keep_days. Returns removed paths."""
    if not backup_root.exists():
        return []

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=keep_days)
    removed: list[Path] = []

    for child in sorted(backup_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            folder_date = datetime.strptime(child.name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue  # not a date folder — skip

        if folder_date < cutoff:
            if dry_run:
                log.info("[dry-run] would remove %s", child)
            else:
                shutil.rmtree(child)
                log.info("removed old backup folder: %s", child)
            removed.append(child)

    return removed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean up old state backup folders")
    p.add_argument("--backup-root", default=None, help="root containing YYYY-MM-DD backup folders")
    p.add_argument("--keep-days", type=int, default=7, help="keep backups younger than N days")
    p.add_argument("--dry-run", action="store_true", help="list what would be removed without deleting")
    return p.parse_args(argv)


def _resolve_backup_root(cli_override: str | None) -> Path:
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("TRADINGBOT_BACKUP_ROOT")
    if env:
        return Path(env)
    return Path("state/backups")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    backup_root = _resolve_backup_root(args.backup_root)
    removed = cleanup(backup_root, keep_days=args.keep_days, dry_run=args.dry_run)
    print(f"{'[dry-run] ' if args.dry_run else ''}removed {len(removed)} folder(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
