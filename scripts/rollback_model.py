#!/usr/bin/env python
"""
Week 68 (S62): Model rollback command.

Usage:
    python scripts/rollback_model.py <version> [options]

Examples:
    # List all registered versions
    python scripts/rollback_model.py --list

    # Roll back to version 3 (updates active pointer only)
    python scripts/rollback_model.py 3

    # Roll back to version 3 and copy checkpoint to a specific path
    python scripts/rollback_model.py 3 --active-model-path models/active/model.zip

    # Use a custom registry directory
    python scripts/rollback_model.py 3 --registry-dir /path/to/registry

Prerequisite: PaperTrader must be restarted after rollback to pick up the new model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure repo root is on sys.path when invoked directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from training.registry.model_registry import ModelRegistry, _DEFAULT_REGISTRY_DIR


def _list_versions(registry: ModelRegistry) -> None:
    versions = registry.list_versions()
    if not versions:
        print("No versions registered.")
        return
    print(f"{'Ver':>4}  {'Registered At':<27}  {'Active':>6}  Tag")
    print("-" * 65)
    for v in versions:
        active_marker = "  <--" if v["is_active"] else ""
        print(
            f"{v['version']:>4}  {v['registered_at']:<27}  "
            f"{str(v['is_active']):>6}  {v['tag']}{active_marker}"
        )


def _show_version(registry: ModelRegistry, version: int) -> None:
    try:
        meta = registry.get_version(version)
    except (KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roll back the active model to a previously registered version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "version",
        nargs="?",
        type=str,
        help="Target version number to roll back to (e.g. 3 or v3).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered versions and exit.",
    )
    parser.add_argument(
        "--show",
        type=int,
        metavar="VERSION",
        help="Print metadata for a specific version and exit.",
    )
    parser.add_argument(
        "--active-model-path",
        metavar="PATH",
        default=None,
        help=(
            "If given, copy the rolled-back checkpoint to this path so a "
            "restarted PaperTrader picks it up automatically."
        ),
    )
    parser.add_argument(
        "--registry-dir",
        metavar="DIR",
        default=_DEFAULT_REGISTRY_DIR,
        help=f"Registry root directory (default: {_DEFAULT_REGISTRY_DIR}).",
    )

    args = parser.parse_args(argv)
    registry = ModelRegistry(registry_dir=args.registry_dir)

    if args.list:
        _list_versions(registry)
        return 0

    if args.show is not None:
        _show_version(registry, args.show)
        return 0

    if args.version is None:
        parser.error("Provide a version number or use --list / --show.")

    # Accept both "3" and "v3"
    raw_version = str(args.version).lstrip("v")
    try:
        target_version = int(raw_version)
    except ValueError:
        print(f"ERROR: invalid version: {args.version!r}", file=sys.stderr)
        return 1

    try:
        stored_path = registry.rollback(
            target_version=target_version,
            active_model_path=args.active_model_path,
        )
    except (KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Rolled back to version {target_version}.")
    print(f"Checkpoint path: {stored_path}")
    if args.active_model_path:
        print(f"Copied to: {args.active_model_path}")
    print("\nRestart PaperTrader to apply the rollback.")
    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
