"""
promote_model.py — Week 75 G4: Manual model promotion CLI.

Usage:
    # Check if promotion is allowed (dry-run, no changes)
    python scripts/promote_model.py --check --from candidate --to staging --version 3

    # Perform promotion
    python scripts/promote_model.py \
        --from staging --to canary --version 3 \
        --actor skylar --reason "walkforward Sharpe=0.82"

    # Use a custom registry directory
    python scripts/promote_model.py --registry /path/to/registry \
        --from canary --to prod --version 3 --actor skylar --reason "7d canary passed"

All promotion events are written to the registry's audit history.
Transitions that violate the state machine are rejected with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from training.registry.model_registry import (
    ModelRegistry,
    VALID_STAGES,
    VALID_TRANSITIONS,
    PROMOTION_CRITERIA,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manually promote a model version through the registry stage machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--registry",
        default=None,
        help="Path to the registry directory (default: ~/.trading_bot/model_registry)",
    )
    p.add_argument(
        "--version", "-v",
        required=True,
        help="Version number to promote (e.g. 3 or v3)",
    )
    p.add_argument(
        "--from", dest="from_stage",
        required=True,
        choices=list(VALID_STAGES),
        help="Expected current stage (checked against registry; fails if mismatch)",
    )
    p.add_argument(
        "--to", dest="to_stage",
        required=True,
        choices=list(VALID_STAGES),
        help="Target stage",
    )
    p.add_argument(
        "--actor",
        default="unknown",
        help="Name of the person/system performing the promotion (stored in audit)",
    )
    p.add_argument(
        "--reason",
        default="",
        help="Free-text justification (stored in audit history)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: check conditions and print result without making changes",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Skip transition validity check (for testing only — use with care)",
    )
    p.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output result as JSON (useful for scripting)",
    )
    return p


def _print_result(ok: bool, message: str, data: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": ok, "message": message, **data}, indent=2))
    else:
        prefix = "OK" if ok else "REJECTED"
        print(f"[{prefix}] {message}")
        if data:
            for k, v in data.items():
                print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Open registry
    if args.registry:
        registry = ModelRegistry(registry_dir=args.registry)
    else:
        registry = ModelRegistry()

    # Resolve version
    try:
        version_str = args.version.lstrip("v")
        version_int = int(version_str)
    except ValueError:
        _print_result(
            False,
            f"Invalid version: {args.version!r}",
            {},
            args.output_json,
        )
        return 1

    # Check version exists
    try:
        meta = registry.get_version(version_int)
    except KeyError:
        _print_result(
            False,
            f"Version v{version_int} not found in registry.",
            {},
            args.output_json,
        )
        return 1

    # Check current stage matches --from
    current_stage = registry.get_stage(version_int)
    if current_stage != args.from_stage:
        _print_result(
            False,
            (
                f"Stage mismatch: version v{version_int} is currently "
                f"{current_stage!r}, not {args.from_stage!r}."
            ),
            {"current_stage": current_stage, "expected": args.from_stage},
            args.output_json,
        )
        return 1

    # Check structural validity
    ok, condition_msg = registry.check_promotion_conditions(version_int, args.to_stage)
    if not ok and not args.force:
        _print_result(
            False,
            condition_msg,
            {
                "version": f"v{version_int}",
                "from_stage": args.from_stage,
                "to_stage": args.to_stage,
            },
            args.output_json,
        )
        return 1

    # Extra human-safety check for canary → prod
    if args.from_stage == "canary" and args.to_stage == "prod":
        if not args.reason:
            _print_result(
                False,
                "canary → prod requires an explicit --reason (e.g. '7d canary passed, ruin_prob<1%').",
                {},
                args.output_json,
            )
            return 1
        if args.actor == "unknown":
            _print_result(
                False,
                "canary → prod requires --actor to be set (human must approve).",
                {},
                args.output_json,
            )
            return 1

    # Print criteria reminder for human stages (not in JSON mode — would corrupt output)
    criteria = PROMOTION_CRITERIA.get((args.from_stage, args.to_stage), "")
    if criteria and not args.check and not args.output_json:
        print(f"\n[INFO] Criteria for {args.from_stage!r} → {args.to_stage!r}: {criteria}")
        print("[INFO] Confirm you have verified these criteria before proceeding.\n")

    if args.check:
        # Dry-run: print what would happen
        _print_result(
            ok or args.force,
            f"DRY-RUN: {condition_msg}",
            {
                "version": f"v{version_int}",
                "name": meta.get("name", ""),
                "current_stage": current_stage,
                "to_stage": args.to_stage,
                "criteria": criteria,
            },
            args.output_json,
        )
        return 0

    # Perform promotion
    try:
        registry.promote(
            version_int,
            to_stage=args.to_stage,
            actor=args.actor,
            reason=args.reason,
            force=args.force,
        )
    except (ValueError, KeyError) as exc:
        _print_result(False, str(exc), {}, args.output_json)
        return 1

    history = registry.get_promotion_history(version_int)
    last_event = history[-1] if history else {}

    _print_result(
        True,
        f"Promoted v{version_int} ({meta.get('name', '')!r}): "
        f"{args.from_stage!r} → {args.to_stage!r}",
        {
            "version": f"v{version_int}",
            "actor": args.actor,
            "reason": args.reason,
            "timestamp": last_event.get("timestamp", ""),
            "total_history_events": len(history),
        },
        args.output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
