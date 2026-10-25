"""Developer Command-Line Interface (CLI) for payload-heal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from payload_heal.lockfile import DEFAULT_LOCKFILE_PATH, LockFileManager
from payload_heal.skeleton import extract_json_skeleton


def cmd_status(args: argparse.Namespace) -> int:
    """Pretty-prints all healed models and field mappings stored in the lockfile."""
    lockfile_path = Path(args.lockfile)
    mgr = LockFileManager(lockfile_path)
    raw = mgr._read_raw()
    models = raw.get("models", {})

    if not models:
        print(f"No healed mappings found in lockfile ({lockfile_path}).")
        return 0

    print("=" * 80)
    print(f"PAYLOAD-HEAL LOCKFILE STATUS: {lockfile_path}")
    print("=" * 80)

    total_models = len(models)
    total_fields = 0
    heuristic_count = 0
    ai_count = 0

    for model_name, fields in models.items():
        print(f"\n[Model: {model_name}]")
        print(f"{'Field':<20} {'Source Path':<35} {'Healed By':<12} {'Updated At'}")
        print("-" * 80)

        for field_name, info in fields.items():
            if not isinstance(info, dict):
                continue
            total_fields += 1
            source_path = str(info.get("source_path", "unknown"))
            healed_by = str(info.get("healed_by", "unknown"))
            updated_at = str(info.get("updated_at", ""))
            # Truncate ISO timestamp to date + time (YYYY-MM-DD HH:MM:SS)
            if "T" in updated_at:
                updated_at = updated_at.replace("T", " ")[:19]

            if healed_by == "heuristic":
                heuristic_count += 1
            elif healed_by == "ai":
                ai_count += 1

            # Format source_path with truncation if necessary
            if len(source_path) > 33:
                source_path_display = source_path[:30] + "..."
            else:
                source_path_display = source_path

            print(f"{field_name:<20} {source_path_display:<35} {healed_by:<12} {updated_at}")

    print("\n" + "-" * 80)
    print(
        f"Summary: {total_models} model(s), {total_fields} field(s) "
        f"({heuristic_count} heuristic, {ai_count} AI)"
    )
    print("=" * 80)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Clears all lockfile mappings or resets a specific model."""
    lockfile_path = Path(args.lockfile)
    mgr = LockFileManager(lockfile_path)

    if args.model:
        removed = mgr.remove_model(args.model)
        if removed:
            print(f"[OK] Successfully reset mappings for model '{args.model}' in {lockfile_path}.")
        else:
            print(f"Model '{args.model}' was not found in lockfile ({lockfile_path}).")
    else:
        mgr.clear()
        print(f"[OK] Successfully reset all mappings in lockfile ({lockfile_path}).")

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Extracts and prints the compressed JSON skeleton with token reduction metrics."""
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File '{json_path}' does not exist.", file=sys.stderr)
        return 1

    try:
        raw_content = json_path.read_text(encoding="utf-8")
        data = json.loads(raw_content)
    except Exception as exc:
        print(f"Error reading or parsing JSON from '{json_path}': {exc}", file=sys.stderr)
        return 1

    skeleton = extract_json_skeleton(data, max_list_sample=args.max_sample)
    skeleton_json = json.dumps(skeleton, indent=2)

    raw_chars = len(raw_content)
    skeleton_chars = len(skeleton_json)
    reduction = max(0.0, ((raw_chars - skeleton_chars) / raw_chars) * 100) if raw_chars > 0 else 0.0

    print("=" * 80)
    print(f"PAYLOAD-HEAL SKELETON INSPECTION: {json_path}")
    print("=" * 80)
    print(
        f"Raw JSON: {raw_chars:,} chars -> Skeleton: {skeleton_chars:,} chars "
        f"({reduction:.1f}% reduction)"
    )
    print("-" * 80)
    print(skeleton_json)
    print("=" * 80)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="payload-heal",
        description="CLI tool to manage and inspect payload-heal lockfiles and schema skeletons.",
    )
    parser.add_argument(
        "--lockfile",
        "-l",
        default=str(DEFAULT_LOCKFILE_PATH),
        help="Path to the lockfile (default: .payload_heal_lock.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # status command
    subparsers.add_parser("status", help="Show all healed models and field mappings.")

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset all mappings or a specific model.")
    reset_parser.add_argument("model", nargs="?", help="Optional model name to reset.")

    # inspect command
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Extract compressed skeleton from a JSON file and show reduction stats.",
    )
    inspect_parser.add_argument("json_file", help="Path to raw JSON file.")
    inspect_parser.add_argument(
        "--max-sample",
        type=int,
        default=1,
        help="Maximum number of list elements to inspect (default: 1)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        return cmd_status(args)
    if args.command == "reset":
        return cmd_reset(args)
    if args.command == "inspect":
        return cmd_inspect(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
