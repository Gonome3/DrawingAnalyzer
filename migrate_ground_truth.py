"""One-shot migration script for the schema change that drops `features`.

Strips two things from every JSON file in a directory:
  * The top-level `features` array (if present).
  * The `feature_ref` key from every object inside `dimensions`.

Everything else is left untouched. Files are rewritten in place with
2-space indentation and trailing newline. The script ONLY prints
aggregate counts -- it never echoes the contents of any file.

Usage:
    python migrate_ground_truth.py [directory]

If no directory is given, defaults to ./ground_truth.
"""

import json
import sys
from pathlib import Path


def migrate_one(path: Path) -> tuple[bool, bool, int]:
    """Migrate a single JSON file in place.

    Returns:
      (changed, features_removed, feature_refs_cleared)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        # Unexpected shape -- skip without touching the file.
        return (False, False, 0)

    features_removed = "features" in data
    if features_removed:
        del data["features"]

    feature_refs_cleared = 0
    dims = data.get("dimensions")
    if isinstance(dims, list):
        for dim in dims:
            if isinstance(dim, dict) and "feature_ref" in dim:
                del dim["feature_ref"]
                feature_refs_cleared += 1

    changed = features_removed or feature_refs_cleared > 0
    if changed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return (changed, features_removed, feature_refs_cleared)


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ground_truth")
    if not target.exists():
        print(f"Directory not found: {target}", file=sys.stderr)
        return 1
    if not target.is_dir():
        print(f"Not a directory: {target}", file=sys.stderr)
        return 1

    json_files = sorted(target.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {target}")
        return 0

    files_total = len(json_files)
    files_changed = 0
    files_with_features_removed = 0
    total_feature_refs_cleared = 0
    errors: list[str] = []

    for path in json_files:
        try:
            changed, features_removed, refs_cleared = migrate_one(path)
        except Exception as e:  # noqa: BLE001
            # Record the filename and error class only -- no file contents.
            errors.append(f"{path.name}: {type(e).__name__}: {e}")
            continue

        if changed:
            files_changed += 1
        if features_removed:
            files_with_features_removed += 1
        total_feature_refs_cleared += refs_cleared

    print(f"Scanned:                       {files_total} file(s) in {target}")
    print(f"Files modified:                {files_changed}")
    print(f"Files with `features` removed: {files_with_features_removed}")
    print(f"`feature_ref` keys cleared:    {total_feature_refs_cleared}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for line in errors:
            print(f"  {line}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
