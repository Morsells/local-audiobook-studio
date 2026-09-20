from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge_tree(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative

        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy model files and persistent project data from V1 or V2 into V3."
    )
    parser.add_argument("old_folder")
    args = parser.parse_args()

    old = Path(args.old_folder).expanduser().resolve()
    if not old.exists():
        raise SystemExit(f"Old project folder does not exist: {old}")

    model_count = merge_tree(old / "models", ROOT / "models")
    data_count = merge_tree(old / ".audiobook_data", ROOT / ".audiobook_data")

    print(f"Copied {model_count} model file(s) and {data_count} project/cache file(s).")
    print("The old project was not modified.")
    print("Start V3 with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
