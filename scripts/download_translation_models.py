from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "en-de": ("Helsinki-NLP/opus-mt-en-de", ROOT / "models" / "translation" / "opus-mt-en-de"),
    "de-en": ("Helsinki-NLP/opus-mt-de-en", ROOT / "models" / "translation" / "opus-mt-de-en"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly download translation model weights for later fully-offline runtime."
    )
    parser.add_argument(
        "--pair",
        choices=["en-de", "de-en", "all"],
        default="en-de",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install translation dependencies first:\n"
            "  pip install -r requirements-translation.txt"
        ) from exc

    pairs = list(MODELS) if args.pair == "all" else [args.pair]

    for pair in pairs:
        repo_id, local_dir = MODELS[pair]
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )

    print("\nDone. The audiobook app uses these local folders with offline mode enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
