from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "0.6b-custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7b-custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7b-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly download Qwen3-TTS weights for fully-local runtime."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="0.6b-custom",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub in the Qwen environment first:\n"
            "  pip install huggingface_hub"
        ) from exc

    repo_id = MODELS[args.model]
    folder_name = repo_id.split("/")[-1]
    local_dir = ROOT / "models" / "qwen3-tts" / folder_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )

    print("\nDone. Start the local Qwen server using this local model directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
