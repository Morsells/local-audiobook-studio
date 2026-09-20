from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "chatterbox" / "multilingual-v3"

REPO_ID = "ResembleAI/chatterbox"
FILES = [
    "ve.pt",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


def main() -> int:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)

    required = [
        "ve.pt",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
    ]

    if all((DEST / name).exists() for name in required):
        print("Chatterbox Multilingual V3 model files are already complete.")
        print(f"Reusing existing local model: {DEST}")
        print("No model re-download is needed.")
        return 0

    print("Downloading Chatterbox Multilingual V3 for LOCAL runtime...")
    print(f"Repository: {REPO_ID}")
    print(f"Destination: {DEST}")
    print()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        revision="main",
        allow_patterns=FILES,
        local_dir=str(DEST),
    )

    missing = [name for name in FILES[:-1] if not (DEST / name).exists()]
    if missing:
        raise RuntimeError(
            "Chatterbox download finished but required files are missing: "
            + ", ".join(missing)
        )

    # local_dir may contain Hugging Face update metadata. It is unnecessary for
    # V3.6 runtime because the sidecar loads through from_local() only.
    shutil.rmtree(DEST / ".cache", ignore_errors=True)

    print()
    print("Chatterbox Multilingual V3 ready.")
    print("Normal audiobook runtime can now remain offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
