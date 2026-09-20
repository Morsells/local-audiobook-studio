from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PKUSEG_HOME = ROOT / "models" / "chatterbox" / "pkuseg"


def main() -> int:
    PKUSEG_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["PKUSEG_HOME"] = str(PKUSEG_HOME)

    print("Preparing Chatterbox tokenizer support assets...")
    print(f"PKUSEG_HOME: {PKUSEG_HOME}")
    print()
    print(
        "Chatterbox initializes its Chinese tokenizer even for German. "
        "This explicit setup step downloads that auxiliary model once so "
        "normal runtime never needs an outbound connection."
    )

    try:
        from spacy_pkuseg import pkuseg
    except Exception as exc:
        print(f"ERROR: could not import spacy_pkuseg: {exc}")
        return 1

    try:
        segmenter = pkuseg()
        # Tiny local functional check; no book/user text involved.
        result = segmenter.cut("中文测试")
        if not result:
            raise RuntimeError("pkuseg returned no tokens")
    except Exception as exc:
        print(f"ERROR: pkuseg model preparation failed: {exc}")
        return 2

    print()
    print("pkuseg auxiliary tokenizer model: ready")
    print("Normal Chatterbox runtime can use this project-local cache offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
