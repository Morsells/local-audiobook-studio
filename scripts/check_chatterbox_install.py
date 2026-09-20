from __future__ import annotations

import inspect
import sys


def main() -> int:
    try:
        import chatterbox
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except Exception as exc:
        print(f"ERROR: Chatterbox import failed: {exc}")
        return 1

    signature = inspect.signature(
        ChatterboxMultilingualTTS.from_local
    )

    print("Chatterbox installation check")
    print("-----------------------------")
    print(f"Package: {getattr(chatterbox, '__file__', '-')}")
    print(f"from_local signature: {signature}")

    if "t3_model" not in signature.parameters:
        print()
        print(
            "ERROR: Installed Chatterbox build does not support "
            "Multilingual V3 local loading."
        )
        print(
            "Rerun Setup-Chatterbox-GPU.bat so the official "
            "V3 release commit is installed."
        )
        return 2

    print("Multilingual V3 local API: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
