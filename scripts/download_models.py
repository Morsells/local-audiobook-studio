from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "kokoro-v1.0.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/kokoro-v1.0.onnx"
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/voices-v1.0.bin"
    ),
}


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"[skip] {destination.name} already exists")
        return

    request = Request(
        url,
        headers={
            "User-Agent": "PDF-Audiobook-Studio/0.1",
        },
    )

    print(f"[download] {destination.name}")

    with urlopen(request) as response, destination.open("wb") as output:
        total = int(
            response.headers.get("Content-Length", "0") or 0
        )
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)

            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1024 / 1024:.1f} MB / "
                    f"{total / 1024 / 1024:.1f} MB "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True,
                )

    print()


def main() -> int:
    for filename, url in FILES.items():
        destination = MODEL_DIR / filename

        try:
            download(
                url,
                destination,
            )
        except Exception as exc:
            destination.unlink(missing_ok=True)
            print(
                f"\nFailed to download {filename}: {exc}",
                file=sys.stderr,
            )
            return 1

    print("\nKokoro model files are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
