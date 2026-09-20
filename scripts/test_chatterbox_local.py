from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import chatterbox_tts_server as server


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=(
            "Die hübsche Frau entwickelte eine langfristige Strategie. "
            "Ihre Stimme klang ruhig, deutlich und natürlich."
        ),
    )
    parser.add_argument(
        "--reference",
        default="",
        help="Optional local German reference voice file.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "chatterbox_smoke_test.wav"),
    )
    args = parser.parse_args()

    model_dir = (
        ROOT / "models" / "chatterbox" / "multilingual-v3"
    )
    server.load_model(model_dir, args.device)

    wav, elapsed = server.synthesize(
        args.text,
        language_id="de",
        audio_prompt_path=args.reference,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
    )

    output = Path(args.output)
    output.write_bytes(wav)

    print()
    print("Chatterbox local German smoke test PASSED")
    print(f"Device:  {server.State.device}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Output:  {output}")
    print(
        "Reference: "
        + (args.reference if args.reference else "built-in voice")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
