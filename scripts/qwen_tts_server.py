from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audiobook_studio.privacy import enforce_offline_model_runtime, install_socket_guard

enforce_offline_model_runtime()
install_socket_guard()


def load_model(model_dir: Path, device: str):
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-tts is not installed in this environment. "
            "Create a separate .venv-qwen and run `pip install -U qwen-tts soundfile`."
        ) from exc

    if not model_dir.exists():
        raise RuntimeError(f"Local Qwen model directory does not exist: {model_dir}")

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kwargs = {"device_map": device}
    precision = "float32"

    if str(device).startswith("cuda"):
        # Do not force BF16 on older NVIDIA architectures such as Turing/RTX 20xx.
        # Torch's capability check is the source of truth.
        try:
            props = torch.cuda.get_device_properties(0)
            native_arch_support = int(props.major) >= 8
            try:
                torch_reports_bf16 = bool(
                    torch.cuda.is_bf16_supported(including_emulation=False)
                )
            except TypeError:
                torch_reports_bf16 = bool(torch.cuda.is_bf16_supported())
            bf16_supported = native_arch_support and torch_reports_bf16
        except Exception:
            bf16_supported = False

        if bf16_supported:
            kwargs["dtype"] = torch.bfloat16
            precision = "bfloat16"
        else:
            kwargs["dtype"] = torch.float16
            precision = "float16"

        kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["dtype"] = torch.float32

    gpu_name = ""
    vram_gb = 0.0
    if str(device).startswith("cuda"):
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            vram_gb = float(props.total_memory) / (1024.0 ** 3)
        except Exception:
            pass

    print(
        f"Loading Qwen3-TTS from {model_dir} on {device} "
        f"with {precision}..."
    )
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        **kwargs,
    )
    print(
        f"Qwen3-TTS ready. GPU={gpu_name or 'none'}, "
        f"VRAM={vram_gb:.1f} GB, precision={precision}"
    )
    return model, device, precision, gpu_name, vram_gb


class State:
    model = None
    mode = "custom_voice"
    model_dir = ""
    device = ""
    precision = "float32"
    gpu_name = ""
    vram_gb = 0.0


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQwenTTS/0.3"

    def log_message(self, format, *args):
        print("[QwenTTS]", format % args)

    def _json(self, status: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "mode": State.mode,
                    "model_dir": State.model_dir,
                    "device": State.device,
                    "precision": State.precision,
                    "gpu_name": State.gpu_name,
                    "vram_gb": round(State.vram_gb, 2),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/synthesize":
            self._json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(data.get("text", "")).strip()
            language = str(data.get("language", "English"))
            mode = str(data.get("mode", State.mode))
            speaker = str(data.get("speaker", "Ryan"))
            instruct = str(data.get("instruct", "")).strip()

            if not text:
                raise ValueError("Text is empty.")

            if mode == "voice_design":
                wavs, sr = State.model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct or "Natural audiobook narration.",
                )
            else:
                kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": speaker,
                }
                if instruct:
                    kwargs["instruct"] = instruct
                wavs, sr = State.model.generate_custom_voice(**kwargs)

            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sr, format="WAV")
            payload = buffer.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        except Exception as exc:
            self._json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local-only Qwen3-TTS sidecar server for Local Audiobook Studio V3."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["custom_voice", "voice_design"],
        default="custom_voice",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7867)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    (
        State.model,
        State.device,
        State.precision,
        State.gpu_name,
        State.vram_gb,
    ) = load_model(model_dir, args.device)
    State.mode = args.mode
    State.model_dir = str(model_dir)

    # Deliberately bind only to loopback.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local Qwen3-TTS server listening only on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
