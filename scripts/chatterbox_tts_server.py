from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import inspect
import json
import re
from pathlib import Path
import sys
import threading
import time
import traceback
import unicodedata
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Chatterbox constructs its multilingual tokenizer eagerly. The upstream
# tokenizer initializes spacy-pkuseg even when the requested language is German.
# Keep that auxiliary model in our own project-local cache so runtime remains
# independent of user-profile caches and never needs a network fallback.
PKUSEG_HOME = ROOT / "models" / "chatterbox" / "pkuseg"
PKUSEG_HOME.mkdir(parents=True, exist_ok=True)
__import__("os").environ["PKUSEG_HOME"] = str(PKUSEG_HOME)

from src.audiobook_studio.privacy import (
    enforce_offline_model_runtime,
    install_socket_guard,
)

# Critical: after explicit setup/download, ordinary Chatterbox runtime is offline.
enforce_offline_model_runtime()
install_socket_guard()


class State:
    model = None
    model_dir: Path | None = None
    device = "cpu"
    gpu_name = ""
    vram_gb = 0.0
    sample_rate = 24000
    builtin_conds = None
    reference_key = ""
    lock = threading.Lock()
    t3_inference_official = None


def resolve_device(torch, requested: str) -> str:
    requested = requested.lower().strip()

    if requested == "cpu":
        return "cpu"

    if requested in {"gpu", "cuda", "cuda:0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_dir: Path, device_request: str) -> None:
    try:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        raise RuntimeError(
            "Chatterbox dependencies are missing. Run Setup-Chatterbox-GPU.bat."
        ) from exc

    required = [
        "ve.pt",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise RuntimeError(
            "Local Chatterbox model is incomplete. Missing: "
            + ", ".join(missing)
        )

    device = resolve_device(torch, device_request)

    if device == "cpu":
        logical = max(1, __import__("os").cpu_count() or 1)
        try:
            torch.set_num_threads(max(1, logical - 1))
        except Exception:
            pass
    else:
        # Safe performance switches for Turing/RTX 20xx.
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    print(
        f"Loading Chatterbox Multilingual V3 from {model_dir} on {device}...",
        flush=True,
    )

    # Privacy-critical: use from_local(), never from_pretrained().
    from_local_signature = inspect.signature(
        ChatterboxMultilingualTTS.from_local
    )
    if "t3_model" not in from_local_signature.parameters:
        raise RuntimeError(
            "Installed Chatterbox package predates the Multilingual V3 local API. "
            "Run Setup-Chatterbox-GPU.bat again to install the official V3 release build."
        )

    model = ChatterboxMultilingualTTS.from_local(
        model_dir,
        device=device,
        t3_model="v3",
    )

    State.model = model
    State.model_dir = model_dir
    State.device = device
    State.sample_rate = int(model.sr)
    State.builtin_conds = model.conds
    State.reference_key = ""
    State.t3_inference_official = model.t3.inference

    if device.startswith("cuda"):
        try:
            props = torch.cuda.get_device_properties(0)
            State.gpu_name = str(props.name)
            State.vram_gb = float(props.total_memory) / (1024.0 ** 3)
        except Exception:
            pass

    print(
        "Chatterbox ready. "
        f"device={State.device}, GPU={State.gpu_name or 'none'}, "
        f"VRAM={State.vram_gb:.1f} GB",
        flush=True,
    )


def _reference_key(path_text: str) -> str:
    if not path_text:
        return ""

    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference voice file does not exist: {path}")

    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def prepare_reference(path_text: str, exaggeration: float) -> None:
    key = _reference_key(path_text)

    if not key:
        if State.reference_key:
            State.model.conds = State.builtin_conds
            State.reference_key = ""
        return

    if key == State.reference_key:
        return

    path = Path(path_text).expanduser().resolve()

    print(
        f"Preparing local Chatterbox reference voice: {path.name}",
        flush=True,
    )
    State.model.prepare_conditionals(
        str(path),
        exaggeration=float(exaggeration),
    )
    State.reference_key = key



_ABBREVIATIONS = {
    "dr.", "prof.", "mr.", "mrs.", "ms.", "st.", "jr.", "sr.",
    "bzw.", "ca.", "z.b.", "u.a.", "d.h.", "etc.", "vgl.",
}


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _looks_like_abbreviation(fragment: str) -> bool:
    stripped = fragment.strip().lower()
    last = stripped.split()[-1] if stripped.split() else ""
    return last in _ABBREVIATIONS or bool(
        re.search(r"(?:\b[A-Za-zÄÖÜäöü]\.){2,}$", stripped)
    )


def _hard_split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]
    return [
        " ".join(words[i:i + max_words]).strip()
        for i in range(0, len(words), max_words)
    ]


def _activate_inference_backend(name: str) -> str:
    if State.model is None:
        raise RuntimeError("Chatterbox model is not loaded.")

    normalized = (name or "official").strip().lower()
    allowed = {"auto", "official", "cuda_graph"}
    if normalized not in allowed:
        raise ValueError(
            f"Unknown inference backend {name!r}; expected one of {sorted(allowed)}."
        )

    if normalized == "auto":
        normalized = (
            "cuda_graph"
            if State.device.startswith("cuda")
            else "official"
        )

    t3 = State.model.t3
    if normalized == "official":
        t3.inference = State.t3_inference_official
    else:
        from scripts.chatterbox_fast_backend import inference_cuda_graph
        t3.inference = types.MethodType(inference_cuda_graph, t3)

    return normalized


_UNICODE_TTS_FALLBACKS = {"ı": "i", "Ł": "L", "ł": "l"}

def _cp1252_encodable(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False

def _windows_tts_compat_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    source = unicodedata.normalize("NFC", text)
    output, changes = [], []
    for char in source:
        if _cp1252_encodable(char):
            output.append(char); continue
        replacement = _UNICODE_TTS_FALLBACKS.get(char, "")
        if not replacement and "LATIN" in unicodedata.name(char, ""):
            decomposed = unicodedata.normalize("NFKD", char)
            candidate = "".join(c for c in decomposed if not unicodedata.combining(c))
            if candidate and _cp1252_encodable(candidate): replacement = candidate
        if replacement:
            output.append(replacement); changes.append((char, replacement))
        else:
            output.append(char)
    return "".join(output), changes

def _generate_chatterbox_phrase(phrase: str, *, language_id: str, exaggeration: float, cfg_weight: float, temperature: float, repetition_penalty: float):
    kwargs = {"language_id": language_id, "audio_prompt_path": None, "exaggeration": float(exaggeration), "cfg_weight": float(cfg_weight), "temperature": float(temperature), "repetition_penalty": float(repetition_penalty)}
    try:
        return State.model.generate(phrase, **kwargs)
    except UnicodeEncodeError as exc:
        compatible, changes = _windows_tts_compat_text(phrase)
        if not changes or compatible == phrase: raise
        detail = ", ".join(f"U+{ord(a):04X} {a!r}->{b!r}" for a,b in changes)
        print(f"[Chatterbox] Windows Unicode compatibility retry after {type(exc).__name__}: {detail}", file=sys.stderr, flush=True)
        return State.model.generate(compatible, **kwargs)

def split_audiobook_phrases(
    text: str,
    *,
    max_phrase_words: int = 36,
    comma_pause_ms: int = 170,
    semicolon_pause_ms: int = 320,
    sentence_pause_ms: int = 520,
) -> list[tuple[str, int]]:
    """
    Split narration into short, punctuation-aware synthesis units.

    The model gets short phrases for fidelity; pauses are inserted in the WAV
    afterwards so punctuation does not depend solely on model prosody.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    max_phrase_words = max(12, min(int(max_phrase_words), 60))
    comma_pause_ms = max(0, min(int(comma_pause_ms), 1000))
    semicolon_pause_ms = max(0, min(int(semicolon_pause_ms), 1500))
    sentence_pause_ms = max(0, min(int(sentence_pause_ms), 2000))

    # First pass: sentence boundaries, but avoid common abbreviations.
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r'[.!?](?:["”’)\]]*)\s+', text):
        candidate = text[start:match.end()].strip()
        if not candidate:
            start = match.end()
            continue
        if _looks_like_abbreviation(candidate):
            continue
        sentences.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    results: list[tuple[str, int]] = []

    for sentence in sentences:
        sentence_end_pause = (
            sentence_pause_ms
            if re.search(r'[.!?](?:["”’)\]]*)$', sentence)
            else 0
        )

        # Split semicolons/colons first. They need a real audible breath.
        major_parts = re.split(r'(?<=[;:])\s+', sentence)

        for major_index, major in enumerate(major_parts):
            major = major.strip()
            if not major:
                continue

            major_end = ""
            if re.search(r'[;:]$', major):
                major_end = major[-1]

            # Comma splitting is selective: only use it when the current clause
            # is long enough to benefit from a breath. This avoids choppy
            # two-word fragments.
            comma_parts = re.split(r'(?<=,)\s+', major)

            current = ""
            for comma_index, part in enumerate(comma_parts):
                part = part.strip()
                if not part:
                    continue

                proposed = f"{current} {part}".strip() if current else part
                is_last_comma_part = comma_index == len(comma_parts) - 1

                if (
                    current
                    and _words(proposed) > max_phrase_words
                ):
                    pause = (
                        comma_pause_ms
                        if current.rstrip().endswith(",")
                        else 90
                    )
                    results.append((current.strip(), pause))
                    current = part
                else:
                    current = proposed

                # If a comma clause is already substantial, let it breathe now.
                if (
                    not is_last_comma_part
                    and _words(current) >= 10
                    and current.rstrip().endswith(",")
                ):
                    results.append((current.strip(), comma_pause_ms))
                    current = ""

            if current:
                pieces = _hard_split_words(current, max_phrase_words)
                for piece_index, piece in enumerate(pieces):
                    is_last_piece = piece_index == len(pieces) - 1
                    is_last_major = major_index == len(major_parts) - 1

                    if not is_last_piece:
                        pause = 90
                    elif major_end in {";", ":"}:
                        pause = semicolon_pause_ms
                    elif is_last_major:
                        pause = sentence_end_pause
                    else:
                        pause = semicolon_pause_ms

                    results.append((piece.strip(), pause))

    # Never append artificial silence after the final phrase; the outer pipeline
    # owns section/chapter pauses.
    if results:
        last_text, _ = results[-1]
        results[-1] = (last_text, 0)

    return [(phrase, pause) for phrase, pause in results if phrase.strip()]

def synthesize(
    text: str,
    *,
    language_id: str,
    audio_prompt_path: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    audiobook_pacing: bool = True,
    max_phrase_words: int = 40,
    comma_pause_ms: int = 100,
    semicolon_pause_ms: int = 230,
    sentence_pause_ms: int = 430,
    inference_backend: str = "official",
) -> tuple[bytes, float, str]:
    import numpy as np
    import soundfile as sf

    if State.model is None:
        raise RuntimeError("Chatterbox model is not loaded.")

    started = time.perf_counter()

    if audiobook_pacing:
        phrases = split_audiobook_phrases(
            text,
            max_phrase_words=max_phrase_words,
            comma_pause_ms=comma_pause_ms,
            semicolon_pause_ms=semicolon_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
        )
    else:
        phrases = [(text.strip(), 0)]

    if not phrases:
        raise ValueError("Text is empty after narration segmentation.")

    arrays: list[np.ndarray] = []

    # One model/GPU context is shared by the local server. Serializing generation
    # avoids CUDA/model-state races, especially while switching reference voices.
    with State.lock:
        prepare_reference(audio_prompt_path, exaggeration)

        previous_inference = State.model.t3.inference
        requested_backend = str(inference_backend or "official").strip().lower()
        active_backend = _activate_inference_backend(requested_backend)

        try:
            for index, (phrase, pause_ms) in enumerate(phrases, start=1):
                print(
                    f"[Chatterbox] phrase {index}/{len(phrases)} "
                    f"({_words(phrase)} words, pause={pause_ms}ms): "
                    f"{phrase[:100]}",
                    flush=True,
                )

                try:
                    wav = _generate_chatterbox_phrase(
                        phrase,
                        language_id=language_id,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                    )
                except Exception as exc:
                    if requested_backend == "auto" and active_backend == "cuda_graph":
                        print(
                            "[Chatterbox] Fast Quality failed; falling back to "
                            f"Official: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        active_backend = _activate_inference_backend("official")
                        wav = _generate_chatterbox_phrase(
                            phrase,
                            language_id=language_id,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                        )
                    else:
                        raise

                if hasattr(wav, "detach"):
                    audio = wav.detach().float().cpu().numpy()
                else:
                    audio = np.asarray(wav, dtype=np.float32)

                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    arrays.append(audio)

                if pause_ms > 0:
                    arrays.append(
                        np.zeros(
                            int(State.sample_rate * pause_ms / 1000),
                            dtype=np.float32,
                        )
                    )
        finally:
            State.model.t3.inference = previous_inference

    if not arrays:
        raise RuntimeError("Chatterbox returned no audio.")

    combined = np.concatenate(arrays)
    buffer = io.BytesIO()
    sf.write(buffer, combined, State.sample_rate, format="WAV")

    return buffer.getvalue(), time.perf_counter() - started, active_backend


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalChatterboxTTS/0.39.1"

    def log_message(self, fmt, *args):
        print("[Chatterbox]", fmt % args)

    def _json(self, status: int, data: dict):
        payload = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "model": "Chatterbox Multilingual V3",
                    "model_dir": str(State.model_dir or ""),
                    "device": State.device,
                    "gpu_name": State.gpu_name,
                    "vram_gb": round(State.vram_gb, 2),
                    "sample_rate": State.sample_rate,
                    "custom_reference_active": bool(State.reference_key),
                    "runtime": "local-only/from_local",
                    "python_utf8": bool(__import__("os").environ.get("PYTHONUTF8")),
                    "unicode_tts_fallback": True,
                    "inference_backends": [
                        "auto",
                        "official",
                        "cuda_graph",
                    ],
                    "fast_quality_cuda_graph": bool(
                        State.device.startswith("cuda")
                    ),
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
            data = json.loads(
                self.rfile.read(length).decode("utf-8")
            )

            text = str(data.get("text", "")).strip()
            language_id = str(
                data.get("language_id", "de")
            ).strip().lower()
            audio_prompt_path = str(
                data.get("audio_prompt_path", "")
            ).strip()

            exaggeration = float(
                data.get("exaggeration", 0.35)
            )
            cfg_weight = float(
                data.get("cfg_weight", 0.30)
            )
            temperature = float(
                data.get("temperature", 0.55)
            )
            repetition_penalty = float(
                data.get("repetition_penalty", 1.15)
            )
            audiobook_pacing = bool(
                data.get("audiobook_pacing", True)
            )
            max_phrase_words = int(
                data.get("max_phrase_words", 40)
            )
            comma_pause_ms = int(
                data.get("comma_pause_ms", 100)
            )
            semicolon_pause_ms = int(
                data.get("semicolon_pause_ms", 230)
            )
            sentence_pause_ms = int(
                data.get("sentence_pause_ms", 430)
            )
            inference_backend = str(
                data.get("inference_backend", "official")
            ).strip().lower()

            if not text:
                raise ValueError("Text is empty.")

            payload, elapsed, active_backend = synthesize(
                text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                audiobook_pacing=audiobook_pacing,
                max_phrase_words=max_phrase_words,
                comma_pause_ms=comma_pause_ms,
                semicolon_pause_ms=semicolon_pause_ms,
                sentence_pause_ms=sentence_pause_ms,
                inference_backend=inference_backend,
            )

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header(
                "X-Generation-Seconds",
                f"{elapsed:.4f}",
            )
            self.send_header(
                "X-Inference-Backend",
                active_backend,
            )

            self.send_header(
                "Content-Length",
                str(len(payload)),
            )
            self.end_headers()
            self.wfile.write(payload)

        except Exception as exc:
            print(
                "[Chatterbox] synthesis error:\n"
                + traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
            self._json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "device": State.device,
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only Chatterbox Multilingual V3 sidecar."
        )
    )
    parser.add_argument(
        "--model-dir",
        default=str(
            ROOT
            / "models"
            / "chatterbox"
            / "multilingual-v3"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--port", type=int, default=7869)
    args = parser.parse_args()

    load_model(
        Path(args.model_dir).expanduser().resolve(),
        args.device,
    )

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        Handler,
    )

    print(
        "Local Chatterbox TTS listening only on "
        f"http://127.0.0.1:{args.port}",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
