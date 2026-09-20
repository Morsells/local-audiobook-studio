from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from .config import (
    CHATTERBOX_MODEL_DIR,
    CHATTERBOX_REQUIRED_FILES,
    KOKORO_MODEL,
    KOKORO_VOICES,
)
from .models import AudioChunk, BuildOptions, TTSEngine
from .privacy import assert_local_url


class ModelMissingError(RuntimeError):
    pass


class TTSUnavailableError(RuntimeError):
    pass


class KokoroEngine:
    engine_id = "kokoro-v1.0"

    def __init__(
        self,
        model_path: Path = KOKORO_MODEL,
        voices_path: Path = KOKORO_VOICES,
    ):
        if not model_path.exists() or not voices_path.exists():
            raise ModelMissingError(
                "Kokoro model files are missing. Run: python scripts/download_models.py"
            )
        from kokoro_onnx import Kokoro
        self.kokoro = Kokoro(str(model_path), str(voices_path))

    def cache_signature(self, chunk: AudioChunk, options: BuildOptions) -> str:
        payload = json.dumps(
            {
                "text": chunk.text,
                "voice": options.voice,
                "language": options.language,
                "speed": round(options.speed, 3),
                "engine": self.engine_id,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def synthesize_chunk(
        self,
        chunk: AudioChunk,
        destination: Path,
        options: BuildOptions,
    ) -> dict[str, float]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        samples, sample_rate = self.kokoro.create(
            chunk.text,
            voice=options.voice,
            speed=options.speed,
            lang=options.language,
        )
        elapsed = time.perf_counter() - started
        sf.write(str(destination), samples, sample_rate)
        duration = len(samples) / float(sample_rate)
        return {
            "elapsed": elapsed,
            "duration": duration,
            "words": float(chunk.word_count),
        }


def local_server_health(url: str, timeout: float = 2.0) -> dict:
    try:
        assert_local_url(url)
        with urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def chatterbox_model_ready() -> bool:
    return all(
        (CHATTERBOX_MODEL_DIR / filename).exists()
        for filename in CHATTERBOX_REQUIRED_FILES
    )


def chatterbox_server_available(
    url: str = "http://127.0.0.1:7869",
    timeout: float = 1.0,
) -> bool:
    try:
        assert_local_url(url)
        with urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def chatterbox_server_health(
    url: str = "http://127.0.0.1:7869",
) -> dict:
    return local_server_health(url)


class ChatterboxServerEngine:
    engine_id = "chatterbox-multilingual-v3-local-server"

    def __init__(self, url: str):
        assert_local_url(url)
        self.url = url.rstrip("/")

        if not self.available():
            raise TTSUnavailableError(
                "The local Chatterbox Multilingual V3 server is not running. "
                "Start with Start-Audiobook-Studio-With-Chatterbox.bat "
                "or the normal launcher."
            )

        # One health lookup per engine instance is enough to resolve Auto and
        # keeps per-chunk synthesis free of extra HTTP probes.
        self._health = self.health() or {}

    def resolve_inference_backend(self, options: BuildOptions) -> str:
        requested = str(
            getattr(options, "chatterbox_inference_backend", "auto") or "auto"
        ).strip().lower()

        aliases = {
            "fast": "cuda_graph",
            "fast_quality": "cuda_graph",
            "compatibility": "official",
        }
        requested = aliases.get(requested, requested)

        supported = set(
            self._health.get("inference_backends", [])
            or []
        )
        cuda_graph_ready = bool(
            self._health.get("fast_quality_cuda_graph", False)
            and "cuda_graph" in supported
        )

        if requested == "auto":
            return "cuda_graph" if cuda_graph_ready else "official"

        if requested == "cuda_graph" and not cuda_graph_ready:
            raise TTSUnavailableError(
                "Fast Quality / CUDA Graph was selected, but the running "
                "Chatterbox sidecar does not support it. Restart with the "
                "current sidecar or choose Auto / Official."
            )

        if requested != "official":
            raise ValueError(
                f"Unknown Chatterbox inference backend: {requested!r}"
            )
        return "official"

    def performance_key(self, options: BuildOptions) -> str:
        return f"{self.engine_id}:{self.resolve_inference_backend(options)}"

    def available(self, timeout: float = 1.5) -> bool:
        return chatterbox_server_available(self.url, timeout=timeout)

    def health(self) -> dict:
        return chatterbox_server_health(self.url)

    def cache_signature(self, chunk: AudioChunk, options: BuildOptions) -> str:
        reference_path = str(options.chatterbox_reference_path or "")
        reference_fingerprint = ""
        if reference_path:
            ref = Path(reference_path)
            if ref.exists():
                stat = ref.stat()
                reference_fingerprint = (
                    f"{ref.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
                )

        payload = json.dumps(
            {
                "text": chunk.text,
                "engine": self.engine_id,
                "server": self.url,
                "language": options.language,
                "inference_backend": self.resolve_inference_backend(options),
                "reference": reference_fingerprint,
                "exaggeration": round(options.chatterbox_exaggeration, 3),
                "cfg_weight": round(options.chatterbox_cfg_weight, 3),
                "temperature": round(options.chatterbox_temperature, 3),
                "repetition_penalty": round(
                    options.chatterbox_repetition_penalty,
                    3,
                ),
                "audiobook_pacing": bool(
                    options.chatterbox_audiobook_pacing
                ),
                "max_phrase_words": int(
                    options.chatterbox_max_phrase_words
                ),
                "comma_pause_ms": int(
                    options.chatterbox_comma_pause_ms
                ),
                "semicolon_pause_ms": int(
                    options.chatterbox_semicolon_pause_ms
                ),
                "sentence_pause_ms": int(
                    options.chatterbox_sentence_pause_ms
                ),
                "speed": round(options.speed, 3),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def synthesize_chunk(
        self,
        chunk: AudioChunk,
        destination: Path,
        options: BuildOptions,
    ) -> dict[str, float]:
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            {
                "text": chunk.text,
                "language_id": options.language,
                "inference_backend": self.resolve_inference_backend(options),
                "audio_prompt_path": str(
                    options.chatterbox_reference_path or ""
                ),
                "exaggeration": float(options.chatterbox_exaggeration),
                "cfg_weight": float(options.chatterbox_cfg_weight),
                "temperature": float(options.chatterbox_temperature),
                "repetition_penalty": float(
                    options.chatterbox_repetition_penalty
                ),
                "audiobook_pacing": bool(
                    options.chatterbox_audiobook_pacing
                ),
                "max_phrase_words": int(
                    options.chatterbox_max_phrase_words
                ),
                "comma_pause_ms": int(
                    options.chatterbox_comma_pause_ms
                ),
                "semicolon_pause_ms": int(
                    options.chatterbox_semicolon_pause_ms
                ),
                "sentence_pause_ms": int(
                    options.chatterbox_sentence_pause_ms
                ),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            f"{self.url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urlopen(request, timeout=1800) as response:
                wav_bytes = response.read()
                header_elapsed = response.headers.get(
                    "X-Generation-Seconds"
                )
                actual_backend = response.headers.get(
                    "X-Inference-Backend",
                    self.resolve_inference_backend(options),
                )
        except HTTPError as exc:
            detail = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                detail = str(parsed.get("error", raw)).strip()
            except Exception:
                detail = ""

            message = "Local Chatterbox synthesis failed"
            if detail:
                message += f": {detail}"
            message += ". See logs/chatterbox.err.log for the server traceback."
            raise RuntimeError(message) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Local Chatterbox synthesis failed: "
                f"{type(exc).__name__}: {exc}. "
                "See logs/chatterbox.err.log."
            ) from exc

        destination.write_bytes(wav_bytes)

        # Chatterbox has no direct speaking-rate parameter. Preserve the
        # project's common speed control through the existing local FFmpeg path.
        if abs(float(options.speed) - 1.0) > 0.01:
            change_wav_tempo(destination, options.speed)

        elapsed = time.perf_counter() - started
        if header_elapsed:
            try:
                elapsed = float(header_elapsed)
            except ValueError:
                pass

        duration = audio_duration(destination)
        return {
            "elapsed": elapsed,
            "duration": duration,
            "words": float(chunk.word_count),
            "backend": str(actual_backend),
            "performance_key": f"{self.engine_id}:{actual_backend}",
        }


class QwenServerEngine:
    engine_id = "qwen3-tts-local-server"

    def __init__(self, url: str):
        assert_local_url(url)
        self.url = url.rstrip("/")
        if not self.available():
            raise TTSUnavailableError(
                "The local Qwen3-TTS server is not running. "
                "Start it with the command shown in the Setup tab."
            )

    def available(self, timeout: float = 1.5) -> bool:
        try:
            with urlopen(f"{self.url}/health", timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    def health(self) -> dict:
        try:
            with urlopen(f"{self.url}/health", timeout=2.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return {}

    def cache_signature(self, chunk: AudioChunk, options: BuildOptions) -> str:
        payload = json.dumps(
            {
                "text": chunk.text,
                "engine": self.engine_id,
                "server": self.url,
                "mode": options.qwen_mode,
                "speaker": options.qwen_speaker,
                "language": options.language,
                "instruct": options.qwen_instruct,
                "speed": round(options.speed, 3),
                "model_label": options.qwen_model_label,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def synthesize_chunk(
        self,
        chunk: AudioChunk,
        destination: Path,
        options: BuildOptions,
    ) -> dict[str, float]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "text": chunk.text,
                "language": qwen_language_name(options.language),
                "mode": options.qwen_mode,
                "speaker": options.qwen_speaker,
                "instruct": options.qwen_instruct,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            f"{self.url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=1800) as response:
                wav_bytes = response.read()
        except Exception as exc:
            raise RuntimeError(
                "Local Qwen3-TTS synthesis failed. Check the Qwen server terminal."
            ) from exc

        destination.write_bytes(wav_bytes)
        if abs(float(options.speed) - 1.0) > 0.01:
            change_wav_tempo(destination, options.speed)

        elapsed = time.perf_counter() - started
        duration = audio_duration(destination)
        return {
            "elapsed": elapsed,
            "duration": duration,
            "words": float(chunk.word_count),
        }


def qwen_language_name(value: str) -> str:
    normalized = value.lower().strip()
    mapping = {
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
        "de": "German",
        "de-de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "cmn": "Chinese",
    }
    return mapping.get(normalized, value)


def qwen_server_available(url: str = "http://127.0.0.1:7867") -> bool:
    try:
        assert_local_url(url)
        with urlopen(url.rstrip("/") + "/health", timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def qwen_server_health(
    url: str = "http://127.0.0.1:7867",
) -> dict:
    return local_server_health(url)


def make_tts_engine(options: BuildOptions):
    if options.tts_engine == TTSEngine.QWEN3:
        return QwenServerEngine(options.qwen_server_url)
    if options.tts_engine == TTSEngine.CHATTERBOX:
        return ChatterboxServerEngine(options.chatterbox_server_url)
    return KokoroEngine()


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def audio_duration(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def merge_wavs_with_pauses(
    items: list[tuple[Path, int]],
    destination: Path,
) -> Path:
    if not items:
        raise ValueError("No audio files to merge.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    arrays: list[np.ndarray] = []
    common_rate: int | None = None

    for file, pause_ms in items:
        samples, sample_rate = read_audio(file)
        if common_rate is None:
            common_rate = sample_rate
        elif sample_rate != common_rate:
            raise RuntimeError(
                f"Sample-rate mismatch: {sample_rate} != {common_rate}"
            )
        arrays.append(samples)
        if pause_ms > 0:
            arrays.append(
                np.zeros(
                    int(common_rate * pause_ms / 1000),
                    dtype=np.float32,
                )
            )

    merged = np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)
    sf.write(str(destination), merged, common_rate or 24000)
    return destination


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def change_wav_tempo(path: Path, speed: float) -> Path:
    speed = float(speed)
    if abs(speed - 1.0) <= 0.01:
        return path
    if not 0.5 <= speed <= 2.0:
        raise ValueError("Tempo adjustment supports 0.5x to 2.0x.")

    ffmpeg = _ffmpeg_exe()
    temp = path.with_suffix(".tempo.wav")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-filter:a",
        f"atempo={speed:.4f}",
        str(temp),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg tempo adjustment failed:\n" + result.stderr[-1500:])
    temp.replace(path)
    return path


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    ffmpeg = _ffmpeg_exe()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            ffmpeg, "-y", "-i", str(wav_path), "-vn",
            "-codec:a", "libmp3lame", "-q:a", "3", str(mp3_path)
        ],
        [
            ffmpeg, "-y", "-i", str(wav_path), "-vn",
            "-codec:a", "mp3", "-b:a", "160k", str(mp3_path)
        ],
    ]
    last_error = ""
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return mp3_path
        last_error = result.stderr
    raise RuntimeError("FFmpeg MP3 conversion failed:\n" + last_error[-2000:])


def _ffmetadata_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _probe_duration_with_ffmpeg(path: Path, ffmpeg: str) -> float:
    duration = audio_duration(path)
    if duration > 0:
        return duration

    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr or "",
    )
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def export_m4b(
    chapter_wavs: list[tuple[str, Path]],
    destination: Path,
    *,
    title: str,
    author: str = "",
    cover_path: Path | None = None,
    chapter_durations: list[float] | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    """Create one chapterized M4B without building a giant intermediate WAV.

    FFmpeg decodes the chapter files directly and streams them into the final
    AAC/M4B encoder. This keeps RAM/temp-disk usage low and exposes real export
    progress to the UI.
    """
    if not chapter_wavs:
        raise ValueError("No chapter audio files available for M4B export.")

    ffmpeg = _ffmpeg_exe()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path: Path | None = None

    durations: list[float] = []
    chapters: list[tuple[str, int, int]] = []
    cursor_ms = 0

    try:
        if progress:
            progress("Preparing M4B chapters", 0.02)

        for index, (chapter_title, audio_path) in enumerate(chapter_wavs):
            known = 0.0
            if chapter_durations and index < len(chapter_durations):
                try:
                    known = float(chapter_durations[index] or 0.0)
                except Exception:
                    known = 0.0

            duration = known if known > 0 else _probe_duration_with_ffmpeg(audio_path, ffmpeg)
            if duration <= 0:
                raise RuntimeError(f"Could not determine duration of {audio_path.name}.")

            durations.append(duration)
            duration_ms = int(round(duration * 1000))
            chapters.append((chapter_title, cursor_ms, cursor_ms + duration_ms))
            cursor_ms += duration_ms

            if progress:
                fraction = 0.02 + 0.08 * (index + 1) / max(1, len(chapter_wavs))
                progress(
                    f"Reading chapter metadata {index + 1}/{len(chapter_wavs)}",
                    fraction,
                )

        metadata_lines = [
            ";FFMETADATA1",
            f"title={_ffmetadata_escape(title)}",
            f"artist={_ffmetadata_escape(author)}",
            f"album={_ffmetadata_escape(title)}",
        ]
        for chapter_title, start_ms, end_ms in chapters:
            metadata_lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={_ffmetadata_escape(chapter_title)}",
            ]

        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".ffmeta",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write("\n".join(metadata_lines))
            metadata_path = Path(handle.name)

        # Use individual inputs + concat filter. Unlike the concat demuxer this
        # also tolerates a rare WAV fallback mixed with normal MP3 chapters.
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        for _, audio_path in chapter_wavs:
            command += ["-i", str(audio_path)]

        metadata_input_index = len(chapter_wavs)
        command += ["-f", "ffmetadata", "-i", str(metadata_path)]

        cover_input_index: int | None = None
        if cover_path and cover_path.exists():
            cover_input_index = len(chapter_wavs) + 1
            command += ["-i", str(cover_path)]

        normalized_labels: list[str] = []
        filter_parts: list[str] = []
        for index in range(len(chapter_wavs)):
            label = f"a{index}"
            normalized_labels.append(f"[{label}]")
            filter_parts.append(
                f"[{index}:a:0]aresample=24000,"
                f"aformat=sample_fmts=fltp:channel_layouts=mono[{label}]"
            )
        filter_parts.append(
            "".join(normalized_labels)
            + f"concat=n={len(chapter_wavs)}:v=0:a=1[aout]"
        )

        command += [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[aout]",
            "-map_metadata",
            str(metadata_input_index),
        ]

        if cover_input_index is not None:
            command += [
                "-map",
                f"{cover_input_index}:v:0",
                "-c:v",
                "copy",
                "-disposition:v:0",
                "attached_pic",
            ]

        command += [
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            str(destination),
        ]

        if progress:
            progress("Encoding M4B", 0.10)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        total_seconds = max(0.001, sum(durations))
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key not in {"out_time_us", "out_time_ms"}:
                    continue
                try:
                    # FFmpeg reports these progress fields in microseconds.
                    encoded_seconds = int(value) / 1_000_000.0
                except (TypeError, ValueError):
                    continue
                ratio = min(1.0, max(0.0, encoded_seconds / total_seconds))
                if progress:
                    progress(
                        f"Encoding M4B — {ratio * 100:.0f}%",
                        0.10 + 0.88 * ratio,
                    )

        stderr = process.stderr.read() if process.stderr is not None else ""
        returncode = process.wait()
        if returncode != 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError("M4B export failed:\n" + stderr[-3000:])

        if progress:
            progress("M4B complete", 1.0)
        return destination

    finally:
        if metadata_path is not None:
            metadata_path.unlink(missing_ok=True)
