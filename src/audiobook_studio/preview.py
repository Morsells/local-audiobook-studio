from __future__ import annotations

from .models import BuildOptions, TTSEngine


def resolve_source_preview_options(
    options: BuildOptions,
    source_language: str,
    chatterbox_available: bool = False,
) -> tuple[BuildOptions, str]:
    """Choose a retained local engine for source-language preview."""
    preview = BuildOptions.from_dict(options.to_dict())
    source = (source_language or "en").lower().strip()

    if source.startswith("en"):
        preview.tts_engine = TTSEngine.KOKORO
        preview.language = "en-us"
        preview.voice = "af_heart"
        return preview, "Kokoro · English source preview"

    if source.startswith("de"):
        if chatterbox_available:
            preview.tts_engine = TTSEngine.CHATTERBOX
            preview.language = "de"
            preview.voice = "chatterbox-multilingual-v3"
            return preview, "Chatterbox V3 · German source preview"
        preview.tts_engine = TTSEngine.QWEN3
        preview.language = "de"
        return preview, "Qwen3-TTS · German source preview"

    preview.tts_engine = TTSEngine.QWEN3
    preview.language = source
    return preview, f"Qwen3-TTS · {source} source preview"
