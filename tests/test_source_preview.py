from src.audiobook_studio.models import BuildOptions, TTSEngine
from src.audiobook_studio.preview import resolve_source_preview_options


def test_english_source_preview_uses_kokoro():
    options = BuildOptions(start_page=1,end_page=2,voice="chatterbox-multilingual-v3",language="de",tts_engine=TTSEngine.CHATTERBOX)
    preview, label = resolve_source_preview_options(options, "en", chatterbox_available=True)
    assert preview.tts_engine == TTSEngine.KOKORO
    assert preview.language == "en-us"
    assert "English source preview" in label


def test_german_source_preview_uses_chatterbox_when_available():
    options = BuildOptions(start_page=1,end_page=2,voice="af_heart",language="en-us",tts_engine=TTSEngine.KOKORO)
    preview, label = resolve_source_preview_options(options, "de", chatterbox_available=True)
    assert preview.tts_engine == TTSEngine.CHATTERBOX
    assert preview.language == "de"
    assert "Chatterbox" in label


def test_german_source_preview_falls_back_to_qwen():
    options = BuildOptions(start_page=1,end_page=2,voice="af_heart",language="en-us",tts_engine=TTSEngine.KOKORO)
    preview, label = resolve_source_preview_options(options, "de", chatterbox_available=False)
    assert preview.tts_engine == TTSEngine.QWEN3
    assert preview.language == "de"
    assert "Qwen3" in label
