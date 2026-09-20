from src.audiobook_studio.models import BuildOptions, TTSEngine
from src.audiobook_studio.preview import resolve_source_preview_options


def test_v36_chatterbox_options_roundtrip():
    options = BuildOptions(
        start_page=1,
        end_page=2,
        voice="chatterbox-multilingual-v3",
        language="de",
        tts_engine=TTSEngine.CHATTERBOX,
        chatterbox_server_url="http://127.0.0.1:7869",
        chatterbox_reference_path=r"C:\local\voice.wav",
        chatterbox_exaggeration=0.45,
        chatterbox_cfg_weight=0.4,
        chatterbox_temperature=0.75,
        chatterbox_repetition_penalty=1.25,
    )

    restored = BuildOptions.from_dict(options.to_dict())

    assert restored.tts_engine == TTSEngine.CHATTERBOX
    assert restored.language == "de"
    assert restored.chatterbox_server_url == "http://127.0.0.1:7869"
    assert restored.chatterbox_reference_path == r"C:\local\voice.wav"
    assert restored.chatterbox_exaggeration == 0.45
    assert restored.chatterbox_cfg_weight == 0.4
    assert restored.chatterbox_temperature == 0.75
    assert restored.chatterbox_repetition_penalty == 1.25


def test_german_source_preview_prefers_chatterbox_when_available():
    options = BuildOptions(
        start_page=1,
        end_page=2,
        voice="martin",
        language="de",
        tts_engine=TTSEngine.CHATTERBOX,
    )

    preview, label = resolve_source_preview_options(options, "de", chatterbox_available=True)

    assert preview.tts_engine == TTSEngine.CHATTERBOX
    assert preview.language == "de"
    assert "Chatterbox" in label


def test_german_source_preview_falls_back_without_chatterbox():
    options = BuildOptions(
        start_page=1, end_page=2, voice="af_heart", language="en-us",
        tts_engine=TTSEngine.KOKORO,
    )
    preview, _ = resolve_source_preview_options(
        options, "de", chatterbox_available=False
    )
    assert preview.tts_engine == TTSEngine.QWEN3
