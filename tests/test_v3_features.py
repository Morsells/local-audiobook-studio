from pathlib import Path

import pytest

from src.audiobook_studio.models import (
    BuildOptions,
    OutputMode,
    ReadingMode,
    TTSEngine,
    TranslationEngine,
)
from src.audiobook_studio.privacy import PrivacyViolation, assert_local_url
from src.audiobook_studio.project import ProjectStore
from src.audiobook_studio.source_parser import analyze_source, get_source_count
from src.audiobook_studio.translation import (
    stable_hash,
    translation_is_current,
)


def test_privacy_allows_loopback_and_blocks_public_hosts():
    assert_local_url("http://127.0.0.1:7867")
    assert_local_url("http://localhost:11434")
    with pytest.raises(PrivacyViolation):
        assert_local_url("https://example.com/api")


def test_markdown_source_becomes_logical_sections(tmp_path):
    path = tmp_path / "book.md"
    path.write_text(
        "# Chapter One\n\nIntro text.\n\n"
        "## Section A\n\nMore text.\n\n"
        "# Chapter Two\n\nFinal text.",
        encoding="utf-8",
    )
    assert get_source_count(path) >= 3
    pages, sections = analyze_source(path, 1, get_source_count(path))
    assert sections
    assert sections[0].chapter_id is not None
    assert any("Chapter Two" in section.title for section in sections)


def test_translation_record_detects_glossary_change():
    source = "Power is useful."
    glossary = {"Power": "Macht"}
    record = {
        "source_hash": stable_hash(source),
        "glossary_hash": stable_hash(glossary),
        "source_language": "en",
        "target_language": "de",
        "engine": TranslationEngine.OPUS.value,
        "model": "opus-mt-en-de",
        "translated_text": "Macht ist nützlich.",
    }

    assert translation_is_current(
        record,
        source,
        glossary,
        source_language="en",
        target_language="de",
        engine=TranslationEngine.OPUS,
        model="opus-mt-en-de",
    )
    assert not translation_is_current(
        record,
        source,
        {"Power": "Kraft"},
        source_language="en",
        target_language="de",
        engine=TranslationEngine.OPUS,
        model="opus-mt-en-de",
    )


def test_project_persists_translation_files(tmp_path, monkeypatch):
    import src.audiobook_studio.project as project_module

    monkeypatch.setattr(project_module, "DATA_DIR", tmp_path)
    store = ProjectStore(b"example book bytes", "book.txt")
    store.save_translation_glossary({"power": "Macht"})
    store.save_translation(
        "section-001",
        {
            "section_id": "section-001",
            "translated_text": "Hallo.",
            "manual_text": "",
        },
    )
    assert store.load_translation_glossary()["power"] == "Macht"
    assert store.get_translation("section-001")["translated_text"] == "Hallo."


def test_v3_options_roundtrip():
    options = BuildOptions(
        start_page=1,
        end_page=10,
        voice="af_heart",
        language="de",
        reading_mode=ReadingMode.NATURAL,
        output_mode=OutputMode.CHAPTER,
        translation_enabled=True,
        translation_engine=TranslationEngine.OPUS,
        translation_source_language="en",
        translation_target_language="de",
        tts_engine=TTSEngine.QWEN3,
        qwen_speaker="Ryan",
    )
    restored = BuildOptions.from_dict(options.to_dict())
    assert restored.translation_enabled is True
    assert restored.translation_engine == TranslationEngine.OPUS
    assert restored.tts_engine == TTSEngine.QWEN3
    assert restored.translation_target_language == "de"
