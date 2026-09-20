from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ReadingMode(str, Enum):
    EXACT = "Exact"
    NATURAL = "Natural"
    STUDY = "Study"


class OutputMode(str, Enum):
    SINGLE = "One file for selected content"
    CHAPTER = "One file per chapter"
    SECTION = "One file per detected section"


class TTSEngine(str, Enum):
    CHATTERBOX = "Chatterbox Multilingual V3 — natural local multilingual"
    KOKORO = "Kokoro — lightweight local English"
    QWEN3 = "Qwen3-TTS — local multilingual"


class TranslationEngine(str, Enum):
    OPUS = "OPUS-MT — local fast"
    OLLAMA = "Ollama — local literary"
    ARGOS = "Argos Translate — local lightweight"


@dataclass
class TextBlock:
    page_number: int
    text: str
    bbox: tuple[float, float, float, float]
    page_height: float
    font_size: float = 0.0
    bold: bool = False
    source_index: int = 0

    @property
    def y0_ratio(self) -> float:
        return self.bbox[1] / max(self.page_height, 1.0)

    @property
    def y1_ratio(self) -> float:
        return self.bbox[3] / max(self.page_height, 1.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBlock":
        return cls(
            page_number=int(data["page_number"]),
            text=str(data["text"]),
            bbox=tuple(float(x) for x in data["bbox"]),
            page_height=float(data["page_height"]),
            font_size=float(data.get("font_size", 0.0)),
            bold=bool(data.get("bold", False)),
            source_index=int(data.get("source_index", 0)),
        )


@dataclass
class PageData:
    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    used_ocr: bool = False

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "blocks": [b.to_dict() for b in self.blocks],
            "used_ocr": self.used_ocr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageData":
        return cls(
            page_number=int(data["page_number"]),
            blocks=[TextBlock.from_dict(x) for x in data.get("blocks", [])],
            used_ocr=bool(data.get("used_ocr", False)),
        )


@dataclass
class Section:
    id: str
    title: str
    start_page: int
    end_page: int
    blocks: list[TextBlock] = field(default_factory=list)
    level: int = 1
    chapter_id: str | None = None
    chapter_title: str | None = None
    parent_id: str | None = None

    @property
    def raw_text(self) -> str:
        return "\n\n".join(block.text.strip() for block in self.blocks if block.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "blocks": [b.to_dict() for b in self.blocks],
            "level": self.level,
            "chapter_id": self.chapter_id,
            "chapter_title": self.chapter_title,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            start_page=int(data["start_page"]),
            end_page=int(data["end_page"]),
            blocks=[TextBlock.from_dict(x) for x in data.get("blocks", [])],
            level=int(data.get("level", 1)),
            chapter_id=data.get("chapter_id"),
            chapter_title=data.get("chapter_title"),
            parent_id=data.get("parent_id"),
        )


@dataclass
class AudioChunk:
    id: str
    section_id: str
    order: int
    text: str
    word_count: int
    source_start_page: int
    source_end_page: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HumanizerOptions:
    mode: ReadingMode = ReadingMode.NATURAL
    remove_citations: bool = True
    remove_urls: bool = True
    remove_cross_references: bool = True
    skip_bibliography: bool = True
    natural_parentheses: bool = True
    normalize_lists: bool = True
    normalize_math: bool = True
    skip_complex_math: bool = True


@dataclass
class BuildOptions:
    start_page: int
    end_page: int
    voice: str
    language: str
    speed: float = 1.0
    reading_mode: ReadingMode = ReadingMode.NATURAL
    output_mode: OutputMode = OutputMode.CHAPTER
    ocr_enabled: bool = False
    ocr_language: str = "eng"
    target_chunk_words: int = 155
    min_chunk_words: int = 80
    max_chunk_words: int = 195
    chunk_pause_ms: int = 250
    section_pause_ms: int = 700
    chapter_pause_ms: int = 1400
    pronunciations: dict[str, str] = field(default_factory=dict)
    replacement_rules: list[dict[str, Any]] = field(default_factory=list)
    skip_strings: list[str] = field(default_factory=list)

    # V3: local translation.
    translation_enabled: bool = False
    translation_engine: TranslationEngine = TranslationEngine.OPUS
    translation_source_language: str = "en"
    translation_target_language: str = "de"
    translation_model: str = ""
    translation_ollama_model: str = ""
    translation_german_address_style: str = "du"  # du | Sie
    translation_require_cached: bool = True
    translation_batch_size: int = 0  # 0 = automatic from hardware/VRAM

    # V3.5: automatic hardware acceleration.
    hardware_mode: str = "auto"  # auto | gpu | cpu

    # V3: pluggable local TTS.
    tts_engine: TTSEngine = TTSEngine.CHATTERBOX
    qwen_server_url: str = "http://127.0.0.1:7867"
    qwen_mode: str = "custom_voice"
    qwen_speaker: str = "Ryan"
    qwen_instruct: str = "Calm, natural, authoritative audiobook narration. Clear pronunciation and moderate pacing."
    qwen_model_label: str = "Qwen3-TTS local"

    # Chatterbox Multilingual V3 local sidecar.
    chatterbox_server_url: str = "http://127.0.0.1:7869"
    chatterbox_reference_path: str = ""
    # V3.7.5: selectable Chatterbox inference path.
    # auto -> CUDA Graph when this sidecar/GPU supports it, otherwise official.
    chatterbox_inference_backend: str = "auto"  # auto | cuda_graph | official
    chatterbox_exaggeration: float = 0.35
    chatterbox_cfg_weight: float = 0.30
    chatterbox_temperature: float = 0.55
    chatterbox_repetition_penalty: float = 1.15
    chatterbox_audiobook_pacing: bool = True
    chatterbox_max_phrase_words: int = 40
    chatterbox_comma_pause_ms: int = 100
    chatterbox_semicolon_pause_ms: int = 230
    chatterbox_sentence_pause_ms: int = 430

    tts_batch_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reading_mode"] = self.reading_mode.value
        data["output_mode"] = self.output_mode.value
        data["tts_engine"] = self.tts_engine.value
        data["translation_engine"] = self.translation_engine.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildOptions":
        copied = dict(data)
        copied["reading_mode"] = ReadingMode(copied.get("reading_mode", ReadingMode.NATURAL.value))
        copied["output_mode"] = OutputMode(copied.get("output_mode", OutputMode.CHAPTER.value))
        raw_tts = copied.get("tts_engine", TTSEngine.CHATTERBOX.value)
        # Unknown/removed historical engine values migrate to Chatterbox.
        try:
            copied["tts_engine"] = TTSEngine(raw_tts)
        except (TypeError, ValueError):
            copied["tts_engine"] = TTSEngine.CHATTERBOX
        copied["translation_engine"] = TranslationEngine(
            copied.get("translation_engine", TranslationEngine.OPUS.value)
        )
        allowed = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in copied.items() if key in allowed}
        return cls(**filtered)
