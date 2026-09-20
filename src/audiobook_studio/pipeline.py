from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from pathlib import Path
from typing import Callable, Any

from .audio import (
    audio_duration,
    convert_wav_to_mp3,
    export_m4b,
    make_tts_engine,
    merge_wavs_with_pauses,
)
from .chunker import chunk_text, word_count
from .config import MODEL_DIR
from .humanizer import apply_pronunciations, humanize
from .models import (
    AudioChunk,
    BuildOptions,
    HumanizerOptions,
    OutputMode,
    Section,
    TranslationEngine,
    TTSEngine,
)
from .project import ProjectStore
from .source_parser import analyze_source
from .translation import (
    ArgosTranslator,
    OllamaTranslator,
    OpusTranslator,
    TranslationRecord,
    stable_hash,
    translation_is_current,
    translation_model_dir,
)


ProgressCallback = Callable[[str, float], None]


def analyze_book(source_path: Path, options: BuildOptions):
    return analyze_source(
        source_path,
        options.start_page,
        options.end_page,
        ocr_enabled=options.ocr_enabled,
        ocr_language=options.ocr_language,
    )


def humanize_sections(
    sections: list[Section],
    options: BuildOptions,
    overrides: dict[str, str] | None = None,
    *,
    apply_pronunciation_rules: bool = True,
) -> list[tuple[Section, str]]:
    result: list[tuple[Section, str]] = []
    humanizer_options = HumanizerOptions(mode=options.reading_mode)
    overrides = overrides or {}

    pronunciations = options.pronunciations if apply_pronunciation_rules else {}

    for section in sections:
        if section.id in overrides and overrides[section.id].strip():
            text = overrides[section.id].strip()
            if apply_pronunciation_rules:
                text = apply_pronunciations(text, pronunciations)
            result.append((section, text))
            continue

        transformed = humanize(
            section.raw_text,
            humanizer_options,
            pronunciations=pronunciations,
            replacement_rules=options.replacement_rules,
            skip_strings=options.skip_strings,
        )
        if transformed.text.strip():
            result.append((section, transformed.text))
        if transformed.stopped_at_bibliography:
            break

    return result


def _translation_model_key(options: BuildOptions) -> str:
    if options.translation_engine == TranslationEngine.OPUS:
        return options.translation_model or f"opus-mt-{options.translation_source_language}-{options.translation_target_language}"
    if options.translation_engine == TranslationEngine.OLLAMA:
        return f"{options.translation_ollama_model}|address={options.translation_german_address_style}|literary=v3.6.9"
    return f"argos-{options.translation_source_language}-{options.translation_target_language}"


def _make_translator(options: BuildOptions):
    source = options.translation_source_language
    target = options.translation_target_language

    if options.translation_engine == TranslationEngine.OPUS:
        model_dir = translation_model_dir(MODEL_DIR, source, target)
        return OpusTranslator(
            model_dir,
            hardware_mode=options.hardware_mode,
            batch_size=options.translation_batch_size,
        )

    if options.translation_engine == TranslationEngine.OLLAMA:
        return OllamaTranslator(
            options.translation_ollama_model,
            german_address_style=options.translation_german_address_style,
        )

    return ArgosTranslator(source, target)


def translate_sections_batch(
    project: ProjectStore,
    sections: list[Section],
    options: BuildOptions,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, dict[str, Any]]:
    """Translate cleaned narration text and persist every section locally."""
    if not options.translation_enabled:
        raise RuntimeError("Enable translation first.")

    source_options = BuildOptions.from_dict(options.to_dict())
    source_options.pronunciations = {}

    prepared = humanize_sections(
        sections,
        source_options,
        project.load_overrides(),
        apply_pronunciation_rules=False,
    )
    prepared_by_id = {section.id: text for section, text in prepared}

    glossary = project.load_translation_glossary()
    translations = project.load_translations()
    memory = project.load_translation_memory()
    translator = None
    model_key = _translation_model_key(options)
    previous_context = ""

    total = max(1, len(sections))
    for index, section in enumerate(sections, start=1):
        source_text = prepared_by_id.get(section.id, "").strip()
        if not source_text:
            continue

        current = translations.get(section.id)
        is_current = translation_is_current(
            current,
            source_text,
            glossary,
            source_language=options.translation_source_language,
            target_language=options.translation_target_language,
            engine=options.translation_engine,
            model=model_key,
        )
        if is_current and not force:
            previous_context = source_text
            if progress:
                progress(
                    f"Translation {index}/{total}: cached — {section.title}",
                    index / total,
                )
            continue

        source_hash = stable_hash(source_text)
        memory_key = stable_hash(
            {
                "source_hash": source_hash,
                "source_language": options.translation_source_language,
                "target_language": options.translation_target_language,
                "engine": options.translation_engine.value,
                "model": model_key,
                "glossary_hash": stable_hash(glossary),
            }
        )

        translated = ""
        if not force and memory_key in memory:
            translated = str(memory[memory_key]).strip()

        if not translated:
            if translator is None:
                translator = _make_translator(options)

            if progress:
                progress(
                    f"Translation {index}/{total}: {section.title}",
                    (index - 1) / total,
                )

            if isinstance(translator, OllamaTranslator):
                translated = translator.translate(
                    source_text,
                    glossary,
                    source_language=options.translation_source_language,
                    target_language=options.translation_target_language,
                    previous_context=previous_context,
                )
            else:
                translated = translator.translate(source_text, glossary)

            memory[memory_key] = translated

        record = TranslationRecord(
            section_id=section.id,
            source_hash=source_hash,
            glossary_hash=stable_hash(glossary),
            source_language=options.translation_source_language,
            target_language=options.translation_target_language,
            engine=options.translation_engine.value,
            model=model_key,
            translated_text=translated,
            manual_text="",
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        project.save_translation(section.id, record.to_dict())
        translations[section.id] = record.to_dict()
        previous_context = source_text

    project.save_translation_memory(memory)
    project.save_translation_settings(
        {
            "enabled": options.translation_enabled,
            "source_language": options.translation_source_language,
            "target_language": options.translation_target_language,
            "engine": options.translation_engine.value,
            "model": model_key,
            "ollama_model": options.translation_ollama_model,
        }
    )

    if progress:
        progress("Translation complete", 1.0)

    return project.load_translations()


def translation_status(
    project: ProjectStore,
    sections: list[Section],
    options: BuildOptions,
) -> list[dict[str, Any]]:
    source_options = BuildOptions.from_dict(options.to_dict())
    source_options.pronunciations = {}
    prepared = humanize_sections(
        sections,
        source_options,
        project.load_overrides(),
        apply_pronunciation_rules=False,
    )
    prepared_by_id = {section.id: text for section, text in prepared}
    glossary = project.load_translation_glossary()
    model_key = _translation_model_key(options)
    records = project.load_translations()

    rows = []
    for section in sections:
        source_text = prepared_by_id.get(section.id, "")
        record = records.get(section.id)
        current = translation_is_current(
            record,
            source_text,
            glossary,
            source_language=options.translation_source_language,
            target_language=options.translation_target_language,
            engine=options.translation_engine,
            model=model_key,
        )
        rows.append(
            {
                "section_id": section.id,
                "title": section.title,
                "chapter": section.chapter_title or section.title,
                "current": current,
                "manual": bool((record or {}).get("manual_text", "").strip()),
                "characters": len((record or {}).get("translated_text", "")),
            }
        )
    return rows


def narration_sections(
    project_dir: Path,
    sections: list[Section],
    options: BuildOptions,
    overrides: dict[str, str] | None = None,
) -> list[tuple[Section, str]]:
    """Resolve the actual text that will be sent to TTS."""
    if not options.translation_enabled:
        return humanize_sections(sections, options, overrides)

    project = ProjectStore.from_existing_dir(project_dir)
    source_options = BuildOptions.from_dict(options.to_dict())
    source_options.pronunciations = {}

    prepared = humanize_sections(
        sections,
        source_options,
        overrides,
        apply_pronunciation_rules=False,
    )
    source_by_id = {section.id: text for section, text in prepared}
    glossary = project.load_translation_glossary()
    records = project.load_translations()
    model_key = _translation_model_key(options)

    result: list[tuple[Section, str]] = []
    missing: list[str] = []

    for section in sections:
        source_text = source_by_id.get(section.id, "").strip()
        if not source_text:
            continue

        record = records.get(section.id)
        if not translation_is_current(
            record,
            source_text,
            glossary,
            source_language=options.translation_source_language,
            target_language=options.translation_target_language,
            engine=options.translation_engine,
            model=model_key,
        ):
            missing.append(section.title)
            continue

        target_text = str(
            record.get("manual_text", "").strip()
            or record.get("translated_text", "").strip()
        )
        target_text = apply_pronunciations(target_text, options.pronunciations)
        if target_text:
            result.append((section, target_text))

    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise RuntimeError(
            "Translation is enabled, but current cached translations are missing for: "
            f"{preview}{extra}. Open the Translation tab and translate the selected sections first."
        )

    return result


def chunks_for_sections(
    prepared: list[tuple[Section, str]],
    options: BuildOptions,
) -> dict[str, list[AudioChunk]]:
    return {
        section.id: chunk_text(
            text,
            section,
            target_words=options.target_chunk_words,
            min_words=options.min_chunk_words,
            max_words=options.max_chunk_words,
        )
        for section, text in prepared
    }


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe[:110] or "audio"


def _group_outputs(
    sections: list[Section],
    chunk_map: dict[str, list[AudioChunk]],
    options: BuildOptions,
    output_basename: str | None = None,
) -> list[dict[str, Any]]:
    if output_basename:
        return [{
            "name": _safe_filename(output_basename),
            "title": output_basename,
            "chunks": [c for s in sections for c in chunk_map.get(s.id, [])],
            "section_ids": [s.id for s in sections],
        }]

    if options.output_mode == OutputMode.SINGLE:
        title = f"Content {options.start_page}-{options.end_page}"
        return [{
            "name": f"content_{options.start_page}-{options.end_page}",
            "title": title,
            "chunks": [c for s in sections for c in chunk_map.get(s.id, [])],
            "section_ids": [s.id for s in sections],
        }]

    if options.output_mode == OutputMode.SECTION:
        return [
            {
                "name": f"{idx:02d}_{_safe_filename(section.title)}",
                "title": section.title,
                "chunks": chunk_map.get(section.id, []),
                "section_ids": [section.id],
            }
            for idx, section in enumerate(sections, start=1)
            if chunk_map.get(section.id)
        ]

    groups: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for section in sections:
        chapter_id = section.chapter_id or section.id
        chapter_title = section.chapter_title or section.title
        if chapter_id not in index_by_id:
            index_by_id[chapter_id] = len(groups)
            groups.append({
                "name": f"{len(groups)+1:02d}_{_safe_filename(chapter_title)}",
                "title": chapter_title,
                "chunks": [],
                "section_ids": [],
            })
        group = groups[index_by_id[chapter_id]]
        group["chunks"].extend(chunk_map.get(section.id, []))
        group["section_ids"].append(section.id)
    return [g for g in groups if g["chunks"]]


def _pause_after_chunk(
    current: AudioChunk,
    next_chunk: AudioChunk | None,
    section_lookup: dict[str, Section],
    options: BuildOptions,
) -> int:
    if next_chunk is None:
        return 0
    if current.section_id == next_chunk.section_id:
        return options.chunk_pause_ms
    current_section = section_lookup[current.section_id]
    next_section = section_lookup[next_chunk.section_id]
    if (current_section.chapter_id or current_section.id) == (
        next_section.chapter_id or next_section.id
    ):
        return options.section_pause_ms
    return options.chapter_pause_ms


def _load_metrics(project_dir: Path) -> dict[str, Any]:
    path = project_dir / "metrics.json"
    if not path.exists():
        return {
            "samples": [],
            "avg_words_per_second": None,
            "avg_realtime_factor": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "samples": [],
            "avg_words_per_second": None,
            "avg_realtime_factor": None,
        }



def _performance_key(engine, options: BuildOptions) -> str:
    resolver = getattr(engine, "performance_key", None)
    if callable(resolver):
        try:
            return str(resolver(options))
        except Exception:
            pass
    return str(getattr(engine, "engine_id", "unknown"))


def _matching_performance_samples(
    project_dir: Path,
    *,
    performance_key: str,
    engine_id: str,
) -> list[dict[str, Any]]:
    samples = list(_load_metrics(project_dir).get("samples", []))

    exact = [
        sample
        for sample in samples
        if sample.get("performance_key") == performance_key
        and float(sample.get("elapsed", 0) or 0) > 0
        and float(sample.get("words", 0) or 0) > 0
    ]
    if exact:
        return exact

    engine_specific = [
        sample
        for sample in samples
        if sample.get("engine_id") == engine_id
        and not sample.get("performance_key")
        and float(sample.get("elapsed", 0) or 0) > 0
        and float(sample.get("words", 0) or 0) > 0
    ]
    return engine_specific


def _record_metric(project_dir: Path, metric: dict[str, Any]) -> None:
    path = project_dir / "metrics.json"
    data = _load_metrics(project_dir)
    samples = list(data.get("samples", []))[-49:]
    samples.append(metric)
    elapsed = sum(max(float(x.get("elapsed", 0)), 0.001) for x in samples)
    words = sum(float(x.get("words", 0)) for x in samples)
    duration = sum(float(x.get("duration", 0)) for x in samples)
    data = {
        "samples": samples,
        "avg_words_per_second": words / elapsed if elapsed else None,
        "avg_realtime_factor": elapsed / duration if duration else None,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")



def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _estimate_chunk_seconds(
    project_dir: Path,
    chunk: AudioChunk,
    engine_id: str,
    performance_key: str,
) -> float | None:
    usable = _matching_performance_samples(
        project_dir,
        performance_key=performance_key,
        engine_id=engine_id,
    )

    if not usable:
        return None

    total_words = sum(float(sample.get("words", 0)) for sample in usable)
    total_elapsed = sum(float(sample.get("elapsed", 0)) for sample in usable)
    if total_words <= 0 or total_elapsed <= 0:
        return None

    words_per_second = total_words / total_elapsed
    return max(1.0, float(chunk.word_count) / words_per_second)


def _synthesize_with_live_progress(
    engine,
    project_dir: Path,
    chunk: AudioChunk,
    destination: Path,
    options: BuildOptions,
    *,
    progress: ProgressCallback | None,
    base_fraction: float,
    span_fraction: float,
    label: str,
) -> dict[str, float]:
    """Run one blocking TTS call in a worker thread while emitting live elapsed/ETA updates."""
    engine_id = str(getattr(engine, "engine_id", "unknown"))
    performance_key = _performance_key(engine, options)
    estimated_seconds = _estimate_chunk_seconds(
        project_dir,
        chunk,
        engine_id,
        performance_key,
    )
    started = time.perf_counter()

    if progress:
        if estimated_seconds:
            progress(
                f"{label} — starting · estimated { _format_duration(estimated_seconds) }",
                base_fraction,
            )
        else:
            progress(
                f"{label} — starting · ETA calibrates after the first completed sample",
                base_fraction,
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            engine.synthesize_chunk,
            chunk,
            destination,
            options,
        )

        while not future.done():
            elapsed = time.perf_counter() - started

            if estimated_seconds:
                local_fraction = min(0.95, elapsed / max(estimated_seconds, 0.001))
                eta = max(0.0, estimated_seconds - elapsed)
                message = (
                    f"{label} — elapsed {_format_duration(elapsed)} · "
                    f"estimated {int(local_fraction * 100)}% · ETA ~{_format_duration(eta)}"
                )
            else:
                # No fake percentage on the first ever sample: show live elapsed time.
                local_fraction = 0.08
                message = (
                    f"{label} — elapsed {_format_duration(elapsed)} · "
                    "measuring your PC's generation speed…"
                )

            if progress:
                progress(
                    message,
                    min(0.995, base_fraction + span_fraction * local_fraction),
                )
            time.sleep(0.5)

        metric = future.result()

    metric = dict(metric)
    metric["engine_id"] = engine_id
    metric.setdefault("performance_key", performance_key)
    metric["voice"] = str(options.voice)
    metric["language"] = str(options.language)

    if progress:
        progress(
            f"{label} — synthesis finished in {_format_duration(metric.get('elapsed', 0.0))}",
            min(0.999, base_fraction + span_fraction),
        )

    return metric


def estimate_stats(
    sections: list[Section],
    options: BuildOptions,
    overrides: dict[str, str] | None,
    project_dir: Path,
) -> dict[str, Any]:
    prepared = narration_sections(project_dir, sections, options, overrides)
    words = sum(word_count(text) for _, text in prepared)
    audiobook_seconds = words / max(1.0, 155.0 * options.speed) * 60.0

    engine = make_tts_engine(options)
    engine_id = str(getattr(engine, "engine_id", "unknown"))
    performance_key = _performance_key(engine, options)

    generation_seconds = None
    realtime_factor = None
    times_realtime = None
    estimate_source = "uncalibrated"
    resolved_backend = ""

    if options.tts_engine == TTSEngine.CHATTERBOX:
        resolver = getattr(engine, "resolve_inference_backend", None)
        if callable(resolver):
            resolved_backend = str(resolver(options))

    samples = _matching_performance_samples(
        project_dir,
        performance_key=performance_key,
        engine_id=engine_id,
    )

    if samples:
        total_words = sum(float(sample.get("words", 0)) for sample in samples)
        total_elapsed = sum(float(sample.get("elapsed", 0)) for sample in samples)
        if total_words > 0 and total_elapsed > 0:
            words_per_second = total_words / total_elapsed
            generation_seconds = words / words_per_second
            if audiobook_seconds > 0:
                realtime_factor = generation_seconds / audiobook_seconds
                times_realtime = (
                    1.0 / realtime_factor
                    if realtime_factor > 0
                    else None
                )
            estimate_source = "recent project measurements"


    if generation_seconds is None:
        metrics = _load_metrics(project_dir)
        wps = metrics.get("avg_words_per_second")
        if wps:
            generation_seconds = words / float(wps)
            if audiobook_seconds > 0:
                realtime_factor = generation_seconds / audiobook_seconds
                times_realtime = (
                    1.0 / realtime_factor
                    if realtime_factor > 0
                    else None
                )
            estimate_source = "legacy project measurements"

    return {
        "words": words,
        "audiobook_seconds": audiobook_seconds,
        "generation_seconds": generation_seconds,
        "realtime_factor": realtime_factor,
        "times_realtime": times_realtime,
        "estimate_source": estimate_source,
        "performance_key": performance_key,
        "inference_backend": resolved_backend,
    }


def _cache_path(engine, project_dir: Path, chunk: AudioChunk, options: BuildOptions) -> Path:
    signature = engine.cache_signature(chunk, options)
    return project_dir / "audio_chunks" / f"{signature}.wav"


def invalidate_section_cache(
    project_dir: Path,
    section: Section,
    options: BuildOptions,
    cleaned_text: str,
) -> int:
    engine = make_tts_engine(options)
    chunks = chunk_text(
        cleaned_text,
        section,
        target_words=options.target_chunk_words,
        min_words=options.min_chunk_words,
        max_words=options.max_chunk_words,
    )
    removed = 0
    for chunk in chunks:
        path = _cache_path(engine, project_dir, chunk, options)
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def generate_test_audio(
    project_dir: Path,
    section: Section,
    cleaned_text: str,
    options: BuildOptions,
    *,
    voice: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    test_options = BuildOptions.from_dict(options.to_dict())
    if voice:
        test_options.voice = voice

    if progress:
        progress("Preparing preview text", 0.05)

    chunks = chunk_text(
        cleaned_text,
        section,
        target_words=min(options.target_chunk_words, 125),
        min_words=min(options.min_chunk_words, 55),
        max_words=min(options.max_chunk_words, 145),
    )
    if not chunks:
        raise RuntimeError("The selected preview has no narratable text.")

    chunk = chunks[0]
    engine = make_tts_engine(test_options)
    signature = engine.cache_signature(chunk, test_options)
    destination = project_dir / "previews" / f"preview-{signature}.wav"

    if destination.exists():
        if progress:
            progress("Preview already cached — ready", 1.0)
        return destination

    metric = _synthesize_with_live_progress(
        engine,
        project_dir,
        chunk,
        destination,
        test_options,
        progress=progress,
        base_fraction=0.10,
        span_fraction=0.85,
        label=f"Synthesizing preview ({chunk.word_count} words)",
    )
    _record_metric(project_dir, metric)

    if progress:
        progress("Preview audio ready", 1.0)

    return destination


def generate_pronunciation_test(
    project_dir: Path,
    source: str,
    spoken_as: str,
    options: BuildOptions,
) -> tuple[Path, Path]:
    dummy = Section("pron-test", "Pronunciation test", 1, 1, [])
    return (
        generate_test_audio(project_dir, dummy, f"{source}.", options),
        generate_test_audio(project_dir, dummy, f"{spoken_as}.", options),
    )


def inspect_chunks(
    project_dir: Path,
    sections: list[Section],
    options: BuildOptions,
    overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    prepared = narration_sections(project_dir, sections, options, overrides)
    chunk_map = chunks_for_sections(prepared, options)
    engine = make_tts_engine(options)

    rows: list[dict[str, Any]] = []
    for section in sections:
        for chunk in chunk_map.get(section.id, []):
            path = _cache_path(engine, project_dir, chunk, options)
            rows.append({
                "chunk_id": chunk.id,
                "section_id": section.id,
                "section": section.title,
                "chapter": section.chapter_title or section.title,
                "words": chunk.word_count,
                "pages": f"{chunk.source_start_page}-{chunk.source_end_page}",
                "cached": path.exists(),
                "audio_path": str(path) if path.exists() else "",
                "duration": audio_duration(path) if path.exists() else 0.0,
                "text": chunk.text,
            })
    return rows


def build_audio(
    project_dir: Path,
    selected_sections: list[Section],
    options: BuildOptions,
    *,
    overrides: dict[str, str] | None = None,
    progress: ProgressCallback | None = None,
    output_basename: str | None = None,
    force_section_ids: set[str] | None = None,
) -> list[Path]:
    prepared = narration_sections(
        project_dir,
        selected_sections,
        options,
        overrides,
    )
    chunk_map = chunks_for_sections(prepared, options)
    all_chunks = [
        chunk
        for section in selected_sections
        for chunk in chunk_map.get(section.id, [])
    ]
    if not all_chunks:
        raise RuntimeError("No narratable text remained after cleanup.")

    engine = make_tts_engine(options)
    chunk_dir = project_dir / "audio_chunks"
    output_dir = project_dir / "output"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    force_section_ids = force_section_ids or set()

    generated: dict[str, Path] = {}
    total = len(all_chunks)

    for index, chunk in enumerate(all_chunks, start=1):
        destination = _cache_path(engine, project_dir, chunk, options)
        force = chunk.section_id in force_section_ids
        base_fraction = (index - 1) / total * 0.88
        span_fraction = 0.88 / total

        if force or not destination.exists():
            metric = _synthesize_with_live_progress(
                engine,
                project_dir,
                chunk,
                destination,
                options,
                progress=progress,
                base_fraction=base_fraction,
                span_fraction=span_fraction,
                label=f"Chunk {index}/{total} ({chunk.word_count} words)",
            )
            _record_metric(project_dir, metric)
        elif progress:
            progress(
                f"Chunk {index}/{total} cached",
                min(0.88, base_fraction + span_fraction),
            )

        generated[chunk.id] = destination

    groups = _group_outputs(
        selected_sections,
        chunk_map,
        options,
        output_basename,
    )
    section_lookup = {section.id: section for section in selected_sections}
    outputs: list[Path] = []
    manifest_groups: list[dict[str, Any]] = []

    for group_index, group in enumerate(groups, start=1):
        chunks = group["chunks"]
        if progress:
            progress(
                f"Merging output {group_index}/{len(groups)}",
                0.88 + 0.10 * group_index / max(1, len(groups)),
            )

        items: list[tuple[Path, int]] = []
        for idx, chunk in enumerate(chunks):
            next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else None
            pause = _pause_after_chunk(
                chunk,
                next_chunk,
                section_lookup,
                options,
            )
            items.append((generated[chunk.id], pause))

        wav_path = output_dir / f"{group['name']}.wav"
        merge_wavs_with_pauses(items, wav_path)

        mp3_path = output_dir / f"{group['name']}.mp3"
        try:
            convert_wav_to_mp3(wav_path, mp3_path)
            outputs.append(mp3_path)
        except Exception:
            outputs.append(wav_path)

        manifest_groups.append({
            "name": group["name"],
            "title": group["title"],
            "wav": wav_path.name,
            "output": outputs[-1].name,
            "section_ids": group["section_ids"],
            "duration_seconds": audio_duration(wav_path),
        })

    manifest = {
        "options": options.to_dict(),
        "outputs": [p.name for p in outputs],
        "groups": manifest_groups,
        "sections": [s.to_dict() | {"blocks": []} for s in selected_sections],
    }
    (project_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    registry_path = project_dir / "output_registry.json"
    try:
        registry = (
            json.loads(registry_path.read_text(encoding="utf-8"))
            if registry_path.exists()
            else {"groups": []}
        )
    except Exception:
        registry = {"groups": []}

    by_name = {
        group.get("name"): group
        for group in registry.get("groups", [])
    }
    for group in manifest_groups:
        by_name[group.get("name")] = group

    registry["groups"] = sorted(
        by_name.values(),
        key=lambda group: str(group.get("name", "")),
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if progress:
        progress("Done", 1.0)

    return outputs


def export_project_m4b(
    project_dir: Path,
    *,
    title: str,
    author: str = "",
    cover_path: Path | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    registry_path = project_dir / "output_registry.json"
    manifest_path = project_dir / "build_manifest.json"
    source_path = registry_path if registry_path.exists() else manifest_path

    if not source_path.exists():
        raise RuntimeError(
            "Generate chapter audio before exporting an M4B."
        )

    manifest = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    chapter_wavs: list[tuple[str, Path]] = []
    chapter_durations: list[float] = []
    for group in manifest.get("groups", []):
        wav = project_dir / "output" / group.get("wav", "")
        final_output = project_dir / "output" / group.get("output", "")

        # Prefer the compact final chapter file (normally MP3). The V3.7 M4B
        # exporter decodes directly into AAC, so a giant intermediate WAV is
        # no longer needed.
        audio_path = final_output if final_output.exists() else wav
        if audio_path.exists():
            chapter_wavs.append(
                (group.get("title", audio_path.stem), audio_path)
            )
            try:
                chapter_durations.append(float(group.get("duration_seconds", 0.0) or 0.0))
            except Exception:
                chapter_durations.append(0.0)

    destination = (
        project_dir
        / "output"
        / f"{_safe_filename(title)}.m4b"
    )

    return export_m4b(
        chapter_wavs,
        destination,
        title=title,
        author=author,
        cover_path=cover_path,
        chapter_durations=chapter_durations,
        progress=progress,
    )
