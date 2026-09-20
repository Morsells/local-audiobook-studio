from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable


@dataclass
class CleanupResult:
    files_removed: int = 0
    bytes_removed: int = 0

    def add(self, other: "CleanupResult") -> None:
        self.files_removed += other.files_removed
        self.bytes_removed += other.bytes_removed


def format_bytes(size: int | float) -> str:
    value = float(max(0, size))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += file_size(item)
    return total


def project_storage_report(project_dir: Path) -> dict[str, int]:
    project_dir = Path(project_dir)
    output_dir = project_dir / "output"

    source_bytes = sum(
        file_size(path)
        for path in project_dir.glob("source.*")
        if path.is_file()
    )
    chunks = directory_size(project_dir / "audio_chunks")
    previews = directory_size(project_dir / "previews")

    output_wav = sum(file_size(path) for path in output_dir.glob("*.wav"))
    output_mp3 = sum(file_size(path) for path in output_dir.glob("*.mp3"))
    output_m4b = sum(file_size(path) for path in output_dir.glob("*.m4b"))
    other_output = max(
        0,
        directory_size(output_dir) - output_wav - output_mp3 - output_m4b,
    )

    protected_names = {
        "project.json",
        "settings.json",
        "analysis.json",
        "pronunciation.json",
        "replacement_rules.json",
        "skip_strings.json",
        "narration_overrides.json",
        "translation_settings.json",
        "translation_glossary.json",
        "translation_memory.json",
        "translations.json",
        "issues.json",
        "bookmarks.json",
        "listening_progress.json",
        "metrics.json",
        "queue.json",
        "queue_control.json",
        "worker_status.json",
        "build_manifest.json",
        "output_registry.json",
        "cover.png",
    }
    metadata = sum(
        file_size(project_dir / name)
        for name in protected_names
        if (project_dir / name).exists()
    )

    total = directory_size(project_dir)
    known = (
        source_bytes
        + chunks
        + previews
        + output_wav
        + output_mp3
        + output_m4b
        + other_output
        + metadata
    )

    return {
        "total": total,
        "source": source_bytes,
        "tts_chunk_cache": chunks,
        "preview_cache": previews,
        "intermediate_wav": output_wav,
        "final_mp3": output_mp3,
        "final_m4b": output_m4b,
        "other_output": other_output,
        "metadata_and_text": metadata,
        "other": max(0, total - known),
    }


def _delete_file(path: Path) -> CleanupResult:
    if not path.exists() or not path.is_file():
        return CleanupResult()
    size = file_size(path)
    try:
        path.unlink()
        return CleanupResult(files_removed=1, bytes_removed=size)
    except OSError:
        return CleanupResult()


def _clear_directory_files(path: Path) -> CleanupResult:
    result = CleanupResult()
    if not path.exists():
        return result

    for item in list(path.rglob("*")):
        if item.is_file():
            result.add(_delete_file(item))

    # Remove empty child directories but keep the root itself.
    for item in sorted(
        [p for p in path.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            item.rmdir()
        except OSError:
            pass

    return result


def clean_temp_files(project_dir: Path) -> CleanupResult:
    project_dir = Path(project_dir)
    result = CleanupResult()

    patterns = [
        "*.tmp",
        "*.tempo.wav",
        "*.m4b-source.wav",
        "*.part",
    ]

    seen: set[Path] = set()
    for pattern in patterns:
        for path in project_dir.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            result.add(_delete_file(path))

    return result


def clear_preview_cache(project_dir: Path) -> CleanupResult:
    return _clear_directory_files(Path(project_dir) / "previews")


def clear_tts_chunk_cache(project_dir: Path) -> CleanupResult:
    return _clear_directory_files(Path(project_dir) / "audio_chunks")


def prune_tts_chunk_cache(
    project_dir: Path,
    keep_paths: Iterable[str | Path],
) -> CleanupResult:
    chunk_dir = Path(project_dir) / "audio_chunks"
    keep = {
        Path(path).resolve()
        for path in keep_paths
        if str(path).strip()
    }

    result = CleanupResult()
    if not chunk_dir.exists():
        return result

    for path in chunk_dir.glob("*.wav"):
        if path.resolve() not in keep:
            result.add(_delete_file(path))

    return result


def clear_redundant_intermediate_wavs(project_dir: Path) -> CleanupResult:
    """Delete merged WAVs only when a same-stem final MP3 already exists.

    This is deliberately conservative. WAVs without a matching MP3 are kept.
    """
    output_dir = Path(project_dir) / "output"
    result = CleanupResult()

    if not output_dir.exists():
        return result

    for wav in output_dir.glob("*.wav"):
        if wav.name.endswith(".m4b-source.wav"):
            result.add(_delete_file(wav))
            continue

        matching_mp3 = wav.with_suffix(".mp3")
        if matching_mp3.exists():
            result.add(_delete_file(wav))

    return result


def safe_cleanup_project(project_dir: Path) -> CleanupResult:
    """Remove only obviously disposable data.

    Preserves:
    - source book
    - translations / glossary / edits
    - final MP3/M4B
    - TTS chunk cache

    Removes:
    - short preview audio
    - temporary files
    - merged WAVs that already have a matching MP3
    """
    result = CleanupResult()
    result.add(clear_preview_cache(project_dir))
    result.add(clean_temp_files(project_dir))
    result.add(clear_redundant_intermediate_wavs(project_dir))
    return result


def compact_finished_project(project_dir: Path) -> CleanupResult:
    """Aggressive but recoverable compaction.

    Keeps source, project data, translations, and final MP3/M4B files.
    Deletes regenerable TTS chunks, previews, temp files, and redundant WAVs.
    """
    result = safe_cleanup_project(project_dir)
    result.add(clear_tts_chunk_cache(project_dir))
    return result


def all_projects_report(data_dir: Path) -> list[dict]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []

    rows = []
    for project_dir in sorted(
        [path for path in data_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    ):
        report = project_storage_report(project_dir)
        rows.append(
            {
                "project": project_dir.name,
                "path": str(project_dir),
                **report,
            }
        )
    return rows


def global_safe_cleanup(data_dir: Path) -> CleanupResult:
    result = CleanupResult()
    for row in all_projects_report(data_dir):
        result.add(safe_cleanup_project(Path(row["path"])))
    return result
