from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from .config import DATA_DIR, DEFAULT_PRONUNCIATIONS
from .models import PageData, Section


def _atomic_json_write(path: Path, data: Any) -> None:
    """Atomically write JSON, tolerating short Windows sharing locks.

    Streamlit and the background queue worker access a few small state files
    concurrently. On Windows, a reader/AV/indexer can briefly hold a file
    without FILE_SHARE_DELETE, causing os.replace() to raise WinError 5/32.
    Those locks are normally transient, so retry rather than killing the worker.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

        last_error: OSError | None = None

        # ~5 seconds total worst-case wait. Typical Windows read-lock races
        # resolve within the first one or two attempts.
        for attempt in range(30):
            try:
                os.replace(temp_name, path)
                return
            except PermissionError as exc:
                last_error = exc
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32}:
                    raise
                last_error = exc

            time.sleep(min(0.02 * (attempt + 1), 0.25))

        raise PermissionError(
            f"Could not replace JSON state file after retries: {path}. "
            "Another process is holding the file open."
        ) from last_error

    finally:
        if os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


class ProjectStore:
    def __init__(self, source_bytes: bytes, original_name: str):
        self.pdf_hash = hashlib.sha256(source_bytes).hexdigest()[:20]
        safe_stem = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(original_name).stem
        ).strip("_") or "book"

        self.project_dir = DATA_DIR / f"{safe_stem}-{self.pdf_hash[:10]}"
        suffix = Path(original_name).suffix.lower() or ".bin"
        self.source_path = self.project_dir / f"source{suffix}"
        # Backwards compatibility for older code; V3 code should use source_path.
        self.source_pdf = self.source_path

        self.meta_file = self.project_dir / "project.json"
        self.settings_file = self.project_dir / "settings.json"
        self.pronunciations_file = self.project_dir / "pronunciation.json"
        self.replacements_file = self.project_dir / "replacement_rules.json"
        self.skip_file = self.project_dir / "skip_strings.json"
        self.overrides_file = self.project_dir / "narration_overrides.json"
        self.analysis_file = self.project_dir / "analysis.json"
        self.issues_file = self.project_dir / "issues.json"
        self.bookmarks_file = self.project_dir / "bookmarks.json"
        self.progress_file = self.project_dir / "listening_progress.json"
        self.metrics_file = self.project_dir / "metrics.json"
        self.queue_file = self.project_dir / "queue.json"
        self.queue_control_file = self.project_dir / "queue_control.json"
        self.worker_status_file = self.project_dir / "worker_status.json"
        self.cover_file = self.project_dir / "cover.png"

        # V3 translation state.
        self.translation_settings_file = self.project_dir / "translation_settings.json"
        self.translation_glossary_file = self.project_dir / "translation_glossary.json"
        self.translation_memory_file = self.project_dir / "translation_memory.json"
        self.translations_file = self.project_dir / "translations.json"

        self.chunk_dir = self.project_dir / "audio_chunks"
        self.preview_dir = self.project_dir / "previews"
        self.output_dir = self.project_dir / "output"

        for directory in (
            self.project_dir,
            self.chunk_dir,
            self.preview_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        # Migrate an older V1/V2 source.pdf in-place when reopening the same project.
        legacy_source = self.project_dir / "source.pdf"
        if not self.source_path.exists() and legacy_source.exists() and suffix == ".pdf":
            self.source_path = legacy_source
            self.source_pdf = legacy_source

        if not self.source_path.exists():
            self.source_path.write_bytes(source_bytes)

        if not self.meta_file.exists():
            self.save_json(
                self.meta_file,
                {
                    "name": original_name,
                    "title": Path(original_name).stem,
                    "author": "",
                    "pdf_hash": self.pdf_hash,
                    "source_suffix": suffix,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "version": 3,
                },
            )
        else:
            meta = self.load_meta()
            changed = False
            if int(meta.get("version", 1)) < 3:
                meta["version"] = 3
                changed = True
            if "source_suffix" not in meta:
                meta["source_suffix"] = suffix
                changed = True
            if changed:
                self.save_json(self.meta_file, meta)

        if not self.pronunciations_file.exists():
            self.save_json(self.pronunciations_file, DEFAULT_PRONUNCIATIONS)

        defaults = [
            (self.replacements_file, []),
            (self.skip_file, []),
            (self.overrides_file, {}),
            (self.issues_file, []),
            (self.bookmarks_file, []),
            (self.progress_file, {}),
            (
                self.metrics_file,
                {
                    "samples": [],
                    "avg_words_per_second": None,
                    "avg_realtime_factor": None,
                },
            ),
            (self.queue_file, {"jobs": []}),
            (
                self.queue_control_file,
                {"paused": True, "stop_after_current": False},
            ),
            (
                self.translation_settings_file,
                {
                    "enabled": False,
                    "source_language": "en",
                    "target_language": "de",
                    "engine": "OPUS-MT — local fast",
                    "model": "",
                },
            ),
            (self.translation_glossary_file, {}),
            (self.translation_memory_file, {}),
            (self.translations_file, {}),
        ]

        for file_path, default in defaults:
            if not file_path.exists():
                self.save_json(file_path, default)

    @staticmethod
    def load_json(path: Path, default: Any = None) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def save_json(path: Path, data: Any) -> None:
        _atomic_json_write(path, data)

    def load_meta(self) -> dict[str, Any]:
        return self.load_json(self.meta_file, {}) or {}

    def update_meta(self, **updates: Any) -> None:
        data = self.load_meta()
        data.update(updates)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_json(self.meta_file, data)

    def load_settings(self) -> dict[str, Any]:
        return self.load_json(self.settings_file, {}) or {}

    def save_settings(self, data: dict[str, Any]) -> None:
        self.save_json(self.settings_file, data)

    def load_pronunciations(self) -> dict[str, str]:
        return self.load_json(self.pronunciations_file, {}) or {}

    def save_pronunciations(self, data: dict[str, str]) -> None:
        self.save_json(self.pronunciations_file, data)

    def load_replacement_rules(self) -> list[dict[str, Any]]:
        return self.load_json(self.replacements_file, []) or []

    def save_replacement_rules(self, data: list[dict[str, Any]]) -> None:
        self.save_json(self.replacements_file, data)

    def load_skip_strings(self) -> list[str]:
        return self.load_json(self.skip_file, []) or []

    def save_skip_strings(self, data: list[str]) -> None:
        self.save_json(self.skip_file, data)

    def load_overrides(self) -> dict[str, str]:
        return self.load_json(self.overrides_file, {}) or {}

    def save_override(self, section_id: str, text: str) -> None:
        data = self.load_overrides()
        if text.strip():
            data[section_id] = text.strip()
        else:
            data.pop(section_id, None)
        self.save_json(self.overrides_file, data)

    def save_analysis(
        self,
        pages: list[PageData],
        sections: list[Section],
        signature: str,
    ) -> None:
        self.save_json(
            self.analysis_file,
            {
                "signature": signature,
                "pages": [p.to_dict() for p in pages],
                "sections": [s.to_dict() for s in sections],
            },
        )

    def load_analysis(self) -> tuple[str | None, list[PageData], list[Section]]:
        data = self.load_json(self.analysis_file, {}) or {}
        return (
            data.get("signature"),
            [PageData.from_dict(x) for x in data.get("pages", [])],
            [Section.from_dict(x) for x in data.get("sections", [])],
        )

    # --- Translation persistence -------------------------------------------------

    def load_translation_settings(self) -> dict[str, Any]:
        return self.load_json(self.translation_settings_file, {}) or {}

    def save_translation_settings(self, data: dict[str, Any]) -> None:
        self.save_json(self.translation_settings_file, data)

    def load_translation_glossary(self) -> dict[str, str]:
        return self.load_json(self.translation_glossary_file, {}) or {}

    def save_translation_glossary(self, data: dict[str, str]) -> None:
        self.save_json(self.translation_glossary_file, data)

    def load_translation_memory(self) -> dict[str, str]:
        return self.load_json(self.translation_memory_file, {}) or {}

    def save_translation_memory(self, data: dict[str, str]) -> None:
        self.save_json(self.translation_memory_file, data)

    def load_translations(self) -> dict[str, dict[str, Any]]:
        return self.load_json(self.translations_file, {}) or {}

    def get_translation(self, section_id: str) -> dict[str, Any] | None:
        return self.load_translations().get(section_id)

    def save_translation(self, section_id: str, record: dict[str, Any]) -> None:
        data = self.load_translations()
        data[section_id] = record
        self.save_json(self.translations_file, data)

    def save_translation_manual_text(self, section_id: str, text: str) -> None:
        data = self.load_translations()
        record = dict(data.get(section_id, {}))
        if not record:
            raise RuntimeError("Translate this section before adding a manual translation override.")
        record["manual_text"] = text.strip()
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        data[section_id] = record
        self.save_json(self.translations_file, data)

    def clear_translation(self, section_id: str) -> None:
        data = self.load_translations()
        data.pop(section_id, None)
        self.save_json(self.translations_file, data)

    # --- Existing V2 project tools ----------------------------------------------

    def add_issue(self, issue: dict[str, Any]) -> None:
        items = self.load_json(self.issues_file, []) or []
        issue = dict(issue)
        issue.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        items.append(issue)
        self.save_json(self.issues_file, items)

    def add_bookmark(self, output_name: str, seconds: float, label: str) -> None:
        items = self.load_json(self.bookmarks_file, []) or []
        items.append(
            {
                "output": output_name,
                "seconds": float(seconds),
                "label": label.strip() or "Bookmark",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.save_json(self.bookmarks_file, items)

    def set_listening_progress(self, output_name: str, seconds: float) -> None:
        data = self.load_json(self.progress_file, {}) or {}
        data[output_name] = float(seconds)
        self.save_json(self.progress_file, data)

    def render_cover(self, page_number: int = 1, force: bool = False) -> Path:
        if self.cover_file.exists() and not force:
            return self.cover_file
        if self.source_path.suffix.lower() != ".pdf":
            raise RuntimeError("Automatic cover rendering is currently available for PDF sources only.")
        with pymupdf.open(self.source_path) as doc:
            index = max(0, min(len(doc) - 1, page_number - 1))
            pix = doc[index].get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False,
            )
            pix.save(self.cover_file)
        return self.cover_file

    @classmethod
    def from_existing_dir(cls, project_dir: str | Path) -> "ProjectStore":
        project_dir = Path(project_dir)
        meta = cls.load_json(project_dir / "project.json", {}) or {}

        candidates = sorted(project_dir.glob("source.*"))
        source = candidates[0] if candidates else project_dir / "source.pdf"

        obj = cls.__new__(cls)
        obj.pdf_hash = str(meta.get("pdf_hash", ""))
        obj.project_dir = project_dir
        obj.source_path = source
        obj.source_pdf = source

        obj.meta_file = project_dir / "project.json"
        obj.settings_file = project_dir / "settings.json"
        obj.pronunciations_file = project_dir / "pronunciation.json"
        obj.replacements_file = project_dir / "replacement_rules.json"
        obj.skip_file = project_dir / "skip_strings.json"
        obj.overrides_file = project_dir / "narration_overrides.json"
        obj.analysis_file = project_dir / "analysis.json"
        obj.issues_file = project_dir / "issues.json"
        obj.bookmarks_file = project_dir / "bookmarks.json"
        obj.progress_file = project_dir / "listening_progress.json"
        obj.metrics_file = project_dir / "metrics.json"
        obj.queue_file = project_dir / "queue.json"
        obj.queue_control_file = project_dir / "queue_control.json"
        obj.worker_status_file = project_dir / "worker_status.json"
        obj.cover_file = project_dir / "cover.png"
        obj.translation_settings_file = project_dir / "translation_settings.json"
        obj.translation_glossary_file = project_dir / "translation_glossary.json"
        obj.translation_memory_file = project_dir / "translation_memory.json"
        obj.translations_file = project_dir / "translations.json"
        obj.chunk_dir = project_dir / "audio_chunks"
        obj.preview_dir = project_dir / "previews"
        obj.output_dir = project_dir / "output"

        # Ensure V3 files exist for migrated V1/V2 projects.
        for file_path, default in [
            (obj.translation_settings_file, {}),
            (obj.translation_glossary_file, {}),
            (obj.translation_memory_file, {}),
            (obj.translations_file, {}),
        ]:
            if not file_path.exists():
                cls.save_json(file_path, default)

        return obj
