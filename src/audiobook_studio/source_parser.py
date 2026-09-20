from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

from .models import PageData, Section, TextBlock
from .pdf_parser import extract_pages, get_page_count
from .semantic import detect_sections


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".epub"}


def source_kind(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".pdf": "PDF",
        ".txt": "Text",
        ".md": "Markdown",
        ".epub": "EPUB",
    }.get(suffix, suffix.lstrip(".").upper() or "Unknown")


def _text_block(index: int, text: str, *, font_size: float = 11.0, bold: bool = False) -> TextBlock:
    return TextBlock(
        page_number=index,
        text=text.strip(),
        bbox=(50.0, 80.0, 550.0, 850.0),
        page_height=900.0,
        font_size=font_size,
        bold=bold,
        source_index=0,
    )


def _assign_hierarchy(sections: list[Section]) -> list[Section]:
    current_chapter_id: str | None = None
    current_chapter_title: str | None = None
    parent_by_level: dict[int, str] = {}

    for section in sections:
        if section.level == 1:
            current_chapter_id = section.id
            current_chapter_title = section.title
            parent_by_level = {1: section.id}
        elif current_chapter_id is None:
            current_chapter_id = section.id
            current_chapter_title = section.title
            parent_by_level = {section.level: section.id}

        section.chapter_id = current_chapter_id
        section.chapter_title = current_chapter_title or section.title
        parent_level = max((lvl for lvl in parent_by_level if lvl < section.level), default=None)
        section.parent_id = parent_by_level.get(parent_level) if parent_level else None
        parent_by_level[section.level] = section.id
        for lvl in list(parent_by_level):
            if lvl > section.level:
                parent_by_level.pop(lvl, None)

    return sections


def _markdown_sections(text: str) -> list[Section]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    entries: list[tuple[int, str, list[str]]] = []
    current_level = 1
    current_title = "Opening"
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body or not entries:
            entries.append((current_level, current_title, current_lines[:]))
        current_lines = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current_lines or current_title != "Opening":
                flush()
            current_level = min(len(match.group(1)), 4)
            current_title = match.group(2).strip()
        else:
            current_lines.append(line)
    flush()

    sections: list[Section] = []
    for index, (level, title, body_lines) in enumerate(entries, start=1):
        body = "\n".join(body_lines).strip()
        combined = f"{title}\n\n{body}".strip()
        sections.append(
            Section(
                id=f"section-{index:03d}",
                title=title,
                start_page=index,
                end_page=index,
                blocks=[_text_block(index, combined, font_size=14.0 if level == 1 else 12.0, bold=True)],
                level=level,
            )
        )
    return _assign_hierarchy(sections)


def _plain_text_sections(text: str) -> list[Section]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    heading_re = re.compile(
        r"^\s*(?:chapter|part|law)\s+(?:\d+|[ivxlcdm]+)\b.*$",
        flags=re.IGNORECASE,
    )
    candidates: list[tuple[str, list[str]]] = []
    title = "Opening"
    body: list[str] = []

    def flush():
        nonlocal body
        if body or not candidates:
            candidates.append((title, body[:]))
        body = []

    for line in lines:
        stripped = line.strip()
        all_caps_heading = (
            stripped
            and stripped.isupper()
            and 1 <= len(stripped.split()) <= 10
            and len(stripped) <= 100
        )
        if heading_re.match(stripped) or all_caps_heading:
            if body:
                flush()
            title = stripped.title() if all_caps_heading else stripped
        else:
            body.append(line)
    flush()

    # If no useful headings, split into manageable logical units.
    if len(candidates) <= 1:
        words = text.split()
        if len(words) > 3500:
            candidates = []
            for i in range(0, len(words), 3000):
                candidates.append((f"Part {i // 3000 + 1}", [" ".join(words[i:i + 3000])]))

    sections: list[Section] = []
    for index, (section_title, body_lines) in enumerate(candidates, start=1):
        combined = f"{section_title}\n\n{''.join(body_lines) if len(body_lines) == 1 else chr(10).join(body_lines)}".strip()
        sections.append(
            Section(
                id=f"section-{index:03d}",
                title=section_title,
                start_page=index,
                end_page=index,
                blocks=[_text_block(index, combined, font_size=14.0, bold=True)],
                level=1,
            )
        )
    return _assign_hierarchy(sections)


def _epub_sections(path: Path) -> list[Section]:
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "EPUB support needs EbookLib and beautifulsoup4. Install requirements.txt again."
        ) from exc

    book = epub.read_epub(str(path))
    sections: list[Section] = []
    index = 0

    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3", "title"])
        title = heading.get_text(" ", strip=True) if heading else f"Section {index + 1}"
        paragraphs = [
            element.get_text(" ", strip=True)
            for element in soup.find_all(["p", "blockquote", "li"])
            if element.get_text(" ", strip=True)
        ]
        text = "\n\n".join(paragraphs).strip()
        if len(text) < 20:
            continue
        index += 1
        sections.append(
            Section(
                id=f"section-{index:03d}",
                title=title,
                start_page=index,
                end_page=index,
                blocks=[_text_block(index, f"{title}\n\n{text}", font_size=14.0, bold=True)],
                level=1,
            )
        )

    if not sections:
        raise RuntimeError("No readable EPUB text was found.")
    return _assign_hierarchy(sections)


def load_nonpdf_sections(path: str | Path) -> list[Section]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _epub_sections(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".md":
        return _markdown_sections(text)
    return _plain_text_sections(text)


def get_source_count(path: str | Path) -> int:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return get_page_count(path)
    return len(load_nonpdf_sections(path))


def analyze_source(
    path: str | Path,
    start_unit: int,
    end_unit: int,
    *,
    ocr_enabled: bool = False,
    ocr_language: str = "eng",
) -> tuple[list[PageData], list[Section]]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        pages = extract_pages(
            path,
            start_unit,
            end_unit,
            ocr_enabled=ocr_enabled,
            ocr_language=ocr_language,
        )
        return pages, detect_sections(pages)

    all_sections = load_nonpdf_sections(path)
    if start_unit < 1 or end_unit > len(all_sections) or start_unit > end_unit:
        raise ValueError(
            f"Invalid content range {start_unit}-{end_unit}; source has {len(all_sections)} logical units."
        )
    selected = all_sections[start_unit - 1:end_unit]

    # Rebase IDs is intentionally avoided: persistent section IDs should remain stable.
    pages = [
        PageData(
            page_number=section.start_page,
            blocks=section.blocks,
            used_ocr=False,
        )
        for section in selected
    ]
    return pages, selected
