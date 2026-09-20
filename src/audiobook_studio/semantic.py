from __future__ import annotations

import re
import statistics

from .models import PageData, Section, TextBlock


NUMBERED_HEADING = re.compile(
    r"^\s*(?:chapter\s+)?(?P<num>\d+(?:\.\d+){0,3})[\s.:\-]+(?P<title>.+?)\s*$",
    flags=re.IGNORECASE,
)
CHAPTER_HEADING = re.compile(
    r"^\s*(?:chapter|part)\s+(?P<num>[ivxlcdm]+|\d+)\b(?:\s*[:.\-]\s*|\s+)(?P<title>.+)?$",
    flags=re.IGNORECASE,
)
LAW_HEADING = re.compile(r"^\s*LAW\s+(?P<num>\d+)\s*$", flags=re.IGNORECASE)

KNOWN_SUBHEADINGS = {
    "judgment",
    "transgression of the law",
    "observance of the law",
    "interpretation",
    "keys to power",
    "reversal",
    "image",
    "authority",
}


def _median_body_font(pages: list[PageData]) -> float:
    sizes = [
        block.font_size
        for page in pages
        for block in page.blocks
        if block.font_size > 0 and len(block.text.split()) >= 4
    ]
    return float(statistics.median(sizes)) if sizes else 10.0


def clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n-–—:.")


def heading_level(text: str) -> int:
    cleaned = clean_heading(text)
    if LAW_HEADING.match(cleaned):
        return 1
    match = NUMBERED_HEADING.match(cleaned)
    if match:
        return min(match.group("num").count(".") + 1, 4)
    if CHAPTER_HEADING.match(cleaned):
        return 1
    if cleaned.lower() in KNOWN_SUBHEADINGS:
        return 3
    return 2


def is_likely_heading(block: TextBlock, body_font: float) -> bool:
    text = clean_heading(block.text)
    if not text or len(text) > 140:
        return False
    words = text.split()
    if len(words) > 16:
        return False

    if LAW_HEADING.match(text) or NUMBERED_HEADING.match(text) or CHAPTER_HEADING.match(text):
        return True
    if text.lower() in KNOWN_SUBHEADINGS:
        return True

    font_signal = block.font_size >= max(body_font * 1.22, body_font + 1.5)
    bold_signal = block.bold and block.font_size >= body_font * 1.02
    titleish = text.istitle() or (text.isupper() and len(words) <= 12)
    return (font_signal and len(words) <= 12) or (bold_signal and titleish)


def _combine_law_title(sections: list[Section]) -> None:
    """For books like The 48 Laws of Power, combine `LAW 1` with the following title."""
    for index, section in enumerate(sections):
        if not LAW_HEADING.match(clean_heading(section.title)):
            continue
        law_match = LAW_HEADING.match(clean_heading(section.title))
        law_label = f"LAW {law_match.group('num')}" if law_match else clean_heading(section.title)
        next_section = sections[index + 1] if index + 1 < len(sections) else None
        next_title = clean_heading(next_section.title) if next_section else ""
        # All-caps title directly after LAW n is normally the human chapter title.
        if next_section and next_section.level == 2 and next_title.isupper() and len(next_title.split()) <= 12:
            section.chapter_title = f"{law_label} — {next_title.title()}"
        else:
            section.chapter_title = law_label


def _assign_hierarchy(sections: list[Section]) -> None:
    current_chapter_id: str | None = None
    current_chapter_title: str | None = None
    parent_by_level: dict[int, str] = {}

    for section in sections:
        if section.level == 1:
            current_chapter_id = section.id
            current_chapter_title = section.chapter_title or section.title
            parent_by_level = {1: section.id}
        elif current_chapter_id is None:
            # Opening material before the first explicit chapter.
            current_chapter_id = section.id
            current_chapter_title = section.title
            parent_by_level = {section.level: section.id}

        section.chapter_id = current_chapter_id
        section.chapter_title = section.chapter_title or current_chapter_title

        parent_level = max((lvl for lvl in parent_by_level if lvl < section.level), default=None)
        section.parent_id = parent_by_level.get(parent_level) if parent_level else None
        parent_by_level[section.level] = section.id
        for lvl in list(parent_by_level):
            if lvl > section.level:
                parent_by_level.pop(lvl, None)


def detect_sections(pages: list[PageData]) -> list[Section]:
    if not pages:
        return []

    body_font = _median_body_font(pages)
    all_blocks = [block for page in pages for block in page.blocks]
    headings = [
        (index, block)
        for index, block in enumerate(all_blocks)
        if is_likely_heading(block, body_font)
    ]

    if not headings:
        only = Section(
            id="selection",
            title=f"Pages {pages[0].page_number}-{pages[-1].page_number}",
            start_page=pages[0].page_number,
            end_page=pages[-1].page_number,
            blocks=all_blocks,
            level=1,
        )
        only.chapter_id = only.id
        only.chapter_title = only.title
        return [only]

    sections: list[Section] = []
    first_heading_idx = headings[0][0]
    if first_heading_idx > 0:
        opening = all_blocks[:first_heading_idx]
        if sum(len(b.text) for b in opening) > 80:
            sections.append(
                Section(
                    id="opening-pages",
                    title="Opening pages",
                    start_page=opening[0].page_number,
                    end_page=opening[-1].page_number,
                    blocks=opening,
                    level=1,
                )
            )

    for h_number, (block_index, heading_block) in enumerate(headings, start=1):
        next_index = headings[h_number][0] if h_number < len(headings) else len(all_blocks)
        section_blocks = all_blocks[block_index:next_index]
        title = clean_heading(heading_block.text) or f"Section {h_number}"
        sections.append(
            Section(
                id=f"section-{h_number:03d}",
                title=title,
                start_page=section_blocks[0].page_number,
                end_page=section_blocks[-1].page_number,
                blocks=section_blocks,
                level=heading_level(title),
            )
        )

    _combine_law_title(sections)
    _assign_hierarchy(sections)
    return sections


def chapter_groups(sections: list[Section]) -> list[tuple[str, str, list[Section]]]:
    """Return (chapter_id, display_title, sections) preserving document order."""
    groups: list[tuple[str, str, list[Section]]] = []
    seen: dict[str, int] = {}
    for section in sections:
        chapter_id = section.chapter_id or section.id
        title = section.chapter_title or section.title
        if chapter_id in seen:
            groups[seen[chapter_id]][2].append(section)
        else:
            seen[chapter_id] = len(groups)
            groups.append((chapter_id, title, [section]))
    return groups
