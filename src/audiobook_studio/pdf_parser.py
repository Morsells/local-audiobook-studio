from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import re
from pathlib import Path

import pymupdf

from .models import PageData, TextBlock


def get_page_count(pdf_path: str | Path) -> int:
    with pymupdf.open(pdf_path) as doc:
        return len(doc)


def _join_line_spans(line: dict) -> tuple[str, float, bool]:
    spans = line.get("spans", [])
    text = "".join(span.get("text", "") for span in spans)
    max_font = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
    bold = any("bold" in str(span.get("font", "")).lower() for span in spans)
    return text, max_font, bold


def _dict_to_blocks(page_number: int, page_height: float, page_dict: dict) -> list[TextBlock]:
    blocks: list[TextBlock] = []

    for source_index, block in enumerate(page_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue

        lines: list[str] = []
        max_font = 0.0
        bold = False

        for line in block.get("lines", []):
            line_text, font_size, line_bold = _join_line_spans(line)
            if line_text.strip():
                lines.append(line_text.rstrip())
            max_font = max(max_font, font_size)
            bold = bold or line_bold

        text = "\n".join(lines).strip()
        if not text:
            continue

        bbox = tuple(float(x) for x in block.get("bbox", (0, 0, 0, 0)))

        blocks.append(
            TextBlock(
                page_number=page_number,
                text=text,
                bbox=bbox,
                page_height=page_height,
                font_size=max_font,
                bold=bold,
                source_index=source_index,
            )
        )

    return blocks


def _extract_page(page, page_number: int, ocr_enabled: bool, ocr_language: str) -> PageData:
    page_dict = page.get_text("dict", sort=True)
    blocks = _dict_to_blocks(page_number, float(page.rect.height), page_dict)
    char_count = sum(len(block.text.strip()) for block in blocks)

    if char_count < 30 and ocr_enabled:
        try:
            textpage = page.get_textpage_ocr(
                language=ocr_language,
                dpi=250,
                full=True,
            )
            ocr_dict = page.get_text("dict", textpage=textpage, sort=True)
            blocks = _dict_to_blocks(page_number, float(page.rect.height), ocr_dict)
            return PageData(page_number=page_number, blocks=blocks, used_ocr=True)
        except Exception as exc:
            raise RuntimeError(
                "OCR was requested but failed. Install Tesseract and make sure "
                "PyMuPDF can find the tessdata directory."
            ) from exc

    return PageData(page_number=page_number, blocks=blocks, used_ocr=False)


def _repeat_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\b\d+\b", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def remove_repeated_headers_footers(pages: list[PageData]) -> list[PageData]:
    if len(pages) < 3:
        return pages

    presence: Counter[str] = Counter()

    for page in pages:
        seen: set[str] = set()
        for block in page.blocks:
            at_edge = block.y1_ratio <= 0.14 or block.y0_ratio >= 0.86
            if not at_edge:
                continue

            key = _repeat_key(block.text)
            if not key or len(key) > 180 or key in seen:
                continue

            seen.add(key)
            presence[key] += 1

    threshold = max(3, math.ceil(len(pages) * 0.45))
    repeated = {key for key, count in presence.items() if count >= threshold}

    cleaned: list[PageData] = []

    for page in pages:
        kept: list[TextBlock] = []
        for block in page.blocks:
            at_edge = block.y1_ratio <= 0.14 or block.y0_ratio >= 0.86
            if at_edge and _repeat_key(block.text) in repeated:
                continue
            kept.append(block)

        cleaned.append(replace(page, blocks=kept))

    return cleaned


def extract_pages(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
    *,
    ocr_enabled: bool = False,
    ocr_language: str = "eng",
) -> list[PageData]:
    with pymupdf.open(pdf_path) as doc:
        if start_page < 1 or end_page > len(doc) or start_page > end_page:
            raise ValueError(
                f"Invalid page range {start_page}-{end_page}. "
                f"The PDF has {len(doc)} pages."
            )

        pages = [
            _extract_page(
                doc[index - 1],
                index,
                ocr_enabled=ocr_enabled,
                ocr_language=ocr_language,
            )
            for index in range(start_page, end_page + 1)
        ]

    pages = remove_repeated_headers_footers(pages)

    if sum(len(page.text.strip()) for page in pages) < 30:
        raise RuntimeError(
            "Almost no selectable text was found. "
            "This is probably a scanned PDF. Enable OCR and install Tesseract."
        )

    return pages
