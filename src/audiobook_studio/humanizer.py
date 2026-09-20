from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import HumanizerOptions, ReadingMode


REFERENCE_HEADING = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|index)\s*$",
    flags=re.IGNORECASE,
)
INLINE_CITATIONS = [
    re.compile(r"\[(?:\d+\s*(?:[-,;]\s*\d+\s*)*)\]"),
    re.compile(r"\[(?:\d+\s*(?:,\s*)?)+\]"),
]
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    flags=re.IGNORECASE,
)
CROSS_REFERENCE = re.compile(
    r"\b(?:see|cf\.?)\s+(?:fig(?:ure)?|table|eq(?:uation)?|section|chapter)"
    r"\s*\.?\s*[A-Za-z0-9.\-]+",
    flags=re.IGNORECASE,
)
PAREN_CROSSREF = re.compile(
    r"\(\s*(?:see|cf\.?)\s+(?:fig(?:ure)?|table|eq(?:uation)?|section|chapter).*?\)",
    flags=re.IGNORECASE,
)
STANDALONE_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+\s*$", flags=re.IGNORECASE)
COMPLEX_MATH_HINTS = re.compile(r"[∑∏∫√≈≠≤≥∞∂∇∀∃⊂⊃∈∉]|\\(?:sum|frac|int|sqrt)\b")


@dataclass
class HumanizedText:
    text: str
    stopped_at_bibliography: bool = False


def _normalize_linebreaks(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _dehyphenate_linebreaks(text: str) -> str:
    return re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)


def _remove_page_number_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not STANDALONE_PAGE_NUMBER.match(line.strip()))


def _remove_citations(text: str) -> str:
    for pattern in INLINE_CITATIONS:
        text = pattern.sub("", text)
    return re.sub(
        r"\((?:[A-Z][A-Za-z'’-]+(?:\s+et\s+al\.)?(?:,\s*)?\s*(?:19|20)\d{2}[a-z]?(?:;\s*)?)+\)",
        "",
        text,
    )


def _naturalize_parentheses(text: str) -> str:
    text = PAREN_CROSSREF.sub("", text)
    return re.sub(r"\(([^()\n]{1,120})\)", lambda m: f", {m.group(1).strip()},", text)


def _normalize_bullets(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        nonlocal bullets
        if not bullets:
            return
        if len(bullets) == 1:
            result.append(bullets[0] + ".")
        elif len(bullets) == 2:
            result.append(f"{bullets[0]}, and {bullets[1]}.")
        else:
            result.append(", ".join(bullets[:-1]) + f", and {bullets[-1]}.")
        bullets = []

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(?:[-–—•*]|\d+[.)])\s+(.+)$", stripped)
        if match:
            bullets.append(match.group(1).rstrip(".;"))
        else:
            flush()
            result.append(line)
    flush()
    return "\n".join(result)


def _normalize_headings(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^\d+(?:\.\d+)+[\s.:\-]+(.+)$", stripped)
        if match and len(stripped.split()) <= 14:
            output.append(match.group(1).rstrip(".") + ".")
            continue
        match = re.match(r"^(chapter|part)\s+([ivxlcdm]+|\d+)\s*[:.\-]\s*(.+)$", stripped, flags=re.I)
        if match:
            output.append(f"{match.group(1).title()} {match.group(2)}. {match.group(3).rstrip('.')}.")
            continue
        # LAW 1 -> Law one-ish is already handled well by most voices; punctuation gives a pause.
        law = re.match(r"^LAW\s+(\d+)\s*$", stripped, flags=re.I)
        if law:
            output.append(f"Law {law.group(1)}.")
            continue
        output.append(line)
    return "\n".join(output)


def _normalize_common_math(text: str, skip_complex: bool) -> str:
    if skip_complex:
        cleaned: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            math_chars = len(re.findall(r"[=+\-*/^_{}<>∑∏∫√≈≠≤≥]", stripped))
            letters = len(re.findall(r"[A-Za-z]", stripped))
            likely_formula = bool(COMPLEX_MATH_HINTS.search(stripped)) or (
                math_chars >= 3 and letters < 25 and len(stripped) < 180
            )
            if likely_formula and len(stripped.split()) <= 20:
                cleaned.append("A mathematical expression is shown here.")
            else:
                cleaned.append(line)
        text = "\n".join(cleaned)

    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " plus ", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", text)
    text = re.sub(r"(?<=\d)\s*[×*]\s*(?=\d)", " times ", text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " divided by ", text)
    text = re.sub(r"(?<=\w)\s*=\s*(?=\w)", " equals ", text)
    text = re.sub(r"\bO\s*\(\s*n\s*(?:\^?\s*2|²)\s*\)", "O of n squared", text, flags=re.I)
    text = re.sub(r"\bO\s*\(\s*n\s+log\s+n\s*\)", "O of n log n", text, flags=re.I)
    return text


def _apply_skip_strings(text: str, skip_strings: list[str]) -> str:
    for value in sorted((x.strip() for x in skip_strings if x.strip()), key=len, reverse=True):
        text = re.sub(re.escape(value), "", text, flags=re.I)
    return text


def _apply_replacement_rules(text: str, rules: list[dict[str, Any]]) -> str:
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        source = str(rule.get("find", "")).strip()
        target = str(rule.get("replace", ""))
        if not source:
            continue
        if rule.get("regex", False):
            try:
                text = re.sub(source, target, text, flags=re.I)
            except re.error:
                continue
        else:
            text = re.sub(re.escape(source), target, text, flags=re.I)
    return text


def apply_pronunciations(text: str, pronunciations: dict[str, str]) -> str:
    for source in sorted(pronunciations, key=len, reverse=True):
        target = pronunciations[source].strip()
        if not source.strip() or not target:
            continue
        text = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, text, flags=re.I)
    return text


def _study_cleanup(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(figure|fig\.|table)\s+\d+(?:\.\d+)*\b", stripped, flags=re.I):
            continue
        if re.search(r"\bcopyright\b|©|downloaded from|all rights reserved", stripped, flags=re.I):
            continue
        kept.append(line)
    return "\n".join(kept)


def _whitespace_cleanup(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip(" ,;\n\t")


def humanize(
    raw_text: str,
    options: HumanizerOptions,
    pronunciations: dict[str, str] | None = None,
    replacement_rules: list[dict[str, Any]] | None = None,
    skip_strings: list[str] | None = None,
) -> HumanizedText:
    text = _normalize_linebreaks(raw_text)
    text = _dehyphenate_linebreaks(text)
    text = _remove_page_number_lines(text)
    text = _apply_skip_strings(text, skip_strings or [])
    text = _apply_replacement_rules(text, replacement_rules or [])

    if options.skip_bibliography:
        kept: list[str] = []
        for line in text.splitlines():
            if REFERENCE_HEADING.match(line.strip()):
                cleaned = _whitespace_cleanup("\n".join(kept))
                cleaned = apply_pronunciations(cleaned, pronunciations or {})
                return HumanizedText(cleaned, True)
            kept.append(line)
        text = "\n".join(kept)

    if options.mode == ReadingMode.EXACT:
        return HumanizedText(apply_pronunciations(_whitespace_cleanup(text), pronunciations or {}))

    text = _normalize_headings(text)
    if options.remove_citations:
        text = _remove_citations(text)
    if options.remove_urls:
        text = URL_PATTERN.sub("", text)
    if options.remove_cross_references:
        text = CROSS_REFERENCE.sub("", text)
        text = PAREN_CROSSREF.sub("", text)
    if options.natural_parentheses:
        text = _naturalize_parentheses(text)
    if options.normalize_lists:
        text = _normalize_bullets(text)
    if options.normalize_math:
        text = _normalize_common_math(text, options.skip_complex_math)
    if options.mode == ReadingMode.STUDY:
        text = _study_cleanup(text)

    text = _whitespace_cleanup(text)
    text = apply_pronunciations(text, pronunciations or {})
    return HumanizedText(text)
