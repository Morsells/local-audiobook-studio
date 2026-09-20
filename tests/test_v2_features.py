from pathlib import Path

from src.audiobook_studio.humanizer import humanize
from src.audiobook_studio.models import HumanizerOptions, ReadingMode, Section, TextBlock, PageData
from src.audiobook_studio.semantic import detect_sections, chapter_groups
from src.audiobook_studio.suspicious import find_suspicious_words


def block(page, text, size=12, bold=False, y=100):
    return TextBlock(page, text, (50, y, 500, y + 30), 800, size, bold, 0)


def test_law_hierarchy_groups_subsections_into_one_chapter():
    pages = [
        PageData(1, [
            block(1, "LAW 1", 20, True, 80),
            block(1, "NEVER OUTSHINE THE MASTER", 18, True, 120),
            block(1, "JUDGMENT", 15, True, 160),
            block(1, "Some body text with enough words to look like normal body text in the document.", 10, False, 200),
            block(1, "TRANSGRESSION OF THE LAW", 15, True, 300),
            block(1, "More body text with enough words to remain body text and not a title.", 10, False, 340),
        ])
    ]
    sections = detect_sections(pages)
    groups = chapter_groups(sections)
    assert groups
    law_group = next(g for g in groups if "LAW 1" in g[1])
    assert len(law_group[2]) >= 3
    assert "Never Outshine" in law_group[1]


def test_replacement_and_skip_rules_apply_before_tts():
    result = humanize(
        "HEADER TEXT\nSee Fig. 4. The value is useful.",
        HumanizerOptions(mode=ReadingMode.NATURAL),
        replacement_rules=[{"enabled": True, "find": "Fig.", "replace": "Figure", "regex": False}],
        skip_strings=["HEADER TEXT"],
    )
    assert "HEADER TEXT" not in result.text
    assert "Figure" in result.text or "value is useful" in result.text


def test_suspicious_names_are_found():
    found = dict(find_suspicious_words("Machiavelli met Richelieu. AES was discussed."))
    assert "Machiavelli" in found
    assert "Richelieu" in found
    assert "AES" in found
