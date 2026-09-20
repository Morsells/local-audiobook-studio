from __future__ import annotations

import re
from collections import Counter

COMMON_CAPITALIZED = {
    "The", "A", "An", "This", "That", "These", "Those", "He", "She", "It", "They", "We", "I",
    "In", "On", "At", "For", "From", "To", "Of", "And", "But", "As", "If", "When", "While", "After",
    "Before", "Law", "Chapter", "Part", "Judgment", "Interpretation", "Reversal", "Power", "Authority",
}


def find_suspicious_words(text: str, limit: int = 80) -> list[tuple[str, int]]:
    """Heuristic review list for names/acronyms/foreign-looking words likely to need TTS tuning."""
    words = re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]{2,}\b", text)
    candidates: list[str] = []
    for word in words:
        stripped = word.strip(".'’-–")
        if len(stripped) < 3:
            continue
        acronym = stripped.isupper() and 2 <= len(stripped) <= 10
        capitalized = stripped[0].isupper() and stripped not in COMMON_CAPITALIZED
        accented = bool(re.search(r"[^A-Za-z'’-]", stripped))
        complex_shape = bool(re.search(r"[A-Z].*[A-Z]", stripped[1:]))
        if acronym or capitalized or accented or complex_shape:
            candidates.append(stripped)
    counts = Counter(candidates)
    return counts.most_common(limit)
