from __future__ import annotations

import re

from .models import AudioChunk, Section


SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])"
)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_oversized_sentence(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()

    if len(words) <= max_words:
        return [sentence.strip()]

    # Prefer punctuation boundaries before hard word slicing.
    pieces = re.split(r"(?<=[,;:])\s+", sentence)
    if len(pieces) > 1:
        result: list[str] = []
        current: list[str] = []

        for piece in pieces:
            projected = word_count(" ".join(current + [piece]))
            if current and projected > max_words:
                result.append(" ".join(current).strip())
                current = [piece]
            else:
                current.append(piece)

        if current:
            result.append(" ".join(current).strip())

        if all(word_count(item) <= max_words for item in result):
            return result

    # Final safety fallback.
    return [
        " ".join(words[index:index + max_words])
        for index in range(0, len(words), max_words)
    ]


def chunk_text(
    text: str,
    section: Section,
    *,
    target_words: int = 155,
    min_words: int = 80,
    max_words: int = 195,
) -> list[AudioChunk]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences: list[str] = []

    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if sentence:
            sentences.extend(
                _split_oversized_sentence(sentence, max_words)
            )

    groups: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        if (
            current
            and current_words >= min_words
            and current_words + sentence_words > max_words
        ):
            groups.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

        if current_words >= target_words:
            groups.append(" ".join(current).strip())
            current = []
            current_words = 0

    if current:
        tail = " ".join(current).strip()

        if groups and word_count(tail) < max(25, min_words // 2):
            combined = groups[-1] + " " + tail

            if word_count(combined) <= max_words + 35:
                groups[-1] = combined
            else:
                groups.append(tail)
        else:
            groups.append(tail)

    return [
        AudioChunk(
            id=f"{section.id}-chunk-{index:04d}",
            section_id=section.id,
            order=index,
            text=group,
            word_count=word_count(group),
            source_start_page=section.start_page,
            source_end_page=section.end_page,
        )
        for index, group in enumerate(groups, start=1)
    ]
