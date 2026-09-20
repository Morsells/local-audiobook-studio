from src.audiobook_studio.chunker import (
    chunk_text,
)
from src.audiobook_studio.models import Section


def section():
    return Section(
        id="section-1",
        title="Test",
        start_page=1,
        end_page=3,
        blocks=[],
    )


def test_long_text_creates_multiple_chunks():
    sentence = (
        "This is a technical sentence with enough "
        "words to make the chunking behavior predictable."
    )

    text = " ".join(
        [sentence] * 60
    )

    chunks = chunk_text(
        text,
        section(),
        target_words=100,
        min_words=60,
        max_words=130,
    )

    assert len(chunks) > 1
    assert all(
        chunk.word_count <= 165
        for chunk in chunks
    )


def test_short_text_stays_together():
    chunks = chunk_text(
        "This is a short section. "
        "It should remain together.",
        section(),
    )

    assert len(chunks) == 1
