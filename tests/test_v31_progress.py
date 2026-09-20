import time
from pathlib import Path

from src.audiobook_studio.models import AudioChunk, BuildOptions
from src.audiobook_studio.pipeline import _record_metric, _synthesize_with_live_progress


class FakeEngine:
    engine_id = "fake-engine"

    def synthesize_chunk(self, chunk, destination, options):
        time.sleep(0.08)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake")
        return {
            "elapsed": 0.08,
            "duration": 0.5,
            "words": float(chunk.word_count),
        }


def test_live_progress_emits_updates_and_eta(tmp_path):
    _record_metric(
        tmp_path,
        {
            "elapsed": 2.0,
            "duration": 4.0,
            "words": 100.0,
            "engine_id": "fake-engine",
        },
    )

    chunk = AudioChunk(
        id="c1",
        section_id="s1",
        order=1,
        text="hello world",
        word_count=50,
        source_start_page=1,
        source_end_page=1,
    )
    options = BuildOptions(
        start_page=1,
        end_page=1,
        voice="test",
        language="en",
    )
    updates = []
    destination = tmp_path / "audio.wav"

    metric = _synthesize_with_live_progress(
        FakeEngine(),
        tmp_path,
        chunk,
        destination,
        options,
        progress=lambda message, fraction: updates.append((message, fraction)),
        base_fraction=0.1,
        span_fraction=0.8,
        label="Test chunk",
    )

    assert destination.exists()
    assert metric["engine_id"] == "fake-engine"
    assert updates
    assert any("estimated" in message for message, _ in updates)
    assert all(0.0 <= fraction <= 1.0 for _, fraction in updates)
