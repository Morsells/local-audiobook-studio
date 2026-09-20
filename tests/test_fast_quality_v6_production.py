from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fast_quality_v6_backend_is_production_profile():
    text = (
        ROOT / "scripts" / "chatterbox_fast_backend.py"
    ).read_text(encoding="utf-8")

    assert 'FAST_QUALITY_PROFILE = "adaptive_v6"' in text
    assert "ADAPTIVE_TOKENS_PER_WORD = 16" in text
    assert "ADAPTIVE_RESERVE_TOKENS = 64" in text
    assert "ADAPTIVE_MIN_TOKENS = 256" in text
    assert "torch.cuda.get_rng_state()" in text
    assert "torch.cuda.set_rng_state(rng_state)" in text
    assert "_las_phrase_word_count" in text
    assert "_las_fast_quality_last" in text


def test_chatterbox_sidecar_supplies_phrase_length_and_logs_fallback():
    text = (
        ROOT / "scripts" / "chatterbox_tts_server.py"
    ).read_text(encoding="utf-8")

    assert 'fast_quality_profile": "adaptive_v6"' in text
    assert "State.model.t3._las_phrase_word_count = _words(phrase)" in text
    assert "Fast Quality profile=adaptive_v6" in text
    assert "fallback=" in text
