from src.audiobook_studio.storage import (
    clear_redundant_intermediate_wavs,
    clear_tts_chunk_cache,
    compact_finished_project,
    project_storage_report,
    prune_tts_chunk_cache,
    safe_cleanup_project,
)


def test_storage_report_and_safe_cleanup(tmp_path):
    project = tmp_path / "book"
    chunks = project / "audio_chunks"
    previews = project / "previews"
    output = project / "output"

    chunks.mkdir(parents=True)
    previews.mkdir()
    output.mkdir()

    (project / "source.pdf").write_bytes(b"s" * 100)
    (project / "translations.json").write_bytes(b"t" * 20)
    (chunks / "chunk.wav").write_bytes(b"c" * 300)
    (previews / "preview.wav").write_bytes(b"p" * 200)
    (output / "chapter.wav").write_bytes(b"w" * 500)
    (output / "chapter.mp3").write_bytes(b"m" * 100)

    report = project_storage_report(project)
    assert report["tts_chunk_cache"] == 300
    assert report["preview_cache"] == 200
    assert report["intermediate_wav"] == 500

    result = safe_cleanup_project(project)
    assert result.bytes_removed >= 700
    assert (chunks / "chunk.wav").exists()
    assert not (previews / "preview.wav").exists()
    assert not (output / "chapter.wav").exists()
    assert (output / "chapter.mp3").exists()
    assert (project / "translations.json").exists()


def test_prune_keeps_current_cache_only(tmp_path):
    project = tmp_path / "book"
    chunks = project / "audio_chunks"
    chunks.mkdir(parents=True)

    keep = chunks / "keep.wav"
    old = chunks / "old.wav"
    keep.write_bytes(b"a" * 10)
    old.write_bytes(b"b" * 20)

    result = prune_tts_chunk_cache(project, [keep])
    assert result.files_removed == 1
    assert keep.exists()
    assert not old.exists()


def test_compact_finished_project_preserves_final_audio_and_text(tmp_path):
    project = tmp_path / "book"
    (project / "audio_chunks").mkdir(parents=True)
    (project / "previews").mkdir()
    (project / "output").mkdir()

    (project / "source.pdf").write_bytes(b"source")
    (project / "translations.json").write_text('{"x":"y"}', encoding="utf-8")
    (project / "audio_chunks" / "a.wav").write_bytes(b"a" * 100)
    (project / "previews" / "p.wav").write_bytes(b"p" * 100)
    (project / "output" / "final.mp3").write_bytes(b"m" * 100)
    (project / "output" / "final.m4b").write_bytes(b"b" * 100)

    compact_finished_project(project)

    assert (project / "source.pdf").exists()
    assert (project / "translations.json").exists()
    assert (project / "output" / "final.mp3").exists()
    assert (project / "output" / "final.m4b").exists()
    assert not (project / "audio_chunks" / "a.wav").exists()
    assert not (project / "previews" / "p.wav").exists()
