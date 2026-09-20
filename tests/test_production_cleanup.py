from pathlib import Path

from src.audiobook_studio.models import TTSEngine

ROOT = Path(__file__).resolve().parents[1]


def test_only_retained_tts_engines_exist():
    assert set(TTSEngine) == {
        TTSEngine.CHATTERBOX,
        TTSEngine.KOKORO,
        TTSEngine.QWEN3,
    }


def test_core_source_has_no_removed_engine_symbols():
    files = [ROOT / "app.py"] + list(
        (ROOT / "src" / "audiobook_studio").glob("*.py")
    )
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in files
    )
    assert "TTSEngine.PIPER" not in text
    assert "TTSEngine.KIKIRI" not in text
    assert "KIKIRI_MODEL_DIR" not in text
    assert "PIPER_MODEL_DIR" not in text


def test_fast_backend_contains_only_production_cuda_graph():
    text = (
        ROOT / "scripts" / "chatterbox_fast_backend.py"
    ).read_text(encoding="utf-8")
    assert "def inference_cuda_graph(" in text
    for dead in (
        "inference_lean",
        "inference_cuda_graph_persistent",
        "inference_cuda_graph_eos_batched",
        "s3_solve_euler_cuda_graph",
        "compile_flow",
        "t3_fp16",
        "s3_fp16",
        "param_cache",
    ):
        assert dead not in text


def test_start_script_has_only_retained_sidecars():
    text = (ROOT / "scripts" / "start_all.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "Chatterbox" in text
    assert "Qwen3-TTS" in text
    assert "Kikiri" not in text
    assert "$id=" not in text.lower()
