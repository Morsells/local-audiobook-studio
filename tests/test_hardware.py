from src.audiobook_studio.hardware import (
    HardwareInfo,
    resolve_device,
    resolve_precision,
)
from src.audiobook_studio.models import BuildOptions, TTSEngine


def test_unknown_historical_tts_setting_migrates_to_chatterbox():
    data = BuildOptions(
        start_page=1,
        end_page=2,
        voice="x",
        language="de",
    ).to_dict()
    data["tts_engine"] = "Removed historical engine"
    data["removed_voice_field"] = "legacy"
    restored = BuildOptions.from_dict(data)
    assert restored.tts_engine == TTSEngine.CHATTERBOX
    assert not hasattr(restored, "removed_voice_field")


def test_turing_gpu_uses_fp16_not_bf16():
    info = HardwareInfo(
        torch_installed=True,
        cuda_available=True,
        gpu_name="NVIDIA GeForce RTX 2080 Ti",
        gpu_vram_gb=11.0,
        compute_capability="7.5",
        bf16_supported=False,
        recommended_device="cuda",
        recommended_precision="float16",
        recommended_translation_batch=8,
    )
    assert resolve_device("auto", info) == "cuda"
    assert resolve_precision("auto", info) == "float16"


def test_gpu_mode_requires_cuda():
    info = HardwareInfo(
        cuda_available=False,
        gpu_name="NVIDIA GPU visible through nvidia-smi",
    )
    try:
        resolve_device("gpu", info)
    except RuntimeError as exc:
        assert "CUDA is unavailable" in str(exc)
    else:
        raise AssertionError("GPU mode should fail without CUDA Torch")
