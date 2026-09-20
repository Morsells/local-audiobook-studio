from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audiobook_studio.hardware import detect_hardware


def main() -> int:
    info = detect_hardware()

    print("Hardware acceleration check")
    print("---------------------------")
    print(f"CPU:              {info.cpu_name}")
    print(f"Logical CPUs:     {info.logical_cpus}")
    print(f"Torch installed:  {info.torch_installed}")
    print(f"Torch version:    {info.torch_version or '-'}")
    print(f"CUDA available:   {info.cuda_available}")
    print(f"Torch CUDA:       {info.cuda_version or '-'}")
    print(f"GPU:              {info.gpu_name or '-'}")
    print(f"VRAM:             {info.gpu_vram_gb:.1f} GB" if info.gpu_vram_gb else "VRAM:             -")
    print(f"Compute cap:      {info.compute_capability or '-'}")
    print(f"BF16 supported:   {info.bf16_supported}")
    print(f"Recommended dev:  {info.recommended_device}")
    print(f"Recommended dtype:{info.recommended_precision}")
    print(f"OPUS batch:       {info.recommended_translation_batch}")

    if info.gpu_name and not info.cuda_available:
        print()
        print("WARNING: NVIDIA GPU exists, but this Python environment cannot use CUDA.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
