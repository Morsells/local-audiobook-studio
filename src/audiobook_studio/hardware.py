from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
import subprocess
from typing import Any


@dataclass
class HardwareInfo:
    cpu_name: str = ""
    logical_cpus: int = 0
    torch_installed: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    compute_capability: str = ""
    bf16_supported: bool = False
    recommended_device: str = "cpu"
    recommended_precision: str = "float32"
    recommended_translation_batch: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_name() -> str:
    value = platform.processor().strip()
    if value:
        return value
    try:
        output = subprocess.check_output(
            ["wmic", "cpu", "get", "name"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) >= 2:
            return lines[1]
    except Exception:
        pass
    return platform.machine() or "CPU"



def _nvidia_smi_info() -> tuple[str, float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=4,
        ).strip().splitlines()[0]
        name, memory_mb = [part.strip() for part in output.split(",", 1)]
        return name, float(memory_mb) / 1024.0
    except Exception:
        return "", 0.0


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo(
        cpu_name=_cpu_name(),
        logical_cpus=int(os.cpu_count() or 1),
    )

    try:
        import torch
    except Exception:
        info.gpu_name, info.gpu_vram_gb = _nvidia_smi_info()
        return info

    info.torch_installed = True
    info.torch_version = str(getattr(torch, "__version__", ""))
    info.cuda_version = str(getattr(torch.version, "cuda", "") or "")
    info.cuda_available = bool(torch.cuda.is_available())

    if info.cuda_available:
        try:
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            info.gpu_name = str(props.name)
            info.gpu_vram_gb = float(props.total_memory) / (1024.0 ** 3)
            info.compute_capability = f"{props.major}.{props.minor}"
        except Exception:
            pass

        try:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            native_arch_support = int(props.major) >= 8  # Ampere or newer
            try:
                torch_reports_bf16 = bool(
                    torch.cuda.is_bf16_supported(including_emulation=False)
                )
            except TypeError:
                torch_reports_bf16 = bool(torch.cuda.is_bf16_supported())
            info.bf16_supported = native_arch_support and torch_reports_bf16
        except Exception:
            info.bf16_supported = False

        info.recommended_device = "cuda"
        # BF16 is used only when Torch explicitly reports support. This avoids
        # forcing BF16 onto Turing cards such as the RTX 2080 Ti.
        info.recommended_precision = (
            "bfloat16" if info.bf16_supported else "float16"
        )

        if info.gpu_vram_gb >= 16:
            info.recommended_translation_batch = 12
        elif info.gpu_vram_gb >= 8:
            info.recommended_translation_batch = 8
        elif info.gpu_vram_gb >= 4:
            info.recommended_translation_batch = 4
        else:
            info.recommended_translation_batch = 2
    else:
        # A CPU-only Torch wheel can hide an otherwise perfectly usable NVIDIA GPU.
        # Keep reporting the physical GPU so the UI can suggest the one-click CUDA setup.
        smi_name, smi_vram = _nvidia_smi_info()
        if smi_name:
            info.gpu_name = smi_name
            info.gpu_vram_gb = smi_vram

        info.recommended_device = "cpu"
        info.recommended_precision = "float32"
        info.recommended_translation_batch = max(
            1,
            min(4, info.logical_cpus // 4),
        )

    return info


def resolve_device(
    hardware_mode: str = "auto",
    info: HardwareInfo | None = None,
) -> str:
    info = info or detect_hardware()
    mode = (hardware_mode or "auto").lower()

    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        if not info.cuda_available:
            raise RuntimeError(
                "GPU mode was requested, but CUDA is unavailable in this Python environment."
            )
        return "cuda"
    return "cuda" if info.cuda_available else "cpu"


def resolve_precision(
    hardware_mode: str = "auto",
    info: HardwareInfo | None = None,
) -> str:
    device = resolve_device(hardware_mode, info)
    if device == "cpu":
        return "float32"
    info = info or detect_hardware()
    return "bfloat16" if info.bf16_supported else "float16"


def configure_torch_cpu_threads(torch_module) -> None:
    logical = int(os.cpu_count() or 1)
    threads = max(1, logical - 1)
    try:
        torch_module.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch_module.set_num_interop_threads(max(1, min(4, logical // 2)))
    except Exception:
        pass
