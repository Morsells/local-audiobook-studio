param(
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Install-CudaTorch {
    param(
        [string]$Name,
        [string]$PythonPath
    )

    if (-not (Test-Path $PythonPath)) {
        Write-Host "${Name}: environment not present; skipping." -ForegroundColor DarkGray
        return
    }

    Write-Host ""
    Write-Host "[$Name] Installing/upgrading CUDA-enabled PyTorch..." -ForegroundColor Cyan

    & $PythonPath -m pip install --upgrade torch --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed for $Name."
    }

    Write-Host "[$Name] Hardware check:" -ForegroundColor Cyan
    & $PythonPath (Join-Path $Root "scripts\check_hardware.py")
}

$MainPython = Join-Path $Root ".venv\Scripts\python.exe"
$QwenPython = Join-Path $Root ".venv-qwen\Scripts\python.exe"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Local Audiobook Studio GPU Acceleration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "CUDA wheel source: $TorchIndex"
Write-Host "This is the compatibility-oriented CUDA 12.6 PyTorch channel."
Write-Host ""

Install-CudaTorch -Name "Main app / OPUS" -PythonPath $MainPython
Install-CudaTorch -Name "Qwen3-TTS" -PythonPath $QwenPython

Write-Host ""
Write-Host "GPU acceleration setup finished." -ForegroundColor Green
Write-Host "Restart the audiobook launcher so the background sidecars reload on CUDA."
Write-Host "Chatterbox has its own pinned Torch runtime; use Setup-Chatterbox-GPU.bat for it." -ForegroundColor DarkGray
