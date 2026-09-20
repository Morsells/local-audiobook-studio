param(
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv-chatterbox"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Chatterbox Multilingual V3 Local GPU Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This explicit setup step downloads packages/models from the internet."
Write-Host "Normal audiobook runtime uses only local files afterward."
Write-Host ""

# Chatterbox 0.1.7 contains a git-based Perth dependency.
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $DefaultGit = "C:\Program Files\Git\cmd\git.exe"

    if (Test-Path $DefaultGit) {
        $env:PATH = (Split-Path $DefaultGit) + ";" + $env:PATH
    }
    elseif (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Git is required by one Chatterbox dependency. Installing Git..."
        winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements

        if (Test-Path $DefaultGit) {
            $env:PATH = (Split-Path $DefaultGit) + ";" + $env:PATH
        }
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to install Chatterbox's Perth dependency. Install Git and rerun this setup."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating isolated .venv-chatterbox with Python 3.12..."
    py -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv-chatterbox with Python 3.12."
    }
}

Write-Host "Updating pip/setuptools/wheel..."
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "pip bootstrap failed."
}

Write-Host ""
Write-Host "Removing any stale PyPI Chatterbox build..."
& $Python -m pip uninstall -y chatterbox-tts | Out-Null

Write-Host "Installing official Chatterbox Multilingual V3 release source..."
& $Python -m pip install -r (Join-Path $Root "requirements-chatterbox.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox V3 source installation failed."
}

Write-Host ""
Write-Host "Checking Chatterbox Multilingual V3 local API..."
& $Python (Join-Path $Root "scripts\check_chatterbox_install.py")
if ($LASTEXITCODE -ne 0) {
    throw "Installed Chatterbox build does not expose the required V3 local API."
}

Write-Host ""
Write-Host "Checking CUDA-enabled PyTorch 2.6.0 + torchaudio 2.6.0..."
& $Python -c "import torch, torchaudio; ok = torch.__version__.startswith('2.6.0+cu126') and torchaudio.__version__.startswith('2.6.0+cu126') and torch.cuda.is_available(); print('CUDA Torch already ready.' if ok else 'CUDA Torch needs installation.'); raise SystemExit(0 if ok else 1)"
$TorchReady = ($LASTEXITCODE -eq 0)

if (-not $TorchReady) {
    Write-Host "Installing CUDA PyTorch 2.6.0 + torchaudio 2.6.0..."
    & $Python -m pip install --upgrade --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed."
    }
}
else {
    Write-Host "Skipping PyTorch reinstall." -ForegroundColor Green
}

Write-Host ""
Write-Host "Downloading Chatterbox Multilingual V3 model files..."
& $Python (Join-Path $Root "scripts\download_chatterbox_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox model download failed."
}

Write-Host ""
Write-Host "Preparing auxiliary tokenizer assets for offline runtime..." -ForegroundColor Cyan
& $Python (Join-Path $Root "scripts\prepare_chatterbox_assets.py")
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox tokenizer asset preparation failed."
}

Write-Host ""
Write-Host "Chatterbox hardware check:" -ForegroundColor Cyan
& $Python (Join-Path $Root "scripts\check_hardware.py")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Hardware check reported a problem."
}

Write-Host ""
Write-Host "Running local German smoke test..." -ForegroundColor Cyan
& $Python (Join-Path $Root "scripts\test_chatterbox_local.py") --device auto
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox local German smoke test failed."
}

Write-Host ""
Write-Host "Chatterbox setup complete." -ForegroundColor Green
Write-Host "Smoke test audio: $Root\chatterbox_smoke_test.wav"
Write-Host ""
Write-Host "Start with:"
Write-Host "  .\Start-Audiobook-Studio-With-Chatterbox.bat"
