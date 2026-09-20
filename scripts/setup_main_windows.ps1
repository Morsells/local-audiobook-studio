$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Local Audiobook Studio - Base Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) was not found. Install Python 3.12 for Windows and rerun this setup."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating .venv with Python 3.12..." -ForegroundColor Cyan
    & py -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv with Python 3.12."
    }
}

Write-Host "Updating pip/setuptools/wheel..."
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "pip bootstrap failed."
}

Write-Host ""
Write-Host "Installing base application dependencies..." -ForegroundColor Cyan
& $Python -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Base dependency installation failed."
}

Write-Host ""
Write-Host "Downloading Kokoro model files..." -ForegroundColor Cyan
& $Python (Join-Path $Root "scripts\download_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "Kokoro model download failed."
}

Write-Host ""
Write-Host "Base setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "For recommended German/multilingual narration, run:"
Write-Host "  .\Setup-Chatterbox-GPU.bat"
Write-Host ""
Write-Host "Then start the studio with:"
Write-Host "  .\Start-Audiobook-Studio-With-Chatterbox.bat"
