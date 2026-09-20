param(
    [string]$Model = "translategemma:12b"
)

$ErrorActionPreference = "Stop"

Write-Host "" 
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Local Audiobook Studio - Literary Translation" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

$Ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $Ollama) {
    Write-Host "Ollama is not installed or not on PATH." -ForegroundColor Red
    Write-Host "Install Ollama for Windows from: https://ollama.com/download" -ForegroundColor Yellow
    Write-Host "Then run this setup again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Ollama:" (ollama --version)
Write-Host ""
Write-Host "Downloading local translation model: $Model" -ForegroundColor Green
Write-Host "This setup step uses the internet. Book text is NOT sent anywhere." -ForegroundColor Yellow
ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    throw "ollama pull failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Installed Ollama models:" -ForegroundColor Cyan
ollama list

Write-Host ""
Write-Host "Running a small local model smoke test..." -ForegroundColor Cyan
$TestScript = Join-Path $PSScriptRoot "test_literary_translation_windows.ps1"
& $TestScript -Model $Model
if ($LASTEXITCODE -ne 0) {
    throw "TranslateGemma smoke test failed. See the Ollama error above."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "For translation, start the studio with:" -ForegroundColor Green
Write-Host "  .\\Start-Audiobook-Studio-Translate.bat"
Write-Host "Then choose: Ollama - local literary / translategemma:12b"
