$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$ExactTargets = @(
    "requirements-piper.txt",
    "requirements-kikiri.txt",
    "Setup-Kikiri-GPU.bat",
    "Start-Audiobook-Studio-With-Kikiri.bat",
    "Setup-Triton-Windows-Chatterbox.bat",
    "Test-Chatterbox-Unicode.bat",
    "Start-Audiobook-Studio-Fast.bat",
    "scripts\setup_kikiri_windows.ps1",
    "scripts\kikiri_tts_server.py",
    "scripts\download_kikiri_models.py",
    "scripts\download_piper_voices.py",
    "scripts\check_kikiri_install.py",
    "scripts\test_kikiri_local.py",
    "scripts\setup_triton_windows_chatterbox.py",
    "scripts\test_chatterbox_unicode_windows.ps1",
    "tests\test_v32_piper_storage.py",
    "tests\test_v34_source_preview.py",
    "tests\test_v35_hardware_kikiri.py",
    "tests\test_v36_chatterbox.py",
    "tests\test_v36_chatterbox_privacy.py",
    "UI_CLEANUP_VALIDATION.txt"
)

foreach ($RelativePath in $ExactTargets) {
    $TargetPath = Join-Path $Root $RelativePath
    if (Test-Path -LiteralPath $TargetPath) {
        Remove-Item -LiteralPath $TargetPath -Force -Recurse -ErrorAction SilentlyContinue
        Write-Host "removed $RelativePath" -ForegroundColor DarkGray
    }
}

$RootPatterns = @(
    "Benchmark-Chatterbox-*.bat",
    "Profile-Chatterbox-*.bat",
    "CHANGELOG_V3*.md",
    "UPGRADE_TO_V3*.txt",
    "VALIDATION*.txt",
    "CLEANUP_AUDIT*.txt",
    "benchmark_chatterbox*"
)

foreach ($Pattern in $RootPatterns) {
    Get-ChildItem -LiteralPath $Root -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $Pattern } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force -Recurse -ErrorAction SilentlyContinue
            Write-Host "removed $($_.Name)" -ForegroundColor DarkGray
        }
}

$ScriptsDir = Join-Path $Root "scripts"
if (Test-Path $ScriptsDir) {
    Get-ChildItem -LiteralPath $ScriptsDir -File -Force -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "benchmark_chatterbox_*.py" -or
            $_.Name -like "profile_chatterbox_*.py"
        } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            Write-Host "removed scripts\$($_.Name)" -ForegroundColor DarkGray
        }
}

foreach ($Base in @(
    (Join-Path $Root "src"),
    (Join-Path $Root "scripts"),
    (Join-Path $Root "tests")
)) {
    if (Test-Path $Base) {
        Get-ChildItem -LiteralPath $Base -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq "__pycache__" } |
            Sort-Object FullName -Descending |
            ForEach-Object {
                Remove-Item -LiteralPath $_.FullName -Force -Recurse -ErrorAction SilentlyContinue
            }
    }
}

Write-Host "" 
Write-Host "Production code cleanup complete." -ForegroundColor Green
Write-Host "Retained TTS: Chatterbox V3, Kokoro, optional Qwen3-TTS." -ForegroundColor Green
Write-Host "Your .audiobook_data and model folders were not touched." -ForegroundColor Green
Write-Host "" 
Write-Host "Start with:" -ForegroundColor Cyan
Write-Host "  .\Start-Audiobook-Studio-With-Chatterbox.bat"
