$ErrorActionPreference = "Stop"

$PatchRoot = Split-Path -Parent $PSScriptRoot
$Target = (Get-Location).Path

Write-Host ""
Write-Host "Local Audiobook Studio - repository polish" -ForegroundColor Cyan
Write-Host "Target: $Target"
Write-Host ""

if (-not (Test-Path (Join-Path $Target ".git"))) {
    throw "Run this from the root of your local Git repository."
}

$copyFiles = @(
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements-translation.txt",
    ".gitattributes",
    "Setup-Audiobook-Studio.bat",
    "scripts\setup_main_windows.ps1",
    "docs\ARCHITECTURE.md"
)

foreach ($relative in $copyFiles) {
    $source = Join-Path $PatchRoot $relative
    $dest = Join-Path $Target $relative

    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Source missing, skipping: $relative"
        continue
    }

    $sourceFull = [System.IO.Path]::GetFullPath($source)
    $destFull = [System.IO.Path]::GetFullPath($dest)

    if ($sourceFull -ieq $destFull) {
        Write-Host "already in place: $relative" -ForegroundColor DarkGray
        continue
    }

    $parent = Split-Path -Parent $dest
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    Copy-Item -LiteralPath $source -Destination $dest -Force
    Write-Host "updated $relative" -ForegroundColor Green
}

$delete = @(
    "CLEANUP.md",
    "CLEANUP_PLAN.txt",
    "CODE_CLEANUP_VALIDATION.txt",
    "Clean-Audiobook-Studio.bat",
    "Finalize-Code-Cleanup.bat",
    "START_HERE.txt",
    "scripts\clean_project.ps1",
    "scripts\finalize_code_cleanup.ps1",
    "scripts\migrate_from_old_version.py"
)

foreach ($relative in $delete) {
    $p = Join-Path $Target $relative
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Force -Recurse
        Write-Host "removed $relative" -ForegroundColor Green
    } else {
        Write-Host "already absent: $relative" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Repository polish applied successfully." -ForegroundColor Green
Write-Host ""
Write-Host "Review with:"
Write-Host "  git status"
Write-Host ""
Write-Host "Then commit with:"
Write-Host '  git add .'
Write-Host '  git commit -m "Polish repository for public use"'
Write-Host '  git push'
Write-Host ""
