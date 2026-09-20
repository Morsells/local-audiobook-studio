param(
    [switch]$DryRun,
    [switch]$RemoveQwen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    Write-Host (" " + $Title) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkGray
}

function Is-ProtectedPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $protected = @(
        (Join-Path $Root ".audiobook_data"),
        (Join-Path $Root ".venv"),
        (Join-Path $Root ".venv-chatterbox"),
        (Join-Path $Root ".streamlit"),
        (Join-Path $Root "src"),
        (Join-Path $Root "app.py"),
        (Join-Path $Root "models\chatterbox"),
        (Join-Path $Root "models\kokoro-v1.0.onnx"),
        (Join-Path $Root "models\voices-v1.0.bin")
    )

    foreach ($p in $protected) {
        $pf = [System.IO.Path]::GetFullPath($p)
        if ($full -eq $pf) {
            return $true
        }
    }
    return $false
}

function Get-SizeBytes([string]$Path) {
    try {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            return (Get-Item -LiteralPath $Path -Force).Length
        }
        if (Test-Path -LiteralPath $Path -PathType Container) {
            $sum = 0L
            Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue |
                ForEach-Object { $sum += $_.Length }
            return $sum
        }
    }
    catch {}
    return 0L
}

function Format-Size([long]$Bytes) {
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N1} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N1} KB" -f ($Bytes / 1KB) }
    return "$Bytes B"
}

$Targets = New-Object System.Collections.Generic.List[string]

function Add-Target([string]$Path) {
    if (-not $Path) { return }
    $full = [System.IO.Path]::GetFullPath($Path)

    if (Is-ProtectedPath $full) {
        Write-Warning "PROTECTED path was requested for deletion and was blocked: $full"
        return
    }

    if (Test-Path -LiteralPath $full) {
        if (-not $Targets.Contains($full)) {
            $Targets.Add($full)
        }
    }
}

function Add-Pattern([string]$Base, [string]$Pattern) {
    if (-not (Test-Path -LiteralPath $Base)) { return }
    Get-ChildItem -LiteralPath $Base -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $Pattern } |
        ForEach-Object { Add-Target $_.FullName }
}

Write-Section "Local Audiobook Studio Cleaner"

Write-Host "Project:" -NoNewline
Write-Host " $Root" -ForegroundColor White
Write-Host "Mode:   " -NoNewline
if ($DryRun) {
    Write-Host "PREVIEW / DRY RUN" -ForegroundColor Yellow
} else {
    Write-Host "DELETE" -ForegroundColor Red
}
Write-Host "Qwen:   " -NoNewline
Write-Host ($(if ($RemoveQwen) { "REMOVE" } else { "KEEP" })) -ForegroundColor Yellow

# ---------------------------------------------------------------------------
# Runtime/transient data
# ---------------------------------------------------------------------------
Add-Target (Join-Path $Root ".runtime")
Add-Target (Join-Path $Root ".pytest_cache")
Add-Target (Join-Path $Root ".audiobook_data\_triton_smoke")
Add-Target (Join-Path $Root ".audiobook_data\_diagnostics")

# Logs are runtime diagnostics, not book/project data.
Add-Target (Join-Path $Root "logs")

# ---------------------------------------------------------------------------
# Rejected TTS stacks
# ---------------------------------------------------------------------------
Add-Target (Join-Path $Root ".venv-kikiri")
Add-Target (Join-Path $Root "models\kikiri")
Add-Target (Join-Path $Root "models\piper")

foreach ($name in @(
    "Setup-Kikiri-GPU.bat",
    "Start-Audiobook-Studio-With-Kikiri.bat",
    "requirements-kikiri.txt",
    "requirements-piper.txt",
    "kikiri_smoke_test.wav"
)) {
    Add-Target (Join-Path $Root $name)
}

foreach ($name in @(
    "check_kikiri_install.py",
    "download_kikiri_models.py",
    "kikiri_tts_server.py",
    "setup_kikiri_windows.ps1",
    "test_kikiri_local.py",
    "download_piper_voices.py"
)) {
    Add-Target (Join-Path $Root ("scripts\" + $name))
}

# ---------------------------------------------------------------------------
# Abandoned benchmark / profiling experiments
# ---------------------------------------------------------------------------
Add-Pattern $Root "benchmark_chatterbox*"
Add-Pattern $Root "Benchmark-Chatterbox-*.bat"
Add-Pattern $Root "Profile-Chatterbox-*.bat"

if (Test-Path (Join-Path $Root "scripts")) {
    Get-ChildItem -LiteralPath (Join-Path $Root "scripts") -File -Force |
        Where-Object {
            $_.Name -like "benchmark_chatterbox_*.py" -or
            $_.Name -like "profile_chatterbox_*.py"
        } |
        ForEach-Object { Add-Target $_.FullName }
}

# Historical experiments that are no longer part of the production path.
foreach ($name in @(
    "Setup-Triton-Windows-Chatterbox.bat",
    "Test-Chatterbox-Unicode.bat",
    "chatterbox_smoke_test.wav"
)) {
    Add-Target (Join-Path $Root $name)
}

foreach ($name in @(
    "setup_triton_windows_chatterbox.py",
    "test_chatterbox_unicode_windows.ps1"
)) {
    Add-Target (Join-Path $Root ("scripts\" + $name))
}

# ---------------------------------------------------------------------------
# Historical release/update clutter
# ---------------------------------------------------------------------------
foreach ($pattern in @(
    "CHANGELOG_V3*.md",
    "UPGRADE_TO_V3*.txt",
    "VALIDATION*.txt",
    "CLEANUP_AUDIT*.txt"
)) {
    Add-Pattern $Root $pattern
}

# ---------------------------------------------------------------------------
# Experimental unit tests only. Keep core functional tests.
# ---------------------------------------------------------------------------
$TestsDir = Join-Path $Root "tests"
if (Test-Path $TestsDir) {
    foreach ($pattern in @(
        "test_v370_fast_export_benchmark.py",
        "test_v372_benchmark_startup.py",
        "test_v373_chatterbox_profiler.py",
        "test_v374_cuda_graph_benchmark.py",
        "test_v376_ab_listening.py",
        "test_v377_deep_profiler.py",
        "test_v378_s3_cuda_graph.py",
        "test_v379_s3_param_cache.py",
        "test_v380_fp16_benchmark.py",
        "test_v381_t3_fp16_cache.py",
        "test_v382_forced_kv_cache.py",
        "test_v383_persistent_graph.py",
        "test_v384_eos_batch.py",
        "test_v385_s3_split_profiler.py",
        "test_v386_flow_compile.py",
        "test_v387_triton_setup.py",
        "test_v388_static_flow_compile.py"
    )) {
        Add-Target (Join-Path $TestsDir $pattern)
    }
}

# Python cache dirs under our own source only; never recurse through venvs.
foreach ($base in @(
    (Join-Path $Root "src"),
    (Join-Path $Root "scripts"),
    (Join-Path $Root "tests")
)) {
    if (Test-Path $base) {
        Get-ChildItem -LiteralPath $base -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq "__pycache__" } |
            ForEach-Object { Add-Target $_.FullName }
    }
}

# ---------------------------------------------------------------------------
# Optional Qwen removal
# ---------------------------------------------------------------------------
if ($RemoveQwen) {
    Add-Target (Join-Path $Root ".venv-qwen")
    Add-Target (Join-Path $Root "models\qwen3-tts")
    Add-Target (Join-Path $Root "Start-Audiobook-Studio-With-Qwen.bat")
    Add-Target (Join-Path $Root "scripts\qwen_tts_server.py")
    Add-Target (Join-Path $Root "scripts\download_qwen3_tts.py")
}

# Sort longest paths first so nested items do not create confusing duplicate
# delete messages when a parent folder is also selected.
$UniqueTargets = $Targets |
    Sort-Object { $_.Length } -Descending |
    Select-Object -Unique

$totalBytes = 0L
foreach ($target in $UniqueTargets) {
    $totalBytes += Get-SizeBytes $target
}

Write-Section "Deletion plan"
Write-Host ("Targets:      {0}" -f $UniqueTargets.Count)
Write-Host ("Approx size:  {0}" -f (Format-Size $totalBytes))
Write-Host ""

foreach ($target in ($UniqueTargets | Sort-Object)) {
    $relative = $target
    if ($target.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $target.Substring($Root.Length).TrimStart('\')
    }
    Write-Host ("  - {0}" -f $relative) -ForegroundColor DarkGray
}

Write-Section "Protected / intentionally kept"
foreach ($keep in @(
    ".audiobook_data\  (books, translations, queue/cache, voice references)",
    ".venv\",
    ".venv-chatterbox\",
    ".streamlit\",
    "models\chatterbox\",
    "models\kokoro-v1.0.onnx",
    "models\voices-v1.0.bin",
    "src\",
    "app.py",
    "scripts\chatterbox_fast_backend.py",
    "scripts\chatterbox_tts_server.py",
    "translation pipeline + Ollama integration",
    "M4B/export pipeline",
    "normal Start / Stop / Status launchers"
)) {
    Write-Host ("  KEEP  " + $keep) -ForegroundColor Green
}

if (-not $RemoveQwen) {
    Write-Host "  KEEP  .venv-qwen / Qwen files (not explicitly selected)" -ForegroundColor Green
}

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN ONLY - nothing was deleted." -ForegroundColor Yellow
    exit 0
}

Write-Section "Stopping running services"
$StopScript = Join-Path $Root "scripts\stop_all.ps1"
if (Test-Path $StopScript) {
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $StopScript
    }
    catch {
        Write-Warning "Stop script reported an error: $($_.Exception.Message)"
    }
} else {
    Write-Warning "scripts\stop_all.ps1 not found; continuing."
}

# Remove experimental Triton package from Chatterbox venv. It was installed
# solely for the abandoned torch.compile experiment.
$ChatterboxPython = Join-Path $Root ".venv-chatterbox\Scripts\python.exe"
if (Test-Path $ChatterboxPython) {
    Write-Host ""
    Write-Host "Removing experimental triton-windows from .venv-chatterbox (if installed)..."
    try {
        & $ChatterboxPython -m pip uninstall -y triton-windows 2>$null | Out-Host
    }
    catch {
        Write-Warning "Could not uninstall triton-windows. File cleanup will continue."
    }
}

Write-Section "Deleting"

$deletedCount = 0
$failed = New-Object System.Collections.Generic.List[string]

foreach ($target in $UniqueTargets) {
    if (-not (Test-Path -LiteralPath $target)) {
        continue
    }

    if (Is-ProtectedPath $target) {
        Write-Warning "Blocked protected path: $target"
        continue
    }

    try {
        Remove-Item -LiteralPath $target -Force -Recurse -ErrorAction Stop
        $deletedCount += 1

        $relative = $target
        if ($target.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
            $relative = $target.Substring($Root.Length).TrimStart('\')
        }
        Write-Host ("  deleted  " + $relative) -ForegroundColor Green
    }
    catch {
        $failed.Add("$target :: $($_.Exception.Message)")
        Write-Warning "Could not delete: $target"
    }
}

# Recreate runtime/log folders so normal launchers have clean destinations.
New-Item -ItemType Directory -Force -Path (Join-Path $Root ".runtime") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

# Remove empty directories caused by cleanup under scripts/tests/models only.
foreach ($base in @(
    (Join-Path $Root "scripts"),
    (Join-Path $Root "tests"),
    (Join-Path $Root "models")
)) {
    if (Test-Path $base) {
        Get-ChildItem -LiteralPath $base -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object {
                $children = @(Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue)
                if ($children.Count -eq 0) {
                    try {
                        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
                    } catch {}
                }
            }
    }
}

Write-Section "Cleanup complete"
Write-Host ("Deleted targets: {0}" -f $deletedCount) -ForegroundColor Green
Write-Host ("Approx planned cleanup size: {0}" -f (Format-Size $totalBytes))

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Warning ("{0} item(s) could not be deleted:" -f $failed.Count)
    foreach ($entry in $failed) {
        Write-Host ("  " + $entry) -ForegroundColor Yellow
    }
    exit 1
}

Write-Host ""
Write-Host "Protected book/project data was not touched." -ForegroundColor Green
Write-Host "Next recommended check:" -ForegroundColor Cyan
Write-Host "  .\Start-Audiobook-Studio-With-Chatterbox.bat"
Write-Host ""
exit 0
