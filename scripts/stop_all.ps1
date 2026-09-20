$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $Root ".runtime"

function Stop-ManagedProcess {
    param(
        [string]$Name,
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        Write-Host "${Name}: no managed PID file."
        return
    }

    try {
        $StoredPid = [int](Get-Content $PidFile -Raw).Trim()
        Get-Process -Id $StoredPid -ErrorAction Stop | Out-Null
        Write-Host "Stopping $Name (PID $StoredPid)..."
        Stop-Process -Id $StoredPid -Force -ErrorAction Stop
        Write-Host "$Name stopped." -ForegroundColor Green
    }
    catch {
        Write-Host "$Name is not running."
    }
    finally {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "Stopping Local Audiobook Studio..." -ForegroundColor Cyan

Stop-ManagedProcess -Name "Streamlit" -PidFile (Join-Path $RuntimeDir "streamlit.pid")
Stop-ManagedProcess -Name "Chatterbox" -PidFile (Join-Path $RuntimeDir "chatterbox.pid")
Stop-ManagedProcess -Name "Qwen3-TTS" -PidFile (Join-Path $RuntimeDir "qwen.pid")

Write-Host ""
Write-Host "Done." -ForegroundColor Green
