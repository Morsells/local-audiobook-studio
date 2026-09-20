$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $Root ".runtime"

function Show-ManagedStatus {
    param(
        [string]$Name,
        [string]$PidFile,
        [string]$Url
    )

    $PidText = "-"

    if (Test-Path $PidFile) {
        try {
            $StoredPid = [int](Get-Content $PidFile -Raw).Trim()
            Get-Process -Id $StoredPid -ErrorAction Stop | Out-Null
            $PidText = "$StoredPid"
        }
        catch {
            $PidText = "stale"
        }
    }

    $Ready = $false
    try {
        $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1
        $Ready = ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500)
    }
    catch {
        $Ready = $false
    }

    $State = if ($Ready) { "READY" } else { "DOWN" }
    Write-Host ("{0,-20} {1,-7} PID {2}" -f $Name, $State, $PidText)
}

Write-Host ""
Write-Host "Local Audiobook Studio status" -ForegroundColor Cyan
Write-Host "-----------------------------"

Show-ManagedStatus -Name "Streamlit" -PidFile (Join-Path $RuntimeDir "streamlit.pid") -Url "http://127.0.0.1:8501"
Show-ManagedStatus -Name "Chatterbox V3" -PidFile (Join-Path $RuntimeDir "chatterbox.pid") -Url "http://127.0.0.1:7869/health"
Show-ManagedStatus -Name "Qwen3-TTS" -PidFile (Join-Path $RuntimeDir "qwen.pid") -Url "http://127.0.0.1:7867/health"

Write-Host ""
