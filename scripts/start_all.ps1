param(
    [ValidateSet("auto", "on", "off")]
    [string]$Qwen = "auto",

    [ValidateSet("auto", "on", "off")]
    [string]$Chatterbox = "auto",

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeDir = Join-Path $Root ".runtime"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$MainPython = Join-Path $Root ".venv\Scripts\python.exe"
$QwenPython = Join-Path $Root ".venv-qwen\Scripts\python.exe"
$ChatterboxPython = Join-Path $Root ".venv-chatterbox\Scripts\python.exe"
$StreamlitPidFile = Join-Path $RuntimeDir "streamlit.pid"
$QwenPidFile = Join-Path $RuntimeDir "qwen.pid"
$ChatterboxPidFile = Join-Path $RuntimeDir "chatterbox.pid"
$StreamlitOutLog = Join-Path $LogDir "streamlit.out.log"
$StreamlitErrLog = Join-Path $LogDir "streamlit.err.log"
$QwenOutLog = Join-Path $LogDir "qwen.out.log"
$QwenErrLog = Join-Path $LogDir "qwen.err.log"
$ChatterboxOutLog = Join-Path $LogDir "chatterbox.out.log"
$ChatterboxErrLog = Join-Path $LogDir "chatterbox.err.log"
$StreamlitUrl = "http://127.0.0.1:8501"
$QwenHealthUrl = "http://127.0.0.1:7867/health"
$ChatterboxHealthUrl = "http://127.0.0.1:7869/health"

function Test-ProcessAlive { param([string]$PidFile)
    if (-not (Test-Path $PidFile)) { return $false }
    try { $StoredPid=[int](Get-Content $PidFile -Raw).Trim(); Get-Process -Id $StoredPid -ErrorAction Stop | Out-Null; return $true }
    catch { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue; return $false }
}
function Stop-ManagedIfRunning { param([string]$Name,[string]$PidFile)
    if (-not (Test-ProcessAlive $PidFile)) { return }
    try { $StoredPid=[int](Get-Content $PidFile -Raw).Trim(); Write-Host "Stopping managed $Name (PID $StoredPid) to free GPU resources..." -ForegroundColor Yellow; Stop-Process -Id $StoredPid -Force }
    catch { Write-Warning "Could not stop managed $Name." }
    finally { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
}
function Test-HttpReady { param([string]$Url,[int]$TimeoutSec=1)
    try { $r=Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec; return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) } catch { return $false }
}
function Show-StartupLogTail { param([string]$Name,[string]$ErrorLog,[int]$Lines=30)
    if ($ErrorLog -and (Test-Path $ErrorLog)) { Write-Host "Last $Lines lines from $Name error log:" -ForegroundColor Yellow; Get-Content $ErrorLog -Tail $Lines -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray } }
}
function Wait-RequiredServiceReady { param([string]$Name,[string]$Url,[int]$TimeoutSeconds,[string]$PidFile="",[string]$ErrorLog="")
    if (Test-HttpReady $Url) { Write-Host "  [READY] $Name" -ForegroundColor Green; return $true }
    $started=Get-Date; $next=0
    while (((Get-Date)-$started).TotalSeconds -lt $TimeoutSeconds) {
        if (Test-HttpReady $Url) { Write-Host "  [READY] $Name" -ForegroundColor Green; return $true }
        if ($PidFile -and (Test-Path $PidFile) -and (-not (Test-ProcessAlive $PidFile))) { Write-Host "  [FAILED] $Name process exited." -ForegroundColor Red; Show-StartupLogTail $Name $ErrorLog; return $false }
        $elapsed=[int]((Get-Date)-$started).TotalSeconds; if ($elapsed -ge $next) { Write-Host "    ... $elapsed / $TimeoutSeconds s" -ForegroundColor DarkGray; $next += 5 }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "  [TIMEOUT] $Name was not ready." -ForegroundColor Red; Show-StartupLogTail $Name $ErrorLog; return $false
}
function Find-QwenModel {
    $base=Join-Path $Root "models\qwen3-tts"; if (-not (Test-Path $base)) { return $null }
    foreach($name in @("Qwen3-TTS-12Hz-0.6B-CustomVoice","Qwen3-TTS-12Hz-1.7B-CustomVoice","Qwen3-TTS-12Hz-1.7B-VoiceDesign")) { $c=Join-Path $base $name; if(Test-Path $c){return $c} }
    $any=Get-ChildItem $base -Directory -ErrorAction SilentlyContinue | Select-Object -First 1; if($any){return $any.FullName}; return $null
}
function Get-QwenMode { param([string]$ModelPath); if($ModelPath -like "*VoiceDesign*"){return "voice_design"}; return "custom_voice" }
function Test-ChatterboxModel {
    $base=Join-Path $Root "models\chatterbox\multilingual-v3"
    foreach($name in @("ve.pt","t3_mtl23ls_v3.safetensors","s3gen.pt","grapheme_mtl_merged_expanded_v1.json","conds.pt")){ if(-not(Test-Path(Join-Path $base $name))){return $false} }
    return $true
}

Write-Host ""; Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Local Audiobook Studio" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan; Write-Host ""
if(-not(Test-Path $MainPython)){ Write-Host "Main environment missing: $MainPython" -ForegroundColor Red; exit 1 }

$qwenModel=Find-QwenModel
$canQwen=(Test-Path $QwenPython) -and ($null -ne $qwenModel)
$canChatterbox=(Test-Path $ChatterboxPython) -and (Test-ChatterboxModel)
$startChatterbox = if($Chatterbox -eq "on"){$canChatterbox}elseif($Chatterbox -eq "off"){$false}else{$canChatterbox}
if($Chatterbox -eq "on" -and -not $canChatterbox){Write-Warning "Chatterbox was requested but its environment/model is missing."}
$startQwen = if($Qwen -eq "on"){$canQwen}elseif($Qwen -eq "off"){$false}else{$canQwen -and -not $startChatterbox}
if($Qwen -eq "on" -and -not $canQwen){Write-Warning "Qwen was requested but its environment/model is missing."}
if($Qwen -eq "on" -and $Chatterbox -eq "auto"){$startChatterbox=$false}

if($startChatterbox -and -not $startQwen){Stop-ManagedIfRunning "Qwen3-TTS" $QwenPidFile}
if($startQwen -and -not $startChatterbox){Stop-ManagedIfRunning "Chatterbox" $ChatterboxPidFile}

if($startChatterbox){
    if(Test-HttpReady $ChatterboxHealthUrl){Write-Host "Chatterbox is already running." -ForegroundColor Green}
    elseif(-not(Test-ProcessAlive $ChatterboxPidFile)){
        Write-Host "Starting Chatterbox Multilingual V3..." -ForegroundColor Yellow
        $proc=Start-Process -FilePath $ChatterboxPython -ArgumentList @("scripts\chatterbox_tts_server.py","--device","auto") -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $ChatterboxOutLog -RedirectStandardError $ChatterboxErrLog -PassThru
        Set-Content $ChatterboxPidFile $proc.Id
    }
}else{Write-Host "Chatterbox: skipped." -ForegroundColor DarkGray}

if($startQwen){
    if(Test-HttpReady $QwenHealthUrl){Write-Host "Qwen3-TTS is already running." -ForegroundColor Green}
    elseif(-not(Test-ProcessAlive $QwenPidFile)){
        Write-Host "Starting Qwen3-TTS..." -ForegroundColor Yellow
        $mode=Get-QwenMode $qwenModel; $quoted='"'+$qwenModel+'"'
        $proc=Start-Process -FilePath $QwenPython -ArgumentList @("scripts\qwen_tts_server.py","--model-dir",$quoted,"--mode",$mode,"--device","auto") -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $QwenOutLog -RedirectStandardError $QwenErrLog -PassThru
        Set-Content $QwenPidFile $proc.Id
    }
}else{Write-Host "Qwen3-TTS: skipped." -ForegroundColor DarkGray}

$ready=$true
if($startChatterbox){if(-not(Wait-RequiredServiceReady "Chatterbox Multilingual V3" $ChatterboxHealthUrl 240 $ChatterboxPidFile $ChatterboxErrLog)){$ready=$false}}
if($startQwen){if(-not(Wait-RequiredServiceReady "Qwen3-TTS" $QwenHealthUrl 240 $QwenPidFile $QwenErrLog)){$ready=$false}}
if(-not $ready){Write-Host "Startup aborted: requested engine did not become ready." -ForegroundColor Red; exit 1}

if(Test-HttpReady $StreamlitUrl){Write-Host "Streamlit is already running." -ForegroundColor Green}
elseif(-not(Test-ProcessAlive $StreamlitPidFile)){
    Write-Host "Starting Local Audiobook Studio..." -ForegroundColor Yellow
    $args=@("-m","streamlit","run","app.py","--server.address","127.0.0.1","--server.port","8501","--browser.gatherUsageStats","false")
    $proc=Start-Process -FilePath $MainPython -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $StreamlitOutLog -RedirectStandardError $StreamlitErrLog -PassThru
    Set-Content $StreamlitPidFile $proc.Id
}
if(-not(Wait-RequiredServiceReady "Local Audiobook Studio web app" $StreamlitUrl 75 $StreamlitPidFile $StreamlitErrLog)){exit 1}
Write-Host "Studio: $StreamlitUrl" -ForegroundColor Cyan
if($startChatterbox){Write-Host "Chatterbox: http://127.0.0.1:7869" -ForegroundColor Cyan}
if($startQwen){Write-Host "Qwen:       http://127.0.0.1:7867" -ForegroundColor Cyan}
if(-not $NoBrowser){Start-Process $StreamlitUrl}
