param(
    [string]$Model = "translategemma:12b"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Local Audiobook Studio - Translation Diagnostic" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

$Ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $Ollama) {
    Write-Host "Ollama is not installed or not on PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Ollama version:" -ForegroundColor Cyan
ollama --version
Write-Host ""
Write-Host "Installed TranslateGemma models:" -ForegroundColor Cyan
ollama list | Select-String -SimpleMatch "translategemma"
Write-Host ""
Write-Host "Current Ollama processes:" -ForegroundColor Cyan
ollama ps
Write-Host ""
Write-Host "Running a tiny EN -> DE model test..." -ForegroundColor Green

$Body = @{
    model = $Model
    messages = @(
        @{
            role = "user"
            content = @"
You are a professional English (en) to German (de) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to German grammar, vocabulary, and cultural sensitivities.
Produce only the German translation, without any additional explanations or commentary. Please translate the following English text into German:


Always make those above you feel comfortably superior.
"@
        }
    )
    stream = $false
    options = @{
        temperature = 0.2
        top_p = 0.9
        top_k = 64
        num_ctx = 4096
        num_predict = 256
    }
} | ConvertTo-Json -Depth 8

try {
    $Response = Invoke-RestMethod `
        -Uri "http://127.0.0.1:11434/api/chat" `
        -Method Post `
        -ContentType "application/json" `
        -Body $Body `
        -TimeoutSec 600

    Write-Host ""
    Write-Host "PASS - Ollama + TranslateGemma responded:" -ForegroundColor Green
    Write-Host $Response.message.content
    Write-Host ""
    ollama ps
}
catch {
    Write-Host ""
    Write-Host "FAIL - Ollama returned an error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
        Write-Host ""
        Write-Host "Server detail:" -ForegroundColor Yellow
        Write-Host $_.ErrorDetails.Message
    }
    Write-Host ""
    Write-Host "Direct model check:" -ForegroundColor Yellow
    Write-Host "  ollama run $Model"
    exit 1
}
