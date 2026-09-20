@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\apply_repo_polish.ps1"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo Repository polish finished successfully.
) else (
  echo Repository polish failed. See error above.
)
pause
exit /b %RC%
