@echo off
cd /d "%~dp0"
setlocal EnableExtensions
title Local Audiobook Studio Cleaner
echo.
echo ======================================================================
echo  Local Audiobook Studio - Safe Project Cleaner
echo ======================================================================
echo.
echo This keeps your books, translations, voice reference, Chatterbox,
echo Kokoro, main venv and normal production pipeline.
echo.
echo Qwen is optional because it may still be useful as a fallback.
set "REMOVE_QWEN=N"
set /p "REMOVE_QWEN=Remove Qwen environment/models too? [j/N]: "
set "QWEN_ARG="
if /I "%REMOVE_QWEN%"=="J" set "QWEN_ARG=-RemoveQwen"
if /I "%REMOVE_QWEN%"=="Y" set "QWEN_ARG=-RemoveQwen"
echo.
echo First showing the exact deletion plan...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\clean_project.ps1" -DryRun %QWEN_ARG%
if errorlevel 1 (
  echo.
  echo Preview failed. Nothing was deleted.
  pause
  exit /b 1
)
echo.
echo ======================================================================
choice /C JN /N /M "Delete exactly the items shown above? [J/N]: "
if errorlevel 2 (
  echo Cancelled. Nothing was deleted.
  pause
  exit /b 0
)
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\clean_project.ps1" %QWEN_ARG%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo Cleanup finished successfully.
) else (
  echo Cleanup finished with warnings/errors. See output above.
)
pause
exit /b %RC%
