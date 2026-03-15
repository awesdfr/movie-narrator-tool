@echo off
setlocal
call "%~dp0tools-env.bat"
if errorlevel 1 exit /b 1

echo ========================================
echo   Running post-fix validation...
echo ========================================
echo.

set "EXIT_CODE=0"
set "REPORT_PATH=%~dp0temp\claude_feedback\latest_report.md"
set "PYTHON_READY=1"

if /i "%PYTHON_EXE%"=="python" (
    where python >nul 2>nul
    if errorlevel 1 (
        set "PYTHON_READY=0"
        set "EXIT_CODE=9009"
    )
) else (
    if not exist "%PYTHON_EXE%" (
        set "PYTHON_READY=0"
        set "EXIT_CODE=9009"
    )
)

if "%PYTHON_READY%"=="1" (
    "%PYTHON_EXE%" "%~dp0backend\cli\post_fix_validation.py" %*
    set "EXIT_CODE=%ERRORLEVEL%"
)

if "%EXIT_CODE%"=="0" goto end

if "%EXIT_CODE%"=="9009" goto fallback
if exist "%REPORT_PATH%" goto end

:fallback
echo.
echo Python validation unavailable or failed before report generation.
echo Switching to PowerShell fallback validator...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\cli\post_fix_validation_fallback.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

:end
echo.
echo Report: %~dp0temp\claude_feedback\latest_report.md
echo Prompt: %~dp0temp\claude_feedback\latest_prompt.txt
exit /b %EXIT_CODE%
