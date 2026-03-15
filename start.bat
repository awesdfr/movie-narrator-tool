@echo off
chcp 65001 >nul
call "%~dp0tools-env.bat"

echo.
echo ========================================
echo     Movie Narrator Tool
echo ========================================
echo.

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ not found
    pause
    exit /b 1
)

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FFmpeg not found
    pause
    exit /b 1
)

if not exist "%~dp0videos\movies" mkdir "%~dp0videos\movies"
if not exist "%~dp0videos\narrations" mkdir "%~dp0videos\narrations"
if not exist "%~dp0videos\reference_audio" mkdir "%~dp0videos\reference_audio"
if not exist "%~dp0videos\subtitles" mkdir "%~dp0videos\subtitles"

if not exist "%~dp0backend\static\index.html" (
    if exist "%NODE_DIR%\node.exe" (
        cd /d "%~dp0frontend"
        if not exist node_modules call npm install
        call npm run build
        cd /d "%~dp0"
    )
)

cd /d "%~dp0backend"
"%PYTHON_EXE%" -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing backend requirements...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
)

echo ========================================
echo   Launching app...
echo   http://127.0.0.1:8000
echo ========================================
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"
"%PYTHON_EXE%" main.py
