@echo off
chcp 65001 >nul
call "%~dp0tools-env.bat"

echo [1/4] Checking Python...
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ not found
    pause
    exit /b 1
)

if not exist "%~dp0backend\.env" copy "%~dp0backend\.env.example" "%~dp0backend\.env" >nul

echo [2/4] Installing backend requirements...
cd /d "%~dp0backend"
"%PYTHON_EXE%" -m pip install torch==2.5.1
if errorlevel 1 (
    echo [ERROR] PyTorch install failed
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install scipy librosa webrtcvad-wheels typing
if errorlevel 1 (
    echo [ERROR] Voiceprint runtime dependency install failed
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install resemblyzer==0.1.4 --no-deps
if errorlevel 1 (
    echo [ERROR] Resemblyzer install failed
    pause
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Backend dependency install failed
    pause
    exit /b 1
)

echo [3/4] Installing frontend dependencies...
if exist "%NODE_DIR%\node.exe" (
    cd /d "%~dp0frontend"
    call npm install
) else (
    echo [WARN] Local Node not found, frontend install skipped
)

echo [4/4] Building frontend...
if exist "%NODE_DIR%\node.exe" (
    call npm run build
) else (
    echo [WARN] Local Node not found, frontend build skipped
)

echo.
echo Setup complete.
pause
