@echo off
chcp 65001 >nul
call "%~dp0tools-env.bat"
cd /d "%~dp0backend"

echo ========================================
echo   Backend starting...
echo   URL: http://127.0.0.1:8000
echo ========================================
echo.

"%PYTHON_EXE%" main.py
pause
