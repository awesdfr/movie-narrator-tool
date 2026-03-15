@echo off
chcp 65001 >nul
call "%~dp0tools-env.bat"
cd /d "%~dp0frontend"

echo ========================================
echo   Frontend dev server starting...
echo   URL: http://127.0.0.1:3000
echo ========================================
echo.

npm run dev
pause
