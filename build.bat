@echo off
chcp 65001 >nul
call "%~dp0tools-env.bat"
cd /d "%~dp0frontend"
if not exist node_modules call npm install
call npm run build
pause
