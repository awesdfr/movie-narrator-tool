@echo off
chcp 65001 >nul
title React 前端服务 - 电影解说工具

cd /d "%~dp0frontend-react"

echo ========================================
echo   React 前端服务启动中...
echo   地址: http://localhost:3001
echo ========================================
echo.

npm install
npm run dev

pause

