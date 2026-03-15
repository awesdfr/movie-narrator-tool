@echo off
chcp 65001 >nul
title 停止服务

echo 正在停止服务...

:: 停止Python进程（后端）
taskkill /f /im python.exe /fi "WINDOWTITLE eq 后端服务*" >nul 2>&1

:: 停止Node进程（前端）
taskkill /f /im node.exe /fi "WINDOWTITLE eq 前端服务*" >nul 2>&1

echo 服务已停止
timeout /t 2 >nul
