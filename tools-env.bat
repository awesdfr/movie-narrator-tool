@echo off
setlocal
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

set "PYTHON_EXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

set "NODE_DIR=%REPO_DIR%\.tools\node-v20.19.0-win-x64"
if exist "%NODE_DIR%\node.exe" set "PATH=%NODE_DIR%;%PATH%"

set "GIT_BIN="
if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT_BIN=%ProgramFiles%\Git\cmd"
if not defined GIT_BIN if exist "%ProgramFiles%\Git\bin\git.exe" set "GIT_BIN=%ProgramFiles%\Git\bin"
if not defined GIT_BIN if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "GIT_BIN=%LOCALAPPDATA%\Programs\Git\cmd"
if defined GIT_BIN set "PATH=%GIT_BIN%;%PATH%"

for /d %%D in ("%REPO_DIR%\.tools\ffmpeg-*essentials_build") do (
    if exist "%%~fD\bin\ffmpeg.exe" (
        set "FFMPEG_BIN=%%~fD\bin"
        goto ffmpeg_done
    )
)
:ffmpeg_done
if defined FFMPEG_BIN set "PATH=%FFMPEG_BIN%;%PATH%"

endlocal & (
    set "REPO_DIR=%REPO_DIR%"
    set "PYTHON_EXE=%PYTHON_EXE%"
    if defined NODE_DIR set "NODE_DIR=%NODE_DIR%"
    if defined GIT_BIN set "GIT_BIN=%GIT_BIN%"
    if defined FFMPEG_BIN set "FFMPEG_BIN=%FFMPEG_BIN%"
    set "PATH=%PATH%"
)
