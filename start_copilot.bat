@echo off
title Meeting Copilot
cd /d "%~dp0"

echo ========================================================
echo               MEETING COPILOT LAUNCHER
echo ========================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0venv
    echo Please create it first: python -m venv venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Meeting Copilot...
python launch_copilot.py

pause

