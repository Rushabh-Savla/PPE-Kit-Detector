@echo off
title Gas Sensor Bridge
color 0B

echo ========================================
echo   Gas Sensor Bridge Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run this from the backend directory
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install aiohttp if needed
echo [2/3] Checking dependencies...
pip show aiohttp >nul 2>&1
if errorlevel 1 (
    echo Installing aiohttp...
    pip install aiohttp
)

REM Run the bridge
echo [3/3] Starting gas sensor bridge...
echo.
echo ========================================
echo   Bridge is running!
echo   Press Ctrl+C to stop
echo ========================================
echo.

python gas_bridge.py

pause
