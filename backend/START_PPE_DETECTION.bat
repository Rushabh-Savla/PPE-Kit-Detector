@echo off
echo ==========================================
echo    PPE Safety Detection Service
echo    Based on: prodbykosta/ppe-safety-detection-ai
echo    Detects: Helmets and Safety Vests
echo ==========================================
echo.
echo Starting PPE Detection Service on port 8002...
echo.
echo Endpoints:
echo   - POST http://localhost:8002/detect - Upload image
echo   - POST http://localhost:8002/detect-frame - Base64 frame
echo   - GET  http://localhost:8002/status - Current status
echo   - WS   ws://localhost:8002/ws - Real-time WebSocket
echo.
echo Press Ctrl+C to stop the service
echo ==========================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate
python ppe_detection.py
pause
