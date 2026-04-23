@echo off
echo ========================================
echo   FarmPulse - Diagnostic Startup
echo ========================================
echo.

echo Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python from https://python.org
    pause
    exit /b
)
echo Python found!
echo.

echo Installing required packages...
pip install uvicorn fastapi python-multipart pillow numpy opencv-python-headless scikit-learn joblib
echo.

echo ========================================
echo Starting Backend Server (port 8000)...
echo ========================================
echo.
cd /d c:\Users\ramat\Downloads\FarmPulse\backend
start "FarmPulse Backend" cmd /k "python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload || (echo. && echo BACKEND FAILED! See error above. && pause)"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo ========================================
echo Starting Frontend Server (port 8080)...
echo ========================================
echo.
cd /d c:\Users\ramat\Downloads\FarmPulse
start "FarmPulse Frontend" cmd /k "python -m http.server 8080 || (echo. && echo FRONTEND FAILED! See error above. && pause)"

timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   DONE! Opening browser...
echo   Frontend: http://localhost:8080
echo   Backend:  http://localhost:8000/docs
echo ========================================
echo.
echo DO NOT close the two black windows!
echo.
start http://localhost:8080
pause
