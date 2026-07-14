@echo off
chcp 65001 >nul
echo ==========================================
echo    ITMO Stars EEG Analysis
echo ==========================================
echo.

REM Проверка виртуального окружения
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
pip install -r requirements.txt

echo.
echo [INFO] Starting API server...
start "EEG API" cmd /k "uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] Waiting for API to start...
timeout /t 3 /nobreak >nul

echo.
echo [INFO] Starting Streamlit dashboard...
start "EEG Dashboard" cmd /k "streamlit run dashboard.py"

echo.
pause