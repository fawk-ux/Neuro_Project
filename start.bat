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
    echo [INFO] Installing dependencies for the first time...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo.
echo [INFO] Starting API server in background window...
start "EEG API" cmd /k "call .venv\Scripts\activate.bat && uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] Waiting for API to start...
timeout /t 3 /nobreak >nul

echo.
echo [INFO] Starting Streamlit dashboard...
streamlit run dashboard.py

echo.
pause