@echo off
REM Launch the NaviGaze processing platform.
REM Double-click this file, or run it from a terminal.

setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found at "%PYTHON%".
    echo Create it first:  py -3.12 -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting NaviGaze platform...
echo Open http://127.0.0.1:5000 in your browser  (press Ctrl+C here to stop).
echo.

REM Open the browser shortly after the server starts.
start "" /b cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5000"

"%PYTHON%" "%~dp0app\server.py"

echo.
echo Server stopped.
pause
