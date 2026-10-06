@echo off
title SENTINEL NIDS

cd /d "%~dp0"

echo.
echo ==================================================
echo             SENTINEL NIDS
echo      AI Network Intrusion Detection System
echo ==================================================
echo.
echo Project directory:
echo %CD%
echo.

REM ==================================================
REM USE PROJECT VIRTUAL ENVIRONMENT
REM ==================================================

set PYTHON=%CD%\venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo.
    echo ERROR: Virtual environment not found!
    echo Expected:
    echo %PYTHON%
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking project Python...
"%PYTHON%" --version

echo.
echo [2/4] Checking Scapy...
"%PYTHON%" -c "import scapy; print('Scapy OK')"

if errorlevel 1 (
    echo.
    echo ERROR: Scapy is not installed in the project venv.
    echo.
    echo Run:
    echo venv\Scripts\python.exe -m pip install scapy
    echo.
    pause
    exit /b 1
)

echo.
echo [3/4] Checking Streamlit...
"%PYTHON%" -c "import streamlit; print('Streamlit OK')"

if errorlevel 1 (
    echo.
    echo ERROR: Streamlit is not installed in the project venv.
    echo.
    echo Run:
    echo venv\Scripts\python.exe -m pip install streamlit
    echo.
    pause
    exit /b 1
)

echo.
echo [4/4] Starting SOC Dashboard...
echo.

start "Sentinel NIDS Dashboard" cmd /k "cd /d ""%CD%"" && ""%PYTHON%"" -m streamlit run src\dashboard.py"

timeout /t 3 /nobreak >nul

echo.
echo ==================================================
echo Dashboard started.
echo http://localhost:8501
echo ==================================================
echo.

echo Starting live detection engine...
echo.
echo IMPORTANT:
echo This terminal should be running as ADMINISTRATOR.
echo Press CTRL+C to stop the detector.
echo.

"%PYTHON%" src\realtime_detection.py

echo.
echo ==================================================
echo Detection engine stopped.
echo ==================================================
pause