@echo off
title Sentinel NIDS - Live Detection

cd /d "%~dp0"

echo.
echo ==========================================
echo       SENTINEL NIDS - DETECTION ENGINE
echo ==========================================
echo.
echo Starting live network detection...
echo.
echo IMPORTANT:
echo This window may require Administrator privileges.
echo Press CTRL+C to stop detection.
echo.

python src\realtime_detection.py

echo.
echo ==========================================
echo Detection engine stopped.
echo ==========================================
pause