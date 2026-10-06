@echo off
title Sentinel NIDS - SOC Dashboard

cd /d "%~dp0"

echo.
echo ==========================================
echo        SENTINEL NIDS - SOC DASHBOARD
echo ==========================================
echo.
echo Starting dashboard...
echo.
echo Open your browser at:
echo http://localhost:8501
echo.

streamlit run src\dashboard.py

pause