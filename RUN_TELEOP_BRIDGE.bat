@echo off
chcp 65001 >nul
title DexDenso Teleoperation Bridge
cls
echo ======================================================================
echo    DexDenso: DENSO RC8 & Isaac Sim Real-Time Teleoperation
echo ======================================================================
echo.
echo [1] Releasing stale sockets and camera handles...
taskkill /F /IM python.exe 2>nul
timeout /t 1 /nobreak >nul
echo.
echo [2] Initializing DexDenso Vision Teleoperation System...
python main.py

echo.
echo Teleoperation process terminated.
pause
