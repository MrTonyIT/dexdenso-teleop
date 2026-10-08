@echo off
chcp 65001 >nul
title DexDenso Teleop - Hardware & Port Reset
cls
echo ======================================================================
echo    DexDenso: Vision & Network Port Reset Utility
echo ======================================================================
echo.
echo Terminating lingering Python processes and releasing camera handles...
taskkill /F /IM python.exe 2>nul
echo.
echo [OK] Teleoperation hardware and socket ports successfully reset.
echo You can now launch RUN_TELEOP_BRIDGE.bat.
echo.
pause
