@echo off
REM Easy3D V10.8.2 - INSTALL FIX - ASCII only, no Chinese, no brackets in echo
REM Run this in the Easy3D folder. It uses the SAME python for pip and for the GUI.
cd /d %~dp0
echo Easy3D-Fix: Python used for pip and GUI:
python --version
echo Easy3D-Fix: Step 1 - check env
python check_env.py
echo.
echo Easy3D-Fix: Step 2 - install missing packages, torch NOT touched
echo Easy3D-Fix: Press any key to start, Ctrl+C to cancel
pause >nul
python -m pip install --upgrade pip
python -m pip install -r requirements_easy3d.txt
echo.
echo Easy3D-Fix: Step 3 - check again
python check_env.py
echo.
echo Easy3D-Fix: If all OK, start GUI with the SAME python:
echo Easy3D-Fix: python Easy3D_GUI_GenStereo.py
echo Easy3D-Fix: Do NOT double-click the py file, Windows may pick another Python.
pause
