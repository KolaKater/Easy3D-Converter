@echo off
REM Easy3D V10.9 - CHECK ENV - ASCII only
REM Run this in the Easy3D folder. It checks the venv with the SAME python that runs the GUI.
REM Use it after 1-4 install, and any time the GUI fails to start or reports a missing package.
REM It does NOT install anything. To fix a broken venv, run INSTALL_FIX.bat.
cd /d %~dp0
if not exist "venv\Scripts\python.exe" (
  echo Easy3D-Check: venv not found.
  echo Easy3D-Check: Run 1_MAKE_VENV.bat first.
  pause
  exit /b 1
)
if not exist "check_env.py" (
  echo Easy3D-Check: check_env.py not found in this folder.
  pause
  exit /b 1
)
echo Easy3D-Check: Python used for GUI:
venv\Scripts\python.exe --version
echo.
echo Easy3D-Check: Checking env...
venv\Scripts\python.exe check_env.py
echo.
echo Easy3D-Check: What to look for:
echo Easy3D-Check: torch 2.6.0+cu124 cuda: True = OK on RTX 3070.
echo Easy3D-Check: cuda: False = CPU torch, delete venv and re-run 1_MAKE_VENV.bat.
echo Easy3D-Check: Missing package = run INSTALL_FIX.bat.
pause
