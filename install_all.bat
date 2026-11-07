@echo off
REM install_all.bat - One-click: venv + pip (incl. tkinterdnd2) + ALL 10 models
REM RELATIVE PATHS ONLY. No absolute path. Place this bat + install_all.py in Easy3D folder, double-click.
REM cd to this bat's folder, then run install_all.py with venv python if it exists, else system python.
cd /d "%~dp0"
if exist "venv_Easy3D\Scripts\python.exe" ( set "PY=venv_Easy3D\Scripts\python.exe" ) else ( set "PY=python" )
echo [install_all] Using %PY% in %CD%
"%PY%" install_all.py
echo.
echo [DONE] Check FINAL CHECK above: 10/10 OK = all models present.
pause
