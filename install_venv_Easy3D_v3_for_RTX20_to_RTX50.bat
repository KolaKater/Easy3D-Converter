@echo off
REM install_venv_Easy3D_v3.bat - Auto-detect GPU (RTX 3070 and above), install matching torch CUDA
REM RELATIVE PATHS ONLY. Place this bat + install_venv_Easy3D_v3.py in Easy3D folder, double-click.
cd /d "%~dp0"
if exist "venv_Easy3D\Scripts\python.exe" ( set "PY=venv_Easy3D\Scripts\python.exe" ) else ( set "PY=python" )
echo [install_venv] Using %PY% in %CD%
"%PY%" install_venv_Easy3D_for_RTX20_to_RTX50.py
pause
