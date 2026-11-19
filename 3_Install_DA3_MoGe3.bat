@echo off
REM Install_DA3_MoGe3.bat v4 -- 双击装 DA3 + MoGe3 (相对路径, 不写死绝对路径)
REM %~dp0 = 这个 BAT 所在文件夹, 先 cd 到那里, 再用它下面的 venv_Easy3D 装.
REM 不要用系统 Python, GUI 用的是 venv_Easy3D 里的 Python, 装到系统 Python 里 test.py 还是找不到.
cd /d "%~dp0"
echo [BAT] Folder: %CD%
echo [BAT] Installing DA3 + MoGe3 into venv_Easy3D ...
if exist "venv_Easy3D\Scripts\python.exe" (
    echo [BAT] Using: venv_Easy3D\Scripts\python.exe
    "venv_Easy3D\Scripts\python.exe" "Install_DA3_MoGe3.py"
) else if exist "venv\Scripts\python.exe" (
    echo [BAT] Using: venv\Scripts\python.exe
    "venv\Scripts\python.exe" "Install_DA3_MoGe3.py"
) else (
    echo [BAT] WARN: venv_Easy3D not found, fallback to system python
    python "Install_DA3_MoGe3.py"
)
echo.
echo [BAT] Done. Log: install_da3_moge3.log (same folder)
echo [BAT] If FAIL, paste last 30 lines of install_da3_moge3.log
pause
