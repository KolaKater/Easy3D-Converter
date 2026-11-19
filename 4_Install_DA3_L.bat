@echo off
REM Install_DA3_L.bat -- 双击装 DA3-L (Large) + 检查/自动下载权重
REM 用 venv_Easy3D 的 Python, 不用系统 Python. 相对路径, 不要写死盘符.
setlocal
cd /d "%~dp0"
if exist "venv_Easy3D\Scripts\python.exe" (
    set PY=venv_Easy3D\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PY=venv\Scripts\python.exe
) else (
    set PY=python
)
echo [INFO] Python: %PY%
echo [INFO] Folder: %CD%
echo [INFO] 正在装 DA3-L 代码 + 修复 numpy/triton/matplotlib + 自动下载 DA3-LARGE 权重 (几个 GB, 别关窗口)...
"%PY%" Install_DA3_L.py
if errorlevel 1 (
    echo.
    echo [FAIL] 有一个没好, 把 install_da3_l.log 最后 30 行贴回来.
) else (
    echo.
    echo [OK] DA3-L 都好了, 回 GUI 选 DA3-L 再跑.
)
pause
