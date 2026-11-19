@echo off
REM Install_Distill.bat -- 双击装 Distill Any Depth Base (ViTB), 权重找不到就自动下载
REM 用 venv_Easy3D 里的 python, 不要用系统 python
chcp 65001 >nul
cd /d "%~dp0"
if exist "venv_Easy3D\Scripts\python.exe" (
    set "PY=venv_Easy3D\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo [FAIL] 找不到 venv_Easy3D\Scripts\python.exe, 先确认 Easy3D 虚拟环境在不在
    pause
    exit /b 1
)
echo [INFO] Python: %PY%
echo [INFO] 权重找不到就自动下载 (约 390MB), 别关窗口
"%PY%" "%~dp0Install_Distill.py" %*
if errorlevel 1 (
    echo [FAIL] 安装失败, 看上面 / install_distill.log
) else (
    echo [OK] Distill 装好了, 再跑 patch_add_distill_gui_test_v105.py, 回 GUI 选 Distill-B
)
pause
