@echo off
REM Easy3D - GenStereo GUI Launcher (waits for GUI, hidden CMD)
REM Based on Launch_Easy3D_GUI.bat - adapted for GenStereo

if /I "%~1"=="-hidden" goto :MAIN
set "VBS=%TEMP%\Easy3D_HideLaunch.vbs"
> "%VBS%" echo Set WshShell = CreateObject("WScript.Shell")
>>"%VBS%" echo WshShell.Run Chr(34) ^& "%~f0" ^& Chr(34) ^& " -hidden", 0
>>"%VBS%" echo Set WshShell = Nothing
wscript //nologo "%VBS%"
del /f /q "%VBS%" >nul 2>&1
exit /b

:MAIN
chcp 65001 >nul
cd /d "%~dp0"

if exist "venv_Easy3D\Scripts\pythonw.exe" (
    set PYTHONW=venv_Easy3D\Scripts\pythonw.exe
    set PYTHON=venv_Easy3D\Scripts\python.exe
) else (
    set PYTHONW=pythonw
    set PYTHON=python
)

if not exist "Easy3D_GUI_GenStereo.py" (
    echo [ERROR] Easy3D_GUI_GenStereo.py not found
    echo Put this BAT in D:\3D\Easy3D\ next to Easy3D_GUI_GenStereo.py
    pause
    exit /b
)

if not exist "output_type.py" (
    echo [WARNING] output_type.py not found - Output Type Auto-Align will be skipped
    timeout /t 2 /nobreak >nul
)

if not exist "Easy3D.ico" (
    echo [WARNING] Easy3D.ico not found - GUI will start without icon
    timeout /t 2 /nobreak >nul
)

echo Starting Easy3D GenStereo GUI, please wait...
REM Clear old log
echo. > easy3d_debug.log

REM Try pythonw first (hidden CMD)
start "" "%PYTHONW%" "Easy3D_GUI_GenStereo.py"

REM Wait 3 seconds for GUI to open, then close this launcher
REM GUI will stay alive because it's separate process
timeout /t 3 /nobreak >nul

REM Check if GUI process is running
tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | find /I "pythonw.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
)

REM Check log file - if GUI crashed, log will contain error
if exist "easy3d_debug.log" (
    for %%A in ("easy3d_debug.log") do if %%~zA GTR 10 (
        echo GUI started, closing launcher...
        timeout /t 1 /nobreak >nul
        exit
    )
)

REM If log still empty, GUI may have failed - try with console to show error
echo GUI did not start with pythonw, trying with console to show error...
"%PYTHON%" "Easy3D_GUI_GenStereo.py"
pause
