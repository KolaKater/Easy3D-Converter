@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
REM ================================================================
REM Easy3D Converter - update.bat v2 (self-update safe)
REM Repo: https://github.com/KolaKater/Easy3D-Converter
REM Usage: double-click update.bat in your Easy3D folder.
REM
REM FIX: A running .bat is read line-by-line by Windows. If we overwrite
REM update.bat while it is running (git pull / ZIP copy), the rest of the
REM script gets garbled. So: on first run, copy self to %TEMP% and re-run
REM from there. The original file in the repo folder can then be safely
REM overwritten. The %TEMP% copy deletes itself at the end.
REM ================================================================

REM --- Self-relocation: if we are NOT already in %TEMP%, copy and re-run from %TEMP% ---
REM Marker: _EASY3D_UPDATE_RUNNING=1 means we are already the TEMP copy.
if not "%_EASY3D_UPDATE_RUNNING%"=="1" (
    set "_EASY3D_UPDATE_RUNNING=1"
    set "_EASY3D_REPO_DIR=%~dp0"
    REM Remove trailing backslash for later use
    copy /Y "%~f0" "%TEMP%\Easy3D-update-run.bat" >nul
    REM Re-run from TEMP, passing repo dir as %1. Keep same window.
    set "_EASY3D_UPDATE_RUNNING=1"
    call "%TEMP%\Easy3D-update-run.bat" "!_EASY3D_REPO_DIR!"
    set "_EXITCODE=!errorlevel!"
    REM TEMP copy deletes itself (the call already returned, file is not locked)
    del /F /Q "%TEMP%\Easy3D-update-run.bat" >nul 2>nul
    exit /b !_EXITCODE!
)

REM --- We are now the TEMP copy. %1 = repo dir ---
set "REPO_DIR=%~1"
if "%REPO_DIR%"=="" set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"
title Easy3D Converter - Update
echo ================================================
echo  Easy3D Converter - Update from GitHub
echo  Repo: https://github.com/KolaKater/Easy3D-Converter
echo  Folder: %CD%
echo ================================================
echo.

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [Update] git not found, trying ZIP update (no git needed)...
    goto :ZIP_UPDATE
)

if exist ".git" (
    echo [Update] Found .git, using git pull...
    echo.
    echo [Git] Current branch / status:
    git branch --show-current
    git status --short
    echo.
    echo [Git] Fetching from GitHub...
    git fetch origin
    if !errorlevel! neq 0 (
        echo [Error] git fetch failed. Check network / GitHub.
        echo Try: open https://github.com/KolaKater/Easy3D-Converter
        pause
        exit /b 1
    )
    git diff --quiet
    if !errorlevel! neq 0 (
        echo [Update] You have local changes, stashing them first (not lost)...
        git stash push -m "Easy3D auto-stash before update %date% %time%"
        set "DID_STASH=1"
    ) else (
        set "DID_STASH=0"
    )
    echo [Git] Pulling...
    git pull --ff-only
    if !errorlevel! neq 0 (
        echo [Warn] Fast-forward pull failed (maybe local commits). Trying rebase...
        git pull --rebase
        if !errorlevel! neq 0 (
            echo [Error] Pull failed. Your changes are in stash, nothing lost.
            echo Run: git stash list
            echo To restore: git stash pop
            pause
            exit /b 1
        )
    )
    if "!DID_STASH!"=="1" (
        echo [Update] Restoring your local changes...
        git stash pop
        if !errorlevel! neq 0 (
            echo [Warn] stash pop had conflicts. Your changes are still in stash.
            echo Run: git stash list
            echo Run: git stash pop after fixing conflicts
        )
    )
    echo.
    echo ================================================
    echo  [Done] Updated to latest version!
    echo  update.bat itself was also updated (safe, ran from TEMP copy).
    echo  Run: start Launch_Easy3D_GUI.bat 
    echo ================================================
    git log --oneline -3
    pause
    exit /b 0
)

:ZIP_UPDATE
echo [Update] No .git found (you downloaded ZIP, not git clone).
echo [Update] Downloading latest ZIP from GitHub...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$zip='%TEMP%\Easy3D-Converter-main.zip';" ^
  "$url='https://github.com/KolaKater/Easy3D-Converter/archive/refs/heads/main.zip';" ^
  "Write-Host \"Downloading $url ...\";" ^
  "try { Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing } catch { curl.exe -L -o $zip $url };" ^
  "if (-not (Test-Path $zip)) { Write-Host '[Error] Download failed'; exit 1 };" ^
  "Write-Host \"Downloaded: $zip\";" ^
  "$dest='%TEMP%\Easy3D-Update-Tmp';" ^
  "if (Test-Path $dest) { Remove-Item -Recurse -Force $dest };" ^
  "Expand-Archive -Path $zip -DestinationPath $dest -Force;" ^
  "$src = Get-ChildItem $dest | Select-Object -First 1 | Select-Object -ExpandProperty FullName;" ^
  "Write-Host \"Extracted to: $src\";" ^
  "Write-Host 'Copying new files (preserving your local data)...';" ^
  "Get-ChildItem $src -Force | ForEach-Object {" ^
  "  $name = $_.Name;" ^
  "  if ($name -in @('venv_Easy3D','venv','models','gfpgan','checkpoints','vis','_cache','_tmp_square','.git','__pycache__')) { Write-Host \"  Skip (your local): $name\"; return };" ^
  "  if ($name -eq 'preset') { Write-Host '  Merge preset/ (do not overwrite your presets)...'; if (-not (Test-Path 'preset')) { New-Item -ItemType Directory -Path 'preset' | Out-Null }; Copy-Item -Path (Join-Path $_.FullName '*') -Destination 'preset' -Recurse -Force -ErrorAction SilentlyContinue; return };" ^
  "  Write-Host \"  Update: $name\";" ^
  "  if ($_.PSIsContainer) { Copy-Item -Path $_.FullName -Destination '.' -Recurse -Force } else { Copy-Item -Path $_.FullName -Destination '.' -Force };" ^
  "};" ^
  "Write-Host 'Cleaning temp...';" ^
  "Remove-Item -Recurse -Force $dest; Remove-Item -Force $zip;"
if %errorlevel% neq 0 (
    echo.
    echo [Error] ZIP update failed.
    echo Manual: open https://github.com/KolaKater/Easy3D-Converter
    echo Click Code ^> Download ZIP, unzip, copy over (keep models/ venv_Easy3D/ preset/).
    pause
    exit /b 1
)
echo.
echo ================================================
echo  [Done] Updated to latest version (ZIP mode)!
echo  Your models/ venv_Easy3D/ preset/ were NOT overwritten.
echo  update.bat itself was also updated (safe, ran from TEMP copy).
echo  Run: start Launch_Easy3D_GUI.bat
echo ================================================
pause
exit /b 0
