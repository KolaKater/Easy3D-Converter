@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Easy3D Converter - Update
echo ================================================
echo  Easy3D Converter - Update from GitHub
echo  Repo: https://github.com/KolaKater/Easy3D-Converter
echo  Folder: %CD%
echo ================================================
echo.

REM ---------- 1) Check git ----------
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [Update] git not found, trying ZIP update (no git needed)...
    goto :ZIP_UPDATE
)

REM ---------- 2) If .git exists -> git pull ----------
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
    REM If local changes would block pull, stash them first
    git diff --quiet
    if !errorlevel! neq 0 (
        echo [Update] You have local changes, stashing them first (not lost)...
        git stash push -m "Easy3D auto-stash before update %date% %time%"
        set "DID_STASH=1"
    ) else (
        set "DID_STASH=0"
    )
    REM Also stash untracked? No - untracked files (your own images) are never touched by pull.
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
            echo Run: git stash list  ^> to see them
            echo Run: git stash pop  ^> after fixing conflicts
        )
    )
    echo.
    echo ================================================
    echo  [Done] Updated to latest version!
    echo  Run: start Easy3D_GUI_GenStereo.py (or your launcher)
    echo ================================================
    git log --oneline -3
    pause
    exit /b 0
)

REM ---------- 3) No .git -> ZIP update ----------
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
  "$skip = @('venv_Easy3D','venv','models','gfpgan','checkpoints','vis','_cache','_tmp_square','.git','__pycache__','*.log','easy3d_debug.log');" ^
  "Get-ChildItem $src -Force | ForEach-Object {" ^
  "  $name = $_.Name;" ^
  "  if ($name -in @('venv_Easy3D','venv','models','gfpgan','checkpoints','vis','_cache','_tmp_square','.git','__pycache__')) { Write-Host \"  Skip (your local): $name\"; return };" ^
  "  if ($name -eq 'preset') { Write-Host '  Merge preset/ (do not overwrite your presets)...'; Copy-Item -Path (Join-Path $_.FullName '*') -Destination 'preset' -Recurse -Force -ErrorAction SilentlyContinue; return };" ^
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
echo  Run: start Launch_Easy3D_GUI.bat
echo ================================================
pause
exit /b 0
