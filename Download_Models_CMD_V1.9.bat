@echo off
REM Download_Models_CMD_V1.9.bat - Easy3D model downloader
REM VERSION: V1.9 - FULL LIST, 34 files, same as V1.7 Python list
REM FIX V1.9: 1) self-elevate to Admin, else Program Files is Zugriff verweigert
REM FIX V1.9: 2) local path: / -> \, else curl --create-dirs fails on mixed slashes
REM FIX V1.9: 3) mkdir with backslashes only, curl does NOT create dirs
REM Old way: BAT only opened CMD + called PY, PY did mkdir + curl + skip + MB/s.
REM V1.9 keeps BAT-only, but with the same mkdir logic PY used.
setlocal
REM --- Admin check: if not admin, re-launch as admin and exit ---
net session >nul 2>&1
if errorlevel 1 (
  echo Requesting Admin rights for Program Files...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b 0
)
set "BASE=https://huggingface.co/KolaKater/Easy3D-models/resolve/main"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
echo Models dir: %ROOT%
echo Total: 34 files, from KolaKater/Easy3D-models only
echo Running as Admin, local paths use backslash only.
echo.
call :DL "gfpgan/weights/GFPGANv1.4.pth"
call :DL "gfpgan/weights/detection_Resnet50_Final.pth"
call :DL "gfpgan/weights/parsing_parsenet.pth"
call :DL "models/depth_anything_v2_vitb.pth"
call :DL "models/depth_anything_v2_vitl.pth"
call :DL "models/depth_anything_v2_vits.pth"
call :DL "models/distill_any_depth_vitb.safetensors"
call :DL "models/distill_any_depth_vits.safetensors"
call :DL "models/FlashDepth/flashdepth_infer_image.py"
call :DL "models/FlashDepth/flashdepth_infer_video.py"
call :DL "models/FlashDepth/configs/flashdepth-l/config.yaml"
call :DL "models/FlashDepth/configs/flashdepth-l/iter_10001.pth"
call :DL "models/FlashDepth/configs/flashdepth-s/config.yaml"
call :DL "models/FlashDepth/configs/flashdepth-s/iter_14001.pth"
call :DL "models/da3-base/config.json"
call :DL "models/da3-base/model.safetensors"
call :DL "models/da3-large/config.json"
call :DL "models/da3-large/model.safetensors"
call :DL "models/da3-small/config.json"
call :DL "models/da3-small/model.safetensors"
call :DL "models/genstereo-v1.5/config.json"
call :DL "models/genstereo-v1.5/denoising_unet.pth"
call :DL "models/genstereo-v1.5/fusion_layer.pth"
call :DL "models/genstereo-v1.5/pose_guider.pth"
call :DL "models/genstereo-v1.5/reference_unet.pth"
call :DL "models/genstereo-v2.1/config.json"
call :DL "models/genstereo-v2.1/denoising_unet.pth"
call :DL "models/genstereo-v2.1/fusion_layer.pth"
call :DL "models/genstereo-v2.1/pose_guider.pth"
call :DL "models/genstereo-v2.1/reference_unet.pth"
call :DL "models/image_encoder/config.json"
call :DL "models/image_encoder/pytorch_model.bin"
call :DL "models/sd-vae-ft-mse/config.json"
call :DL "models/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"
echo.
echo DONE. Press any key to close.
pause >nul
exit /b 0
:DL
REM %~1 = URL path with forward slashes, e.g. gfpgan/weights/GFPGANv1.4.pth
set "URLREL=%~1"
set "LOCALREL=%~1"
set "LOCALREL=%LOCALREL:/=\%"
echo [DL] %URLREL%
for %%D in ("%ROOT%\%LOCALREL%") do set "DIR=%%~dpD"
if not exist "%DIR%" mkdir "%DIR%"
if errorlevel 1 (echo [FAIL] cannot mkdir %DIR% & echo. & exit /b 0)
if exist "%ROOT%\%LOCALREL%" del "%ROOT%\%LOCALREL%"
curl -f -L -C - --retry 3 --retry-delay 5 -o "%ROOT%\%LOCALREL%" "%BASE%/%URLREL%"
if errorlevel 1 (
  echo [FAIL] %URLREL%
) else (
  echo [OK] %URLREL%
)
echo.
exit /b 0
