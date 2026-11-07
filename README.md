# Easy3D Converter

**Convert a single image into a side-by-side (SBS) stereo 3D image.**

Easy3D Converter is a free, open-source Tkinter GUI for [GenStereo (ICCV 2025)](https://github.com/FQiao/GenStereo) — a diffusion-based single-image-to-stereo model. It is designed to run on a consumer GPU such as an **NVIDIA RTX 3070 8GB**, clean file naming, and batch conversion.

[[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[[Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[[GPU: RTX 3070+](https://img.shields.io/badge/GPU-RTX%203070%2B-brightgreen.svg)](https://www.nvidia.com/)

<img width="1275" height="844" alt="图片" src="https://github.com/user-attachments/assets/c7236244-1795-440f-adee-42a301965283" />
<img width="3070" height="1600" alt="1793898215341-done-photo-reality_Easy3D_V21_FDL_R1024_D100_B040_F030_S20_G181_FRBoth_PAR_Seed402260655_Aligned_Cut" src="https://github.com/user-attachments/assets/82054335-fa0c-4c07-a6aa-9cd1fdce0240" />

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone / Download](#1-clone--download)
  - [2. Create the Virtual Environment](#2-create-the-virtual-environment)
  - [3. Download All Models](#3-download-all-models)
  - [4. Verify](#4-verify)
- [Update](#update)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Input / Output](#input--output)
  - [Parameters](#parameters)
  - [Output Naming](#output-naming)
  - [Output Type](#output-type)
- [Models & Components](#models--components)
  - [Model List](#model-list)
  - [Programs & Libraries](#programs--libraries)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Credits & License](#credits--license)

---

## Features

- **Single image → SBS 3D:** Left = original, Right = AI-generated (or symmetric / swapped, see Gen Mode).
- **Batch conversion:** Drop a single image or a whole folder. The model is loaded **once** for the whole batch (no reload per image).
- **Streaming output:** Each image is converted to SBS as soon as its inference finishes — you don't wait for the whole batch.
- **Drag & drop:** Drop a file / folder onto the Input box or the whole window (`tkinterdnd2`).
- **Presets:** Save / Load / Delete parameter presets. Config is auto-saved to `preset/easy3d_gui_config.json`.
- **Depth / Warped export:** Optional `_DEPTH` and `_WARPED` outputs.
- **Face Restore (optional):** GFPGAN on the merged SBS image, both eyes.
- **Live preview:** Full-resolution preview that follows the window size.
- **Clean naming:** `Original_Easy3D_V15_DA2L_R1024_D100_B100_F050_S50_G150_PAR_Seed12345.jpg` — all parameters are in the filename.
- **RTX 3070 friendly:** v1.5 / 512–1024 px is the recommended setting. 1536 px+ is possible but slow / may OOM on 8 GB.
- **One-click update:** Double-click `update.bat` to update to the latest version from GitHub. Your `models/`, `venv_Easy3D/`, `preset/` are NOT overwritten.

---

## How It Works

```
Original (W x H, any aspect)
  → GenStereo (Depth + Warp + Diffusion, single pass)
  → Right view (Res x Res)
  → Merge Left | Right → SBS
  → (Optional) GFPGAN Face Restore on SBS
  → (Optional) Output Type: Parallel / Cross-Eyed / Auto-Align
  → Save as JPG, quality 95, subsampling=0
```

GenStereo itself is a dual-UNet diffusion model:

- **Reference UNet** copies appearance / lighting from the original image.
- **Denoising UNet** generates the right view.
- **Warp** uses a depth map to warp the left view to the right view (true lighting / details, but with holes).
- **AdaptiveFusion** mixes Warped + Generated to fill holes.

Depth comes from **Depth Anything V2** (default) or **FlashDepth** (optional). See [Models](#models--components).

---

## Requirements

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / 11 (64-bit) | Windows 11 |
| Python | 3.11.9 | 3.11.9 |
| GPU | NVIDIA RTX 2060 6GB | **RTX 3070 8GB** or above (4070 / 4090 / 5070 / 5090) |
| VRAM | 6 GB (512 px, v1.5) | 8 GB+ (1024 px, v1.5) |
| Disk | ~25 GB (venv + all models) | ~30 GB |
| CUDA | Driver with CUDA 12.4+ | Latest NVIDIA driver |

> **Note:** Training is **not** supported on a consumer GPU (GenStereo training needs A100-scale hardware). This project is inference only.

---

## Installation

All paths below are **relative** to the Easy3D folder. No absolute paths are hard-coded. Copy the folder anywhere and run.

### 1. Clone / Download

```bash
git clone https://github.com/KolaKater/Easy3D-Converter.git
cd Easy3D-Converter
```

Or download the ZIP and extract it. Do **not** use a path with Chinese characters or parentheses if you use the `.bat` launchers — `C:\3D\Easy3D\` is a safe example.

### 2. Create the Virtual Environment

This step auto-detects your GPU and installs the matching PyTorch CUDA build:

| GPU | Installed Torch |
|---|---|
| RTX 3070 / 3080 / 3090 (Ampere) | `torch 2.6.0+cu124` |
| RTX 4070 / 4080 / 4090 (Ada) | `torch 2.6.0+cu124` |
| RTX 5070 / 5080 / 5090 (Blackwell) | `torch 2.7.1+cu128` |
| RTX 2060 / 2070 / 2080 (Turing) | `torch 2.6.0+cu124` |
| No NVIDIA GPU | `torch 2.6.0` CPU (slow, testing only) |

**Double-click:**

```
install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat
```

Or from a terminal:

```bash
python install_venv_Easy3D_v3_for_RTX20_to_RTX50.py
```

This creates `venv_Easy3D\`, installs PyTorch **first** (so it is not pulled as CPU-only), then installs `numpy`, `pillow`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `einops`, `omegaconf`, `scipy`, `opencv-python`, `timm`, `gradio`, `tkinterdnd2`, `xformers`, and the GFPGAN stack (`gfpgan`, `basicsr`, `facexlib`, `realesrgan`).

> To force a clean rebuild, delete the `venv_Easy3D\` folder and run the `.bat` again.

### 3. Download All Models

**Double-click:**

```
install_all.bat
```

Or:

```bash
venv_Easy3D\Scripts\python.exe install_all.py
```

This downloads all 10 model targets (see [Model List](#model-list)), skips any file that already exists and is not empty, and never downloads to `checkpoints/` (legacy folder — all models live in `models/`).

Final check should read:

```
10/10 OK
```

If it does not, see the `MISSING` lines and download those files manually to the same relative paths.

### 3. 1 Download FFmpeg
FFmpeg: Download from https://www.gyan.dev/ffmpeg/builds/ 
and put ffmpeg.exe into ffmpeg/bin/

### 4. Verify

```bash
venv_Easy3D\Scripts\python.exe -c "import torch, tkinterdnd2; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

Expected on RTX 3070:

```
torch: 2.6.0+cu124 cuda: True
```

---

## Update

Keep Easy3D Converter up to date. No need to re-download the whole ZIP or re-install models / venv.

**Double-click:**

```
update.bat
```

That's it. Wait for `[Done] Updated to latest version!`, then double-click `Launch_Easy3D_GUI.bat` again.

### What it does

| Your install | How it updates |
|---|---|
| `git clone` (has `.git`) | `git fetch` + `git pull --ff-only`. If you changed files locally, it `git stash`es them first, pulls, then `git stash pop`s them back. |
| ZIP download (no `.git`, or no `git` installed) | Downloads `https://github.com/KolaKater/Easy3D-Converter/archive/refs/heads/main.zip`, extracts it, copies new files over. |

### What is NOT overwritten

- `models/` — all your downloaded weights (GenStereo / VAE / MDE / FlashDepth)
- `venv_Easy3D/` / `venv/` — your Python environment
- `preset/` — your presets + `easy3d_gui_config.json` (merged, not overwritten)
- `vis/` / `_cache/` / `*.log` — your temp / logs

`update.bat` itself is also updated. It is safe to update itself: on first run it copies itself to `%TEMP%` and re-runs from there, so overwriting the original `update.bat` mid-update does not break the script.

### If update fails

| Problem | Fix |
|---|---|
| `git fetch failed` / `git pull failed` | Check network / GitHub. Your local changes are in `git stash` — run `git stash list` to see them, `git stash pop` to restore. |
| ZIP download failed | Open https://github.com/KolaKater/Easy3D-Converter manually, click `Code` > `Download ZIP`, unzip, copy over (keep `models/` `venv_Easy3D/` `preset/`). |
| First run after updating from old `update.bat` v1 | Double-click `update.bat` again — v1 could not update itself safely, v2 can. After that, one click is enough. |

---

## Usage

### Quick Start

1. Double-click `Launch_Easy3D_GUI.bat` (or run `venv_Easy3D\Scripts\python.exe Easy3D_GUI_GenStereo.py`).
2. **Input:** Click **Browse File** / **Browse Folder**, or drag & drop an image / folder onto the Input box.
3. **Output:** Leave **Default (InputName_3D)** — output goes to `<InputName>_3D\` next to the input. Or choose **Custom** and pick a folder.
4. Set parameters (recommended first run below).
5. Click **Start**. Watch **Log** / **Preview** / **File List**. Click **Suspend** to pause, **Cancel** to stop.

**Recommended first run (RTX 3070 8GB):**

```
Finetuned:  genstereo-v2.1 
MDE:        FlashDepth-L 
Resolution: 1280 - Inference 
Steps:      20 - Fast
Depth:      1.00
Baseline:   0.40
Fusion:     0.40
Guidance:   1.80
Gen Mode:   Orig_Right
Face Restore: None
Output Type: 1. Parallel (Default)
```

### Input / Output

| Item | Details |
|---|---|
| Input formats | `.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tif` `.tiff` `.gif` `.avif` `.heic` `.heif` |
| Output format | Always JPG, `quality=95, subsampling=0, optimize=True` |
| Output folder | Default: `<InputName>_3D\` / `<InputFolder>_3D\` next to input. Custom: any folder (must differ from input folder if input is a folder). |
| Excluded | `Easy3D.png` (preview placeholder) is never converted. |
| Temp files | `vis/<stem>/` and `_tmp_square/` are deleted per image after its SBS is built. No leftovers. |

### Parameters

#### Depth Strength

Depth-map contrast. Scales relative depth differences.

| Value | Effect |
|---|---|
| 0.2 | Very flat |
| 1.0 | Normal (recommended) |
| 1.2 | Slightly stronger |
| 2.0 | Exaggerated — floating, black edges, tearing |

#### Baseline / Disparity Scale

Absolute eye distance. Total disparity.

```
scale_factor = 0.15 * Baseline * DepthStrength
```

| Value | Effect |
|---|---|
| 0.2 | Very weak stereo |
| 1.0 | Normal (recommended) |
| 1.2 | Slightly stronger |
| 2.0 | Dizzy, ghosting |

**Tip:** Keep `Depth × Baseline ≈ 1.0 – 1.4` for comfortable viewing. Fix one, tune the other. `Depth 2.0 × Baseline 2.0 = 4.0` will almost always ghost.

`Baseline` controls overall stereo strength. `Depth Strength` controls how far near / far are pulled apart. To make the image “more 3D”, raise `Baseline` first. To separate foreground / background more, raise `Depth Strength`.

#### Fusion Strength

Mix of Warped (true lighting / details, with holes) vs Diffusion-generated (fills holes, guessed lighting).

| Value | Effect |
|---|---|
| 0.0 | All Warped — holes remain |
| 0.5 | Balanced (default, recommended) |
| 1.0 | All Generated — lighting may drift |

> **Note (GUI v4.0):** `Fusion Strength` is recorded in the filename (`_F050_`). It does **not** change inference speed, and in v4.0 it is not passed to `test.py` — see `build_command()` in `Easy3D_GUI_GenStereo.py` if you want to wire it to `AdaptiveFusion`.

#### Guidance Scale

How much the diffusion follows the condition.

| Value | Effect |
|---|---|
| 0.0 | Ignores condition, drifts |
| 1.5 | Default (recommended) |
| 5.0 | Too rigid, repetitive |

`Guidance` does **not** affect speed — it is a weight, not a step count.

#### Resolution

| Res | RTX 3070 8GB | Notes |
|---|---|---|
| 512 | Fast, stable | v1.5 default |
| 640–768 | Balanced | 768 = v2.1 default |
| 1024 | Slower | Recommended max for batch on 8 GB |
| 1280 | Slow | Single images OK, batch slow |
| 1536+ | Very slow / may OOM | Lower Res if OOM — do not switch to patch mode |

Available: `512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048`.

#### Steps

Number of diffusion steps. Linear with speed.

| Steps | Use |
|---|---|
| 20 | Fast preview |
| 30 | Balanced |
| 50 | Default quality |
| 75 | High quality, slow, diminishing returns |

#### Seed

Reproducibility. `Random (Default)` = random each run. Or pick `42 / 123 / 999 / 2025`, or type a custom integer (`0 – 2147483647`). Custom values appear as `12345 - Custom`.

#### Gen Mode

| Mode | Left | Right | Passes | Notes |
|---|---|---|---|---|
| `Orig_Right` (default) | Original | AI | 1 | Most common |
| `Orig_Left` | AI | Original | 1 | Mirrored disparity |
| `Both / Symmetric` | AI (−0.075) | AI (+0.075) | 2 | Most comfortable, symmetric blur, total disparity still 0.15 |

#### Face Restore

| Option | Effect |
|---|---|
| `None` (default) | No face restore |
| `Gfpgan_Both - Both Eyes` | GFPGAN on the merged SBS image, both eyes at once. Filename adds `_FRBoth`. |

Requires `gfpgan/weights/` (see [Model List](#model-list)). If GFPGAN fails to load, the SBS is saved unchanged and the Log shows the reason.

#### Low VRAM

On by default. Recommended for RTX 3070 8GB.

#### Save Depth / Save Warped

Optional. Saves `_DEPTH` (from `vis/disp.png`) and `_WARPED` (from `vis/warped.png`) as JPGs alongside the SBS.

### Output Naming

```
<OriginalStem>_Easy3D_<Model><MDE>_R<Res>_D<Depth>_B<Baseline>_F<Fusion>_S<Steps>_G<Guidance>[_FRBoth]_[PAR/XD]_Seed<seed>.jpg
```

Example:

```
White Rose_Easy3D_V15_DA2L_R1024_D100_B100_F050_S50_G150_PAR_Seed457315609.jpg
```

| Token | Meaning |
|---|---|
| `V15` / `V21` | `genstereo-v1.5` / `genstereo-v2.1` |
| `DA2L` / `DA2B` / `DA2S` | Depth Anything V2 ViT-L / B / S |
| `FDL` / `FDS` / `FD` | FlashDepth-L / S / (legacy) |
| `MDEC` | Custom MDE |
| `R1024` | Resolution = 1024 |
| `D100` | Depth Strength = 1.00 × 100 |
| `B100` | Baseline = 1.00 × 100 |
| `F050` | Fusion Strength = 0.50 × 100 |
| `S50` | Steps = 50 |
| `G150` | Guidance = 1.50 × 100 |
| `_FRBoth` | Face Restore (GFPGAN, both eyes) |
| `PAR` / `XD` | Output Type (Parallel / Cross-Eyed) |
| `Seed...` | Seed |

Suffixes: `_DEPTH` for depth map, `_WARPED` for warped image.

### Output Type

Handled by `output_type.py` (do not modify — it only handles already-merged SBS images):

| # | Option | Effect |
|---|---|---|
| 1 | Parallel (Default) | Left \| Right, for parallel viewing / VR |
| 2 | Cross-Eyed (XD Swap) | Right \| Left, for cross-eyed viewing |
| 3 | Auto-Align Parallel | Parallel + auto horizontal shift (convergence) |
| 4 | Auto-Align Cross-Eyed (XD) | XD + auto horizontal shift |
| 5 | Auto-Align Parallel+XD (Both) | Saves 2 files, one Parallel + one XD |

> **Note:** GenStereo uses **parallel cameras** (convergence angle = 0°). There is no “virtual camera horizontal rotation angle” parameter. `Auto-Align` provides the equivalent effect by horizontal shifting — convergence without keystone distortion.

---

## Models & Components

### Model List

All models are **relative** to the Easy3D folder. `install_all.bat` downloads them. Existing files are skipped.

| # | Relative Path | Source | Size | Used For |
|---|---|---|---|---|
| 1 | `models/image_encoder/` (`config.json` + `pytorch_model.bin`) | `lambdalabs/sd-image-variations-diffusers` / fallback `openai/clip-vit-large-patch14` | ~600 MB | CLIP ViT-L/14 image encoder (GenStereo) |
| 2 | `models/sd-vae-ft-mse/` (`config.json` + `diffusion_pytorch_model.safetensors`) | `stabilityai/sd-vae-ft-mse` | ~335 MB | VAE (fixed, not an option in GUI) |
| 3 | `models/depth_anything_v2_vitl.pth` | `depth-anything/Depth-Anything-V2-Large` | ~1.35 GB | Depth Anything V2 ViT-L (default MDE) |
| 4 | `models/FlashDepth/configs/flashdepth-s/iter_14001.pth` (+ `config.yaml`) | `DepthAnything/FlashDepth` | ~100 MB+ | FlashDepth-S MDE (fast) |
| 5 | `models/FlashDepth/configs/flashdepth-l/iter_10001.pth` (+ `config.yaml`) | `DepthAnything/FlashDepth` | ~300 MB+ | FlashDepth-L MDE (official) |
| 6 | `models/genstereo-v1.5/` (`config.json` + `denoising_unet.pth` + `reference_unet.pth` + `pose_guider.pth` + `fusion_layer.pth`) | `FQiao/GenStereo` | ~6.9 GB | GenStereo v1.5 (512 px, RTX 3070 recommended) |
| 7 | `models/genstereo-v2.1/` (same 5 files) | `FQiao/GenStereo-sd2.1` | ~6.9 GB | GenStereo v2.1 (768 px, slower, may OOM on 8 GB) |
| 8 | `gfpgan/weights/GFPGANv1.4.pth` | `TencentARC/GFPGAN` v1.3.4 | ~348 MB | Face Restore (GFPGAN) |
| 9 | `gfpgan/weights/parsing_parsenet.pth` | `TencentARC/GFPGAN` v1.3.4 | ~30–80 MB | Face parsing (GFPGAN) |
| 10 | `gfpgan/weights/detection_Resnet50_Final.pth` | `xinntao/facexlib` v0.2.2 | ~100 MB | Face detection (GFPGAN) |

> **Legacy note:** `checkpoints/` is **not** used. All models live in `models/`. If you see `checkpoints/depth_anything_v2_vitl.pth`, it is a duplicate from an old version — delete it. GUI v4.0 never writes to `checkpoints/`.

### Programs & Libraries

| Component | Version / Source | Purpose |
|---|---|---|
| **GenStereo** | [FQiao/GenStereo](https://github.com/FQiao/GenStereo) (ICCV 2025) | Core stereo generation (dual-UNet diffusion) |
| **Easy3D Converter** | This repo — `Easy3D_GUI_GenStereo.py` (Tkinter) | GUI, batch, streaming, SBS merging |
| **test.py** | This repo (GenStereo official CLI, patched) | Single-pass inference, batch, TF32, xformers/SDPA |
| **output_type.py** | This repo | SBS Parallel / XD / Auto-Align (authoritative for output type) |
| **StereoAutoAlign Ver0.30** | Masuji Suto (Muttyan) — released January 10, 2022 | Auto-Align: horizontal shift / convergence for SBS (called by `output_type.py` via `ffmpeg/bin/stereoautoalign030`) |
| **GFPGAN_face.py** | This repo | GFPGAN face restore on SBS |
| **flashdepth_model.py** | This repo (wrapper for FlashDepth) | FlashDepth MDE inference |
| **Depth Anything V2** | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (`extern/DAM2/`) | Default depth estimation |
| **FlashDepth** | [DepthAnything/FlashDepth](https://github.com/DepthAnything/FlashDepth) | Optional fast depth estimation |
| **PyTorch** | 2.6.0+cu124 (30/40 series) / 2.7.1+cu128 (50 series) | Deep learning |
| **torchvision / torchaudio** | Matching CUDA build | PyTorch companions |
| **xformers** | 0.0.29.post3 (cu124) / 0.0.30 (cu128) | Memory-efficient attention (SDPA fallback if missing) |
| **diffusers** | 0.31.0 | Diffusion pipeline |
| **transformers** | 5.22.1 | CLIP / model loading |
| **accelerate / safetensors / huggingface_hub** | 1.15.0 / 0.8.0 / 2.4.0 | Model loading |
| **GFPGAN / basicsr / facexlib / realesrgan** | 1.3.8 / 1.4.2 / 0.3.0 / 0.3.0 | Face restore stack |
| **gradio** | 4.44.1 | (Dependency, not used for GUI) |
| **tkinterdnd2** | 0.7.0 | Drag & drop |
| **Pillow / numpy / opencv-python / einops / omegaconf / scipy / timm** | 10.4.0 / 1.26.4 / 4.11.0.86 / 0.8.2 / 2.4.0 / 1.17.1 / 0.6.7 | Image / math / config |
| **Triton** | — | Not required on Windows; falls back gracefully |

---

## Project Structure

```
Easy3D/
├── Easy3D_GUI_GenStereo.py      # Tkinter GUI (v4.0) - main entry
├── Launch_Easy3D_GUI.bat        # GUI launcher (hidden via VBS, tries pythonw)
├── update.bat                    # One-click update from GitHub (git pull / ZIP, self-update safe)
├── test.py                      # GenStereo inference (batch, TF32)
├── test_arbitrary_size.py        # UNUSED in GUI (patch-based, reference only) - DO NOT USE
├── output_type.py                # SBS Parallel / XD / Auto-Align - DO NOT MODIFY
├── GFPGAN_face.py               # GFPGAN face restore on SBS
├── flashdepth_model.py          # FlashDepth wrapper
├── install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat / .py  # Create venv, auto-detect GPU, install torch + deps
├── install_all.bat / .py        # One-click: venv + pip + all 10 models
├── download_models.py           # Model downloader (legacy, install_all.py is preferred)
├── requirements.txt / requirements_full.txt  # pip freeze / pip list
├── models/                      # All GenStereo / VAE / MDE models (see Model List)
│   ├── image_encoder/
│   ├── sd-vae-ft-mse/
│   ├── depth_anything_v2_vitl.pth
│   ├── FlashDepth/configs/flashdepth-s/ / flashdepth-l/
│   ├── genstereo-v1.5/
│   └── genstereo-v2.1/
├── gfpgan/weights/              # GFPGAN weights (see Model List)
├── extern/DAM2/                 # Depth Anything V2 submodule (git submodule)
├── GenStereo/                   # GenStereo model code (dual-UNet, etc.)
├── preset/                      # GUI config + presets (easy3d_gui_config.json, easy3d_presets.json)
├── Easy3D.png / Easy3D.ico     # GUI placeholder / icon
├── _tmp_square/                 # Temp squares (auto-deleted)
└── vis/                         # Temp GenStereo outputs (auto-deleted per image)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available() == False` / CPU torch | You installed CPU torch. Delete `venv_Easy3D\`, re-run `install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat` (it installs CUDA torch **first**). |
| `pillow 13 vs gradio` / `numpy 2.x` errors | `install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat` pins `pillow==10.4.0`, `numpy==1.26.4`. Do not upgrade them. |
| `CLIPVisionModelWithProjection` / `torch >= 2.6` error | Upgrade to `torch 2.6.0+cu124` (30/40 series) or `2.7.1+cu128` (50 series) — `install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat` does this. |
| `xformers` not available | Optional. GUI falls back to SDPA. On 50 series, use `xformers 0.0.30` + `cu128`. |
| `Triton` missing on Windows | Normal. Falls back gracefully. |
| `tkinterdnd2` missing / drag-drop not working | Run `venv_Easy3D\Scripts\python.exe -m pip install tkinterdnd2==0.7.0` (included in `install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat`). |
| `GFPGAN loaded: False` | Run `fix_basicsr` / check `basicsr` shim for `torchvision.transforms.functional_tensor.rgb_to_grayscale` (removed in torchvision 0.21). See `GFPGAN_face.py` `get_last_error()`. |
| `image_encoder` corrupted / `PytorchStreamReader` error | Delete `models/image_encoder/` and re-run `install_all.bat`. It tries `lambdalabs` → `hf-mirror` → `openai/clip-vit-large-patch14`. |
| `checkpoints/depth_anything_v2_vitl.pth` reappears | Old GUI v1.0 bug (GUI passed `checkpoints/...` to `test.py`). GUI v4.0 uses `models/` only. Delete `checkpoints/` — it will not reappear on v4.0. |
| `vis/` truncated PNG / SBS failed | Streaming watcher requires stable files (size + mtime unchanged twice). Re-run — `vis/` is kept on failure for manual recovery. |
| OOM on 3070 8GB | Lower **Resolution** (e.g. `1536 → 1024 → 768`) and **Steps** (`50 → 30`). Do not switch to patch mode. |
| `extern/DAM2` empty | ZIP download lacks submodules. Run `git submodule update --init --recursive` or `git clone https://github.com/DepthAnything/Depth-Anything-V2.git extern/DAM2`. |
| Batch output duplicated (last image twice) | Stop watcher before final sweep (see `start_conversion()` in GUI). |
| `Flux.2 Klein` / other models cannot be loaded | `Finetuned` only supports `genstereo-v1.5` / `genstereo-v2.1` (SD UNet arch). Flux (DiT) is a different architecture — `load_state_dict` will fail with `size mismatch`. |
| `update.bat` fails on first run after updating from v1 | Double-click `update.bat` again — v1 could not update itself safely, v2 can. After that, one click is enough. |

---

## FAQ

**Q: Can I use it for video?**
No. No, at the moment it is single-image only. Frame-by-frame causes flicker / depth jumps. For video, use StereoWorld / StereoCrafter / SVG.

**Q: Does Fusion / Guidance / Depth / Baseline affect speed?**
No. They are weights. Speed is determined by **Steps**, **Resolution**, and **Finetuned Model** (v1.5 vs v2.1) only.

**Q: Can I use Flux.2 Klein as Finetuned?**
No. This converter needs SD 1.5 / 2.1 UNet weights. Flux is DiT + Flow Matching, different VAE, different weight shapes.

---

## Credits & License

### Credits

This project builds on the work of many open-source authors and teams. Thank you to all of them.

| Component | Authors / Team | Link / License |
|---|---|---|
| **GenStereo** | Fan Qiao et al. — ICCV 2025 | [FQiao/GenStereo](https://github.com/FQiao/GenStereo) — see its repo for license |
| **Depth Anything V2** | Depth Anything Team | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) — Apache-2.0 |
| **FlashDepth** | Depth Anything Team / FlashDepth Authors | [DepthAnything/FlashDepth](https://github.com/DepthAnything/FlashDepth) — see its repo for license |
| **Stable Diffusion** | CompVis / Stability AI / Runway | [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) — CreativeML Open RAIL-M |
| **CLIP** | OpenAI | [openai/CLIP](https://github.com/openai/CLIP) — MIT |
| **GFPGAN** | Tencent ARC — Xintao Wang et al. | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) — Apache-2.0 |
| **StereoAutoAlign Ver0.30** | Masuji Suto (Muttyan) — released January 10, 2022 | Auto-Align for SBS stereo (horizontal shift / convergence). Used by `output_type.py` for Auto-Align Parallel / XD. Original tool by Masuji Suto (Muttyan), Ver0.30 released January 10, 2022. See `ffmpeg/bin/stereoautoalign030/` in this repo. |
| **BasicSR / Real-ESRGAN / FaceXLib** | Xintao Wang et al. | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) / [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) / [facexlib](https://github.com/xinntao/facexlib) — BSD-3-Clause / Apache-2.0 |
| **Hugging Face Diffusers / Transformers** | Hugging Face | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [transformers](https://github.com/huggingface/transformers) — Apache-2.0 |
| **PyTorch** | Meta AI | [pytorch/pytorch](https://github.com/pytorch/pytorch) — BSD-3-Clause |
| **xFormers** | Meta AI / xFormers Team | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) — BSD-3-Clause |
| **TkinterDnD2** | Michael Lange / TkinterDnD2 contributors | [TkinterDnD2](https://github.com/pmgagne/tkinterdnd2) — MIT |
| **Easy3D Converter** | Tony Lin (this repo) | MIT (see below) — GUI, batch, streaming, SBS merging, naming |

**Model weights** are distributed by their original authors under their own licenses. This repo does **not** re-license them. Before redistributing any downloaded weight file, check its original repo / Hugging Face page for its license:

- `FQiao/GenStereo` / `FQiao/GenStereo-sd2.1` — GenStereo weights
- `stabilityai/sd-vae-ft-mse` — VAE
- `lambdalabs/sd-image-variations-diffusers` / `openai/clip-vit-large-patch14` — Image Encoder
- `depth-anything/Depth-Anything-V2-Large` — Depth Anything V2
- `DepthAnything/FlashDepth` — FlashDepth
- `TencentARC/GFPGAN` — GFPGAN weights
- StereoAutoAlign Ver0.30 (Masuji Suto / Muttyan, January 10, 2022) — Auto-Align tool, see `ffmpeg/bin/stereoautoalign030/`
- `xinntao/facexlib` — Face detection weights

### License

**Easy3D Converter code in this repository is released under the MIT License:**

```
MIT License

Copyright (c) 2026 Tony Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Third-party code, models, and weights retain their original licenses.** See the table above and each upstream repo for details.

### Disclaimer

This project is for research / personal use. Stereo 3D viewing can cause eye strain / dizziness — take breaks, and do not use while driving or operating machinery. The authors are not responsible for any discomfort or misuse.

---

**Easy3D Converter — Made for RTX 3070, shared for everyone. Enjoy your 3D!**
