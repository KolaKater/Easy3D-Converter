"""install_all.py - One-click: venv + pip (incl. tkinterdnd2) + ALL 10 models. For GUI v4.0.
RELATIVE PATHS ONLY. No absolute path anywhere.
Run: double-click install_all.bat, or: venv_Easy3D\Scripts\python.exe install_all.py
Base = folder where this file lives (Path(__file__).parent). Copy this file + bat to Easy3D folder and run.
NEVER downloads to checkpoints/. Skips any file that already exists and is not empty.

10 targets (your list, all relative to Easy3D folder):
  1. models/image_encoder/                            (config.json + pytorch_model.bin)
  2. models/sd-vae-ft-mse/                            (config.json + diffusion_pytorch_model.safetensors)
  3. models/depth_anything_v2_vitl.pth
  4. models/FlashDepth/configs/flashdepth-s/iter_14001.pth (+ config.yaml)
  5. models/FlashDepth/configs/flashdepth-l/iter_10001.pth (+ config.yaml)
  6. models/genstereo-v1.5/                            (config.json + 4x .pth)
  7. models/genstereo-v2.1/                            (config.json + 4x .pth)
  8. gfpgan/weights/GFPGANv1.4.pth
  9. gfpgan/weights/parsing_parsenet.pth
  10. gfpgan/weights/detection_Resnet50_Final.pth
Plus pip: torch 2.6.0+cu124, torchvision, xformers, diffusers, transformers, gradio,
numpy, pillow, einops, omegaconf, scipy, timm, opencv, tkinterdnd2, gfpgan/basicsr/facexlib/realesrgan.
"""
import pathlib, shutil, subprocess, sys, urllib.request

# RELATIVE ONLY: base = this file's folder, never an absolute path
HERE = pathlib.Path(__file__).parent
MODELS = HERE / "models"
GFPGAN_W = HERE / "gfpgan" / "weights"
CHECKPOINTS = HERE / "checkpoints"
VENV_PY = HERE / "venv_Easy3D" / "Scripts" / "python.exe"

def ok(p, min_bytes=1000):
    p = pathlib.Path(p)
    good = p.exists() and p.is_file() and p.stat().st_size > min_bytes
    # Print RELATIVE path only (relative to HERE), never absolute
    try:
        rel = p.relative_to(HERE)
    except Exception:
        rel = p.name
    print(f"  [{'OK' if good else 'MISSING'}] {rel} ({p.stat().st_size if p.exists() else 0} bytes)")
    return good

def dl_https(url, dest, min_bytes=1000):
    dest = pathlib.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if ok(dest, min_bytes): return True
    try: rel = dest.relative_to(HERE)
    except Exception: rel = dest.name
    print(f"  HTTPS {url} -> {rel}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1024*1024)
        return ok(dest, min_bytes)
    except Exception as e:
        print(f"  FAILED {url}: {e}")
        try: dest.unlink()
        except Exception: pass
        return False

def dl_hf(repo_id, filename, dest, min_bytes=1000):
    dest = pathlib.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if ok(dest, min_bytes): return True
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print(f"  hf_hub not available: {e}")
        return False
    try:
        try: rel = dest.relative_to(HERE)
        except Exception: rel = dest.name
        print(f"  HF {repo_id}:{filename} -> {rel}")
        got = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent), local_dir_use_symlinks=False)
        if pathlib.Path(got) != dest and pathlib.Path(got).exists():
            shutil.copy2(got, dest)
        return ok(dest, min_bytes)
    except Exception as e:
        print(f"  HF FAILED {filename}: {e}")
        return False

def run(cmd, check=False):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return r.returncode

print("=== [1/3] venv + pip (incl. tkinterdnd2) ===")
print(f"  Base (relative): {HERE.name} (this file's folder)")
if not VENV_PY.exists():
    print("  Creating venv_Easy3D ...")
    run([sys.executable, "-m", "venv", str(HERE / "venv_Easy3D")], check=True)
else:
    print("  venv_Easy3D already exists, reuse it.")
PY = str(VENV_PY)
run([PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
print("  Installing torch CUDA FIRST (must be before other packages)...")
run([PY, "-m", "pip", "install", "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
REQ = HERE / "requirements.txt"
if REQ.exists():
    print("  Installing requirements.txt (with --no-deps to keep CUDA torch)...")
    run([PY, "-m", "pip", "install", "-r", str(REQ), "--no-deps"])
    run([PY, "-m", "pip", "install", "numpy==1.26.4", "pillow==10.4.0", "diffusers==0.31.0", "transformers==5.22.1", "accelerate==1.15.0", "huggingface_hub==2.4.0", "safetensors==0.8.0", "einops==0.8.2", "omegaconf==2.4.0", "scipy==1.17.1", "opencv-python==4.11.0.86", "timm==0.6.7", "gradio==4.44.1", "tkinterdnd2==0.7.0"])
else:
    print("  No requirements.txt, installing core list...")
    run([PY, "-m", "pip", "install", "numpy==1.26.4", "pillow==10.4.0", "gradio==4.44.1", "transformers==5.22.1", "diffusers==0.31.0", "accelerate==1.15.0", "safetensors==0.8.0", "huggingface_hub==2.4.0", "opencv-python==4.11.0.86", "einops==0.8.2", "omegaconf==2.4.0", "scipy==1.17.1", "timm==0.6.7", "tkinterdnd2==0.7.0"])
print("  Installing xformers + GFPGAN stack...")
run([PY, "-m", "pip", "install", "xformers==0.0.29.post3", "--index-url", "https://download.pytorch.org/whl/cu124"])
run([PY, "-m", "pip", "install", "gfpgan==1.3.8", "basicsr==1.4.2", "facexlib==0.3.0", "realesrgan==0.3.0", "yapf==0.43.0"])
print("  Verifying pip ...")
run([PY, "-c", "import torch, tkinterdnd2, omegaconf, einops, scipy; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('tkinterdnd2/omegaconf/einops/scipy: OK')"])

print("\n=== [2/3] 10 models (skip if already OK, never re-download, never checkpoints/) ===")
MODELS.mkdir(exist_ok=True); GFPGAN_W.mkdir(parents=True, exist_ok=True)
print("\n-- [1/10] models/image_encoder --")
enc = MODELS / "image_encoder"; enc.mkdir(exist_ok=True)
if not ok(enc / "pytorch_model.bin", 100_000_000):
    done = dl_https("https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json", enc / "config.json")
    if done: dl_https("https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin", enc / "pytorch_model.bin", 100_000_000)
if not ok(enc / "pytorch_model.bin", 100_000_000):
    print("  Fallback: openai/clip-vit-large-patch14")
    dl_https("https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json", enc / "config.json")
    dl_https("https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin", enc / "pytorch_model.bin", 100_000_000)
print("\n-- [2/10] models/sd-vae-ft-mse --")
vae = MODELS / "sd-vae-ft-mse"; vae.mkdir(exist_ok=True)
if not ok(vae / "diffusion_pytorch_model.safetensors", 10_000_000):
    if not dl_hf("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.safetensors", vae / "diffusion_pytorch_model.safetensors", 10_000_000):
        dl_https("https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors", vae / "diffusion_pytorch_model.safetensors", 10_000_000)
    dl_hf("stabilityai/sd-vae-ft-mse", "config.json", vae / "config.json", 100)
print("\n-- [3/10] models/depth_anything_v2_vitl.pth --")
dam = MODELS / "depth_anything_v2_vitl.pth"
if not ok(dam, 1_000_000_000):
    if not dl_hf("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth", dam, 1_000_000_000):
        dl_https("https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth", dam, 1_000_000_000)
print("\n-- [4/10] + [5/10] models/FlashDepth --")
for short, fn in [("flashdepth-s", "iter_14001.pth"), ("flashdepth-l", "iter_10001.pth")]:
    d = MODELS / "FlashDepth" / "configs" / short; d.mkdir(parents=True, exist_ok=True)
    dest = d / fn
    if not ok(dest, 100_000_000):
        if not dl_hf("DepthAnything/FlashDepth", f"configs/{short}/{fn}", dest, 100_000_000):
            dl_https(f"https://huggingface.co/DepthAnything/FlashDepth/resolve/main/configs/{short}/{fn}", dest, 100_000_000)
    cfg = d / "config.yaml"
    if not cfg.exists():
        if not dl_hf("DepthAnything/FlashDepth", f"configs/{short}/config.yaml", cfg, 100):
            dl_https(f"https://huggingface.co/DepthAnything/FlashDepth/resolve/main/configs/{short}/config.yaml", cfg, 100)
    if not ok(dest, 100_000_000):
        print(f"  [WARN] {dest.name} still missing - download manually from DepthAnything/FlashDepth and place it here.")
fd_code = MODELS / "FlashDepth" / "flashdepth" / "model.py"
if not ok(fd_code, 1000):
    print("  [WARN] FlashDepth/flashdepth/model.py missing - copy official FlashDepth code into models/FlashDepth/flashdepth/")
print("\n-- [6/10] + [7/10] models/genstereo-v1.5 / v2.1 --")
for ver, repo in [("genstereo-v1.5", "FQiao/GenStereo"), ("genstereo-v2.1", "FQiao/GenStereo-sd2.1")]:
    vd = MODELS / ver; vd.mkdir(exist_ok=True)
    for fn in ["config.json", "denoising_unet.pth", "fusion_layer.pth", "pose_guider.pth", "reference_unet.pth"]:
        dest = vd / fn
        if ok(dest, 1000): continue
        if not dl_hf(repo, fn, dest, 1000):
            dl_https(f"https://huggingface.co/{repo}/resolve/main/{fn}", dest, 1000)
print("\n-- [8/10] + [9/10] + [10/10] gfpgan/weights --")
for url, dest, mn in [
    ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth", GFPGAN_W / "GFPGANv1.4.pth", 100_000_000),
    ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/parsing_parsenet.pth", GFPGAN_W / "parsing_parsenet.pth", 10_000_000),
    ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/detection_Resnet50_Final.pth", GFPGAN_W / "detection_Resnet50_Final.pth", 10_000_000),
]:
    if not ok(dest, mn):
        dl_https(url, dest, mn)

print("\n=== [3/3] Fix checkpoints/ duplicate (should NOT exist) ===")
dup = CHECKPOINTS / "depth_anything_v2_vitl.pth"
if dup.exists():
    print("  Found duplicate: checkpoints/depth_anything_v2_vitl.pth")
    if ok(MODELS / "depth_anything_v2_vitl.pth", 1_000_000_000):
        try:
            dup.unlink()
            print("  Deleted duplicate checkpoints/depth_anything_v2_vitl.pth (models/ kept).")
        except Exception as e:
            print(f"  Could not delete duplicate: {e}")
    else:
        try:
            shutil.move(str(dup), str(MODELS / "depth_anything_v2_vitl.pth"))
            print("  Moved checkpoints/depth_anything_v2_vitl.pth -> models/ (no re-download).")
        except Exception as e:
            print(f"  Move failed: {e}")
else:
    print("  No duplicate in checkpoints/ - good.")
if CHECKPOINTS.exists():
    for p in list(CHECKPOINTS.rglob("*.pth")):
        cand = MODELS / p.name
        if cand.exists() and cand.stat().st_size == p.stat().st_size and p != cand:
            try: p.unlink(); print(f"  Deleted duplicate {p.name}")
            except Exception: pass
    try:
        if not any(CHECKPOINTS.iterdir()):
            CHECKPOINTS.rmdir(); print("  Removed empty checkpoints/ folder.")
    except Exception: pass

print("\n=== FINAL CHECK (10/10) ===")
checks = [
    ok(MODELS / "image_encoder" / "pytorch_model.bin", 100_000_000),
    ok(MODELS / "sd-vae-ft-mse" / "diffusion_pytorch_model.safetensors", 10_000_000),
    ok(MODELS / "depth_anything_v2_vitl.pth", 1_000_000_000),
    ok(MODELS / "FlashDepth" / "configs" / "flashdepth-s" / "iter_14001.pth", 100_000_000),
    ok(MODELS / "FlashDepth" / "configs" / "flashdepth-l" / "iter_10001.pth", 100_000_000),
    ok(MODELS / "genstereo-v1.5" / "denoising_unet.pth", 1000),
    ok(MODELS / "genstereo-v2.1" / "denoising_unet.pth", 1000),
    ok(GFPGAN_W / "GFPGANv1.4.pth", 100_000_000),
    ok(GFPGAN_W / "parsing_parsenet.pth", 10_000_000),
    ok(GFPGAN_W / "detection_Resnet50_Final.pth", 10_000_000),
]
print(f"\n{sum(checks)}/10 OK")
if sum(checks) == 10:
    print("DONE - All models present. GUI v4.0 should now run without re-downloading to checkpoints/.")
else:
    print("Some models still missing - see MISSING lines above, download those manually.")
