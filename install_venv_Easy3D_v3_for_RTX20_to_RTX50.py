"""install_venv_Easy3D_v3.py - Auto-detect GPU, install matching torch CUDA. RELATIVE PATHS ONLY.
Supports: RTX 3070 and above (30/40/50 series), plus fallback.
Logic:
  - Run nvidia-smi, get GPU name.
  - 50 series (5070/5080/5090/5060, Blackwell sm100/sm120) -> torch 2.7.1+cu128, xformers 0.0.30
  - 40 series (4070/4080/4090, Ada sm89) + 30 series (3070/3080/3090, Ampere sm86) -> torch 2.6.0+cu124, xformers 0.0.29.post3
  - 20 series / GTX / older -> torch 2.6.0+cu124 still works (sm75), same as 30 series
  - No NVIDIA GPU / nvidia-smi missing -> CPU torch (slow, but runs)
Why two torch versions?
  cu124 does NOT have Blackwell (sm120) kernels. 50 series on cu124 will say
  "no kernel image is available" or fall back to CPU. cu128 has Blackwell kernels.
  cu128 also works on 30/40 series, but we keep cu124 for 30/40 to match your tested requirements.txt.
Usage: double-click install_venv_Easy3D_v3.bat, or: python install_venv_Easy3D_v3.py
"""
import pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).parent
VENV = HERE / "venv_Easy3D"
VENV_PY = VENV / "Scripts" / "python.exe"

def run(cmd, check=False):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return r.returncode

def detect_gpu():
    """Returns (gpu_name, cuda_index, torch_ver, torchvision_ver, torchaudio_ver, xformers_ver)."""
    name = ""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().splitlines()[0].strip()
            print(f"  [GPU] nvidia-smi: {name}")
    except Exception as e:
        print(f"  [GPU] nvidia-smi not found / failed: {e}")
    # 50 series: 5090, 5080, 5070, 5060, 5050, also 50xx Ti / SUPER
    if re.search(r"\b50\d{2}\b", name):
        print("  [GPU] Detected RTX 50 series (Blackwell) -> torch 2.7.1+cu128")
        return (name, "cu128", "torch==2.7.1+cu128", "torchvision==0.22.1+cu128", "torchaudio==2.7.1+cu128", "xformers==0.0.30")
    # 40 / 30 / 20 series all work on cu124
    if re.search(r"\b(40\d{2}|30\d{2}|20\d{2})\b", name):
        print(f"  [GPU] Detected RTX {name} -> torch 2.6.0+cu124 (tested on your RTX 3070)")
        return (name, "cu124", "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124", "xformers==0.0.29.post3")
    if name:
        print(f"  [GPU] Unknown NVIDIA GPU '{name}', trying cu124 (works for most Ampere/Ada/Turing)")
        return (name, "cu124", "torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchaudio==2.6.0+cu124", "xformers==0.0.29.post3")
    print("  [GPU] No NVIDIA GPU found -> CPU torch (slow, for testing only)")
    return ("", "cpu", "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0", None)

def main():
    print("=== install_venv_Easy3D v3 - Auto GPU ===")
    print(f"  Base: {HERE.name} (this file's folder, relative only)")
    gpu_name, cuda_idx, torch_pkg, tv_pkg, ta_pkg, xformers_pkg = detect_gpu()
    # 1. venv
    if not VENV_PY.exists():
        print("  [1/6] Creating venv_Easy3D ...")
        run([sys.executable, "-m", "venv", str(VENV)], check=True)
    else:
        print("  [1/6] venv_Easy3D exists, reuse it.")
    PY = str(VENV_PY)
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # 2. torch (must be FIRST)
    print(f"  [2/6] Installing {torch_pkg} ({cuda_idx}) ...")
    if cuda_idx == "cpu":
        run([PY, "-m", "pip", "install", torch_pkg, tv_pkg, ta_pkg], check=True)
    else:
        run([PY, "-m", "pip", "install", torch_pkg, tv_pkg, ta_pkg, "--index-url", f"https://download.pytorch.org/whl/{cuda_idx}"], check=True)
    # 3. requirements.txt or core list (skip torch lines, already installed)
    print("  [3/6] Installing project packages ...")
    REQ = HERE / "requirements.txt"
    if REQ.exists():
        run([PY, "-m", "pip", "install", "-r", str(REQ), "--no-deps"])
    run([PY, "-m", "pip", "install", "numpy==1.26.4", "pillow==10.4.0", "diffusers==0.31.0", "transformers==5.22.1", "accelerate==1.15.0", "huggingface_hub==2.4.0", "safetensors==0.8.0", "einops==0.8.2", "omegaconf==2.4.0", "scipy==1.17.1", "opencv-python==4.11.0.86", "timm==0.6.7", "gradio==4.44.1", "tkinterdnd2==0.7.0"])
    # 4. xformers matching CUDA
    print("  [4/6] Installing xformers ...")
    if xformers_pkg:
        if cuda_idx == "cpu":
            run([PY, "-m", "pip", "install", xformers_pkg])
        else:
            run([PY, "-m", "pip", "install", xformers_pkg, "--index-url", f"https://download.pytorch.org/whl/{cuda_idx}"])
    # 5. GFPGAN stack
    print("  [5/6] Installing GFPGAN stack ...")
    run([PY, "-m", "pip", "install", "gfpgan==1.3.8", "basicsr==1.4.2", "facexlib==0.3.0", "realesrgan==0.3.0", "yapf==0.43.0"])
    # 6. Verify
    print("\n=== VERIFY ===")
    run([PY, "-c", "import torch; print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"])
    run([PY, "-c", "import tkinterdnd2, omegaconf, einops; print('tkinterdnd2/omegaconf/einops: OK')"])
    print(f"\n[DONE] venv_Easy3D ready for GPU: {gpu_name or 'CPU'} ({cuda_idx})")
    print("  Next: double-click install_all.bat to download the 10 models, or run GUI v4.0.")

if __name__ == "__main__":
    main()
