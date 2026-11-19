#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install_Distill.py -- Easy3D 一键装 Distill Any Depth Base (ViTB)
权重找不到就自动下载 (用户要求: 不要只检查, 缺了就下)
  官方权重: https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/base/model.safetensors
  本地位置: models/distill_any_depth_vitb.safetensors  (约 97.5M 参数, 文件约 390MB)
  代码: torch.hub nagadomi/Depth-Anything_iw3:main 的 DistillAnyDepth, 第一次跑 test.py 时 torch.hub 会自动把代码下到 HF 缓存
  依赖: timm + safetensors + huggingface-hub (DA3/MoGe3 那套 numpy>=2 / pillow<13 / triton / matplotlib 修复也一起做, 免得装完又炸)

用法:
  双击 Install_Distill.bat
    = 装代码 + 找不到权重就自动下载 + 检查
  venv_Easy3D\\Scripts\\python.exe Install_Distill.py --check-only
    = 只检查, 不装不下
  venv_Easy3D\\Scripts\\python.exe Install_Distill.py --no-download
    = 装代码, 但权重缺了也不下 (只报 MISSING)
  venv_Easy3D\\Scripts\\python.exe Install_Distill.py --download-models
    = 强制重下权重 (哪怕已有, 也校验大小, 不对就重下)
  venv_Easy3D\\Scripts\\python.exe Install_Distill.py --repair
    = 只修 numpy/pillow/triton/matplotlib, 不装 Distill
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "install_distill.log"
WEIGHT_LOCAL = HERE / "models" / "distill_any_depth_vitb.safetensors"
HF_REPO = "xingyang1/Distill-Any-Depth"
HF_FILENAME = "base/model.safetensors"
TRITON_PIN = "triton-windows>=3.2.0.post11,<3.3"
# DistillAnyDepth 代码 (torch.hub 那份) 要 timm 做 backbone, 别漏
DISTILL_DEPS = ["timm", "safetensors", "huggingface-hub"]


def _log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _find_python():
    for cand in (
        HERE / "venv_Easy3D" / "Scripts" / "python.exe",
        HERE / "venv" / "Scripts" / "python.exe",
        HERE / "venv_Easy3D" / "bin" / "python",
        HERE / "venv" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


def _run(cmd, label):
    _log(f"[RUN] {label}: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(HERE))
        _log(f"[DONE] {label} exit={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        _log(f"[FAIL] {label}: {e}")
        return False


def _check_import(python, mod):
    r = subprocess.run(
        [python, "-c", f"import {mod}; print('{mod} ok')"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    ok = r.returncode == 0
    _log(f"[CHECK] import {mod}: {'OK' if ok else 'FAIL'} {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    return ok


def _check_models():
    ok = WEIGHT_LOCAL.exists() and WEIGHT_LOCAL.stat().st_size > 10 * 1024 * 1024
    if ok:
        _log(f"[MODEL] OK models/distill_any_depth_vitb.safetensors ({WEIGHT_LOCAL.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        _log("[MODEL] MISSING models/distill_any_depth_vitb.safetensors -- 找不到就自动下载")
    # Distill_model.py 在不在 models/ 里, 也顺手查一下 (补丁 / 手动拷都行)
    wrapper = HERE / "models" / "Distill_model.py"
    if wrapper.exists():
        _log(f"[WRAPPER] OK models/Distill_model.py")
    else:
        _log("[WRAPPER] MISSING models/Distill_model.py -- 把改好的 Distill_model.py 拷到 models/ 里再跑 GUI")
    return ok


def _download_models(python):
    """找不到就自动下: HF 下 base/model.safetensors, 拷到 models/distill_any_depth_vitb.safetensors."""
    _log(f"[MODEL] DOWNLOAD {HF_REPO} {HF_FILENAME} -> models/distill_any_depth_vitb.safetensors (约 390MB, 别关窗口)")
    code = (
        "from pathlib import Path\n"
        "import shutil\n"
        "from huggingface_hub import hf_hub_download\n"
        "src = hf_hub_download(repo_id='xingyang1/Distill-Any-Depth', filename='base/model.safetensors')\n"
        "dst = Path('models/distill_any_depth_vitb.safetensors')\n"
        "dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "print(f'[MODEL] HF cache -> {src}')\n"
        "shutil.copy2(src, dst)\n"
        "print(f'[MODEL] DONE {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)')\n"
    )
    return _run([python, "-c", code], "download Distill weight from HF")


def _repair(python):
    _log("[REPAIR] pip install numpy>=2,<3 pillow>=10,<13")
    return _run([python, "-m", "pip", "install", "numpy>=2,<3", "pillow>=10,<13"], "repair numpy/pillow")


def _fix_triton(python):
    _log(f"[FIX-TRITON] pip install {TRITON_PIN}")
    return _run([python, "-m", "pip", "install", TRITON_PIN], "pin triton-windows")


def _fix_matplotlib(python):
    _log("[FIX-MPL] pip install matplotlib>=3.9,<3.10 (NumPy 2 兼容版)")
    return _run([python, "-m", "pip", "install", "matplotlib>=3.9,<3.10"], "upgrade matplotlib for NumPy 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只检查, 不装不下")
    ap.add_argument("--check-models", action="store_true", help="只检查权重, 不装代码")
    ap.add_argument("--download-models", action="store_true", help="强制重下权重")
    ap.add_argument("--no-download", action="store_true", help="权重缺了也不下, 只报 MISSING")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--no-upgrade-pip", action="store_true")
    args = ap.parse_args()
    LOG_FILE.write_text("", encoding="utf-8")
    python = _find_python()
    _log(f"[INFO] Python: {python}")
    _log(f"[INFO] Folder: {HERE}")

    if args.check_models:
        ok = _check_models()
        return 0 if ok else 1
    if args.repair:
        _fix_triton(python)
        _fix_matplotlib(python)
        _repair(python)
        return 0
    if args.check_only:
        ok_dep = _check_import(python, "timm") and _check_import(python, "safetensors")
        ok_model = _check_models()
        _log(f"[RESULT] Distill deps: {'OK' if ok_dep else 'NOT INSTALLED'} | 权重: {'OK' if ok_model else 'MISSING'}")
        return 0 if (ok_dep and ok_model) else 1

    git_ok = _run(["git", "--version"], "check git")
    if not git_ok:
        _log("[FAIL] 没找到 git. torch.hub 要用 git 下 Distill 代码, 先装 git: https://git-scm.com/download/win , 装完重开 cmd 再双击 BAT.")
        return 1
    if not args.no_upgrade_pip:
        _run([python, "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip")

    # 1) 装 Distill 依赖 (timm 是 DistillAnyDepth backbone 必须的, 别用 --no-deps)
    if not _check_import(python, "timm"):
        _run([python, "-m", "pip", "install"] + DISTILL_DEPS, "install Distill deps (timm/safetensors/huggingface-hub)")
    else:
        _log("[SKIP] timm 已装, 补齐 safetensors/huggingface-hub")
        _run([python, "-m", "pip", "install", "safetensors", "huggingface-hub"], "ensure safetensors/huggingface-hub")

    # 2) 修 DA3/MoGe3 那套冲突 (numpy/pillow/triton/matplotlib, 不修后面也会炸)
    _repair(python)
    _fix_triton(python)
    _fix_matplotlib(python)

    # 3) 权重: 找不到就自动下, 默认行为; --no-download 才不下; --download-models 强制重下
    model_ok = _check_models()
    if args.download_models or (not model_ok and not args.no_download):
        if not _download_models(python):
            _log("[FAIL] 权重下载失败, 检查网络 / HF 能不能上, 或手动下: https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/base/model.safetensors 另存为 models/distill_any_depth_vitb.safetensors")
            return 1
        model_ok = _check_models()
    elif not model_ok:
        _log("[MODEL] --no-download: 权重缺了, 按你的要求没下. 要下就去掉 --no-download 再跑一次.")

    ok_dep = _check_import(python, "timm")
    _log(f"[RESULT] Distill deps: {'OK' if ok_dep else 'FAIL'} | 权重: {'OK' if model_ok else 'FAIL'} | log: {LOG_FILE}")
    if ok_dep and model_ok:
        _log("[OK] Distill 装好了. 再跑补丁 patch_add_distill_gui_test_v105.py, 回 GUI 选 Distill-B 就能跑.")
        return 0
    _log("[FAIL] 有一个没装好, 把 install_distill.log 最后 30 行贴回来.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
