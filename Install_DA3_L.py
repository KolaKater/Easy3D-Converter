#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install_DA3_L.py -- Easy3D 一键装 DA3-L (Large) + 检查/自动下载权重

和 Install_DA3_MoGe3.py v7 的区别:
  v7 是 DA3-B/S + MoGe3, 默认只检查权重、不自动下 (几个 GB, 怕你流量打爆).
  这个是专门给 DA3-L 的, 按你的要求: 检查模型, 如果没有就自动下载.
  默认就是自动下, 不用加 --download-models. 想只检查不下, 加 --check-only / --no-download.

装什么:
  1) pip: depth_anything_3 (GitHub --no-deps, 不让它降 numpy / 升 pillow) + 最小依赖
     e3nn / einops / omegaconf / safetensors / huggingface-hub / pillow-heif
  2) 修复: numpy>=2,<3 + pillow>=10,<13 + triton-windows 3.2.0.post21 + matplotlib>=3.9,<3.10
     (前面 v78/v79 的坑: triton 3.9 炸 xformers, matplotlib 3.6.2 在 NumPy 2 下 _ARRAY_API not found)
  3) 权重: depth-anything/DA3-LARGE -> models/da3-large/ (含 config.json + model.safetensors)
     为什么要下整个目录, 不是只下 da3mono-large.safetensors?
       你手动下的 models/da3mono-large.safetensors 单独一个文件缺 config.json,
       depth_anything_3 官方 API 没法直接从单个 .safetensors 加载, 必须有个带 config.json 的目录.
       所以 Install 会把完整 DA3-LARGE 下到 models/da3-large/, DA_v3_l.py 优先用这个目录.
       你那个 da3mono-large.safetensors 可以留着, 不删, 但它不是正式加载路径.

检查哪几个 (相对路径, 以脚本所在目录为 Easy3D 根目录):
  models/da3-large/config.json + models/da3-large/model.safetensors  (两个都有才算 OK)
  models/da3mono-large.safetensors  (只作提示, 有它也不算 OK, 因为缺 config)

用法 (相对路径, 在 Easy3D 根目录):
  双击 Install_DA3_L.bat
  或: venv_Easy3D\\Scripts\\python.exe Install_DA3_L.py
  或: venv_Easy3D\\Scripts\\python.exe Install_DA3_L.py --check-only   (只检查, 不装不下)
  或: venv_Easy3D\\Scripts\\python.exe Install_DA3_L.py --no-download  (只装代码+修复, 不下权重)
  或: venv_Easy3D\\Scripts\\python.exe Install_DA3_L.py --repair      (只修 numpy/pillow+triton+matplotlib)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "install_da3_l.log"

DA3_URL = "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_MIN_DEPS = ["e3nn", "einops", "omegaconf", "safetensors", "huggingface-hub", "pillow-heif"]
TRITON_PIN = "triton-windows>=3.2.0.post11,<3.3"

HF_REPO = "depth-anything/DA3-LARGE"
LOCAL_DIR = HERE / "models" / "da3-large"
MONO_FILE = HERE / "models" / "da3mono-large.safetensors"


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
    """返回 (ok, missing_list). ok = da3-large/ 里 config.json + model.safetensors 都有."""
    cfg = LOCAL_DIR / "config.json"
    w = LOCAL_DIR / "model.safetensors"
    has_cfg, has_w = cfg.exists(), w.exists()
    if has_cfg and has_w:
        _log(f"[MODEL] OK models/da3-large/ (config.json + model.safetensors, HF: {HF_REPO})")
    else:
        miss = []
        if not has_cfg:
            miss.append("config.json")
        if not has_w:
            miss.append("model.safetensors")
        _log(f"[MODEL] MISSING models/da3-large/ 缺: {', '.join(miss)} (HF: {HF_REPO})")
    if MONO_FILE.exists():
        _log(f"[MODEL] INFO models/da3mono-large.safetensors 已有 ({MONO_FILE.stat().st_size // 1024 // 1024} MB), 但它单独缺 config.json, 官方 API 不能直接加载, 仍需完整 da3-large/ 目录")
    else:
        _log("[MODEL] INFO models/da3mono-large.safetensors 没有 (没关系, 正式加载用 models/da3-large/)")
    return (has_cfg and has_w)


def _download_models(python):
    # 自动下载: snapshot_download 下整个 DA3-LARGE (含 config.json), 几个 GB, 别关窗口
    _log(f"[MODEL] DOWNLOAD {HF_REPO} -> models/da3-large/ (自动下载, 几个 GB)")
    code = (
        "from huggingface_hub import snapshot_download\n"
        "print('[MODEL] snapshot_download depth-anything/DA3-LARGE ...', flush=True)\n"
        "snapshot_download(repo_id='depth-anything/DA3-LARGE', local_dir='models/da3-large')\n"
        "print('[MODEL] DONE models/da3-large/')\n"
    )
    return _run([python, "-c", code], "download DA3-LARGE")


def _repair(python):
    _log("[REPAIR] pip install numpy>=2,<3 pillow>=10,<13")
    return _run([python, "-m", "pip", "install", "numpy>=2,<3", "pillow>=10,<13"], "repair numpy/pillow")


def _fix_triton(python):
    _log(f"[FIX-TRITON] pip install {TRITON_PIN}")
    return _run([python, "-m", "pip", "install", TRITON_PIN], "pin triton-windows")


def _fix_matplotlib(python):
    _log("[FIX-MPL] pip install matplotlib>=3.9,<3.10 (NumPy 2 兼容版)")
    return _run([python, "-m", "pip", "install", "matplotlib>=3.9,<3.10"], "upgrade matplotlib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只检查, 不装不下")
    ap.add_argument("--no-download", action="store_true", help="只装代码+修复, 不下权重")
    ap.add_argument("--repair", action="store_true", help="只修 numpy/pillow + triton + matplotlib")
    ap.add_argument("--no-upgrade-pip", action="store_true")
    args = ap.parse_args()

    LOG_FILE.write_text("", encoding="utf-8")
    python = _find_python()
    _log(f"[INFO] Python: {python}")
    _log(f"[INFO] Folder: {HERE}")

    if args.repair:
        _fix_triton(python)
        _fix_matplotlib(python)
        _repair(python)
        ok_d = _check_import(python, "depth_anything_3")
        _check_models()
        _log(f"[RESULT] DA3 代码: {'OK' if ok_d else 'FAIL'}")
        return 0 if ok_d else 1

    if args.check_only:
        ok_d = _check_import(python, "depth_anything_3")
        ok_m = _check_models()
        _log(f"[RESULT] DA3 代码: {'OK' if ok_d else 'NOT INSTALLED'} | DA3-L 权重: {'OK' if ok_m else 'MISSING'}")
        return 0 if (ok_d and ok_m) else 1

    # ---- 正常安装: 代码 + 修复 + 权重 ----
    if not args.no_upgrade_pip:
        _run([python, "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip")

    ok_da3 = _check_import(python, "depth_anything_3")
    if not ok_da3:
        if _run([python, "-m", "pip", "install", "--no-deps", DA3_URL], "install DA3 --no-deps"):
            _log("[INFO] 补 DA3 推理最小依赖 (不用 open3d/pycolmap/evo/moviepy)")
            _run([python, "-m", "pip", "install"] + DA3_MIN_DEPS, "install DA3 min deps")
            ok_da3 = _check_import(python, "depth_anything_3")
    else:
        _log("[SKIP] depth_anything_3 已装, 跳过")

    _repair(python)
    _fix_triton(python)
    _fix_matplotlib(python)

    # 权重: 默认自动下载, 除非 --no-download
    ok_model = _check_models()
    if not ok_model and not args.no_download:
        _download_models(python)
        ok_model = _check_models()
    elif not ok_model and args.no_download:
        _log("[MODEL] --no-download, 跳过下载. 要下: 去掉 --no-download 再跑一次.")

    ok_da3 = _check_import(python, "depth_anything_3")
    _log(f"[RESULT] DA3 代码: {'OK' if ok_da3 else 'FAIL'} | DA3-L 权重: {'OK' if ok_model else 'FAIL'} | log: {LOG_FILE}")
    if ok_da3 and ok_model:
        _log("[OK] DA3-L 都好了, 回 GUI 选 DA3-L 再跑.")
        return 0
    _log("[FAIL] 有一个没好, 把 install_da3_l.log 最后 30 行贴回来.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
