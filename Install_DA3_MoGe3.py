#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install_DA3_MoGe3.py v7 -- Easy3D 一键装 DA3 + MoGe3 (完整版)
v1-v6 的坑都已合进: venv_Easy3D / DA3 --no-deps / numpy>=2 pillow<13 / triton 3.2.0.post21 / matplotlib>=3.9
v7 新增: 模型权重检查 + 可选自动下载 (默认只检查, 不偷偷下大文件)

为什么 v7 默认不自动下载:
  3 个权重加起来几个 GB, 双击 BAT 每次都自动下会很久, 还可能把你流量打爆.
  所以默认: 只检查, 缺了就打 [MISSING] + 告诉你怎么下.
  你确认要下: venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --download-models

权重是怎么用的 (看了 wrapper 代码, 不是猜的):
  MoGe (moge_3_model.py:313): 本地有 models/moge-3-vitl/model.pt 就用本地, 没有就
      MoGeModel.from_pretrained("Ruicheng/moge-3-vitl") 自动从 HF 下到 HF 缓存, 不是 models/ 里.
  DA3 (DA_v3_bs.py:334-339): 先看 models/da3-base/ 里有没有 config.json, 有才用本地目录,
      没有就 DepthAnything3.from_pretrained("depth-anything/DA3-BASE"/"DA3-SMALL") 自动从 HF 下到缓存.
      只放 model.safetensors、没有 config.json, 它会忽略你的文件、去 HF 重新下.
  所以 v7 下载用 snapshot_download 下整个仓库 (含 config.json + model.safetensors), 下完 wrapper 才能用本地.

检查哪 3 个 (相对路径, 以脚本所在目录为 Easy3D 根目录):
  models/moge-3-vitl/model.pt  (或同目录下任意 .pt / .safetensors, 只要有一个就算有)
  models/da3-base/model.safetensors (或 da3mono-base.safetensors / config.json)
  models/da3-small/model.safetensors (或 da3mono-small.safetensors / config.json)

用法:
  双击 Install_DA3_MoGe3.bat                      (只装代码 + 检查权重, 不下权重)
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --check-only
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --check-models   (只检查权重, 不装代码)
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --download-models (装代码 + 下缺的权重)
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --repair
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --fix-triton
  venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --fix-matplotlib
"""
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "install_da3_moge3.log"
MOGE_URL = "git+https://github.com/microsoft/MoGe.git"
DA3_URL = "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_MIN_DEPS = ["e3nn", "einops", "omegaconf", "safetensors", "huggingface-hub", "pillow-heif"]
TRITON_PIN = "triton-windows>=3.2.0.post11,<3.3"

# 权重: (本地目录, HF 仓库, 本地算有 的文件名/后缀)
MODEL_WEIGHTS = [
    ("models/moge-3-vitl", "Ruicheng/moge-3-vitl", ("model.pt",)),
    ("models/da3-base", "depth-anything/DA3-BASE", ("model.safetensors", "da3mono-base.safetensors", "config.json")),
    ("models/da3-small", "depth-anything/DA3-SMALL", ("model.safetensors", "da3mono-small.safetensors", "config.json")),
]

def _log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(msg + "\n")
    except Exception: pass

def _find_python():
    for cand in (HERE / "venv_Easy3D" / "Scripts" / "python.exe", HERE / "venv" / "Scripts" / "python.exe", HERE / "venv_Easy3D" / "bin" / "python", HERE / "venv" / "bin" / "python"):
        if cand.exists(): return str(cand)
    return sys.executable

def _run(cmd, label):
    _log(f"[RUN] {label}: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=str(HERE))
        _log(f"[DONE] {label} exit={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        _log(f"[FAIL] {label}: {e}"); return False

def _check_import(python, mod):
    r = subprocess.run([python, "-c", f"import {mod}; print('{mod} ok')"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    ok = r.returncode == 0
    _log(f"[CHECK] import {mod}: {'OK' if ok else 'FAIL'} {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    return ok

def _check_numpy_pillow(python):
    r = subprocess.run([python, "-c", "import numpy, PIL; print(numpy.__version__, PIL.__version__)"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    _log(f"[CHECK] numpy / pillow: {r.stdout.strip() or r.stderr.strip()[-200:]}")
    return r.returncode == 0

def _repair(python):
    _log("[REPAIR] pip install numpy>=2,<3 pillow>=10,<13")
    return _run([python, "-m", "pip", "install", "numpy>=2,<3", "pillow>=10,<13"], "repair numpy/pillow")

def _fix_triton(python):
    _log(f"[FIX-TRITON] pip install {TRITON_PIN}")
    return _run([python, "-m", "pip", "install", TRITON_PIN], "pin triton-windows 3.2.0.post21")

def _fix_matplotlib(python):
    _log("[FIX-MPL] pip install matplotlib>=3.9,<3.10 (NumPy 2 兼容版)")
    return _run([python, "-m", "pip", "install", "matplotlib>=3.9,<3.10"], "upgrade matplotlib for NumPy 2")

def _check_models():
    # 只检查, 不下载. 返回 {本地目录: True/False}
    result = {}
    for rel, repo, names in MODEL_WEIGHTS:
        d = HERE / rel
        found = []
        if d.exists():
            for n in names:
                if (d / n).exists(): found.append(n)
            # MoGe: 同目录任意 .pt / .safetensors 也算有 (你可能改过名)
            if not found and "moge" in rel:
                found = [p.name for p in d.glob("*.pt")] + [p.name for p in d.glob("*.safetensors")]
        ok = bool(found)
        result[rel] = ok
        if ok:
            _log(f"[MODEL] OK {rel} -> {', '.join(found)} (HF: {repo})")
        else:
            _log(f"[MODEL] MISSING {rel} (HF: {repo}) -- 需要: {', '.join(names)}")
    return result

def _download_models(python):
    # 下载缺的权重: snapshot_download 下整个仓库 (含 config.json), 下完 wrapper 才能用本地
    _log("[MODEL] --download-models: 只下缺的, 已有的跳过, 几个 GB, 别关窗口")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        "jobs = ["
        " ('models/moge-3-vitl','Ruicheng/moge-3-vitl'),"
        " ('models/da3-base','depth-anything/DA3-BASE'),"
        " ('models/da3-small','depth-anything/DA3-SMALL'),"
        "]\n"
        "for rel, repo in jobs:\n"
        "    d = Path(rel)\n"
        "    has = d.exists() and any(p.suffix in ('.pt','.safetensors') for p in d.iterdir()) if d.exists() else False\n"
        "    has = has or (d / 'config.json').exists()\n"
        "    if has:\n"
        "        print(f'[MODEL] SKIP 已有 {rel}')\n"
        "        continue\n"
        "    print(f'[MODEL] DOWNLOAD {repo} -> {rel}', flush=True)\n"
        "    snapshot_download(repo_id=repo, local_dir=str(d))\n"
        "    print(f'[MODEL] DONE {rel}')\n"
    )
    return _run([python, "-c", code], "download missing model weights")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--check-models", action="store_true", help="只检查权重, 不装代码")
    ap.add_argument("--download-models", action="store_true", help="装代码 + 下缺的权重 (几个 GB, 显式才下)")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--fix-triton", action="store_true")
    ap.add_argument("--fix-matplotlib", action="store_true")
    ap.add_argument("--skip-moge", action="store_true")
    ap.add_argument("--skip-da3", action="store_true")
    ap.add_argument("--no-upgrade-pip", action="store_true")
    args = ap.parse_args()
    LOG_FILE.write_text("", encoding="utf-8")
    python = _find_python()
    _log(f"[INFO] Python: {python}"); _log(f"[INFO] Folder: {HERE}")

    if args.check_models:
        res = _check_models()
        _log(f"[RESULT] 权重: {sum(res.values())}/{len(res)} 已有")
        return 0 if all(res.values()) else 1
    if args.fix_matplotlib:
        _fix_matplotlib(python); return 0
    if args.fix_triton:
        _fix_triton(python); return 0
    if args.repair:
        _fix_triton(python); _fix_matplotlib(python); _repair(python); _check_numpy_pillow(python)
        ok_m = _check_import(python, "moge"); ok_d = _check_import(python, "depth_anything_3")
        _check_models()
        _log(f"[RESULT] MoGe3: {'OK' if ok_m else 'FAIL'} | DA3: {'OK' if ok_d else 'FAIL'}")
        return 0 if (ok_m and ok_d) else 1

    if not args.check_only:
        git_ok = _run(["git", "--version"], "check git")
        if not git_ok:
            _log("[FAIL] 没找到 git. 先装 git: https://git-scm.com/download/win , 装完重开 cmd 再双击 BAT."); return 1
        if not args.no_upgrade_pip: _run([python, "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip")

    ok_moge = _check_import(python, "moge"); ok_da3 = _check_import(python, "depth_anything_3")
    if args.check_only:
        _check_numpy_pillow(python); _check_models()
        _log(f"[RESULT] MoGe3: {'OK' if ok_moge else 'NOT INSTALLED'} | DA3: {'OK' if ok_da3 else 'NOT INSTALLED'}")
        return 0 if (ok_moge and ok_da3) else 1

    if not args.skip_moge and not ok_moge:
        if _run([python, "-m", "pip", "install", MOGE_URL], "install MoGe3"): ok_moge = _check_import(python, "moge")
    elif not args.skip_moge: _log("[SKIP] MoGe3 已装, 跳过")
    if not args.skip_da3 and not ok_da3:
        if _run([python, "-m", "pip", "install", "--no-deps", DA3_URL], "install DA3 --no-deps"):
            _log("[INFO] 补 DA3 推理最小依赖 (不用 open3d/pycolmap/evo/moviepy)")
            _run([python, "-m", "pip", "install"] + DA3_MIN_DEPS, "install DA3 min deps")
            ok_da3 = _check_import(python, "depth_anything_3")
    elif not args.skip_da3: _log("[SKIP] DA3 已装, 跳过")

    _repair(python); _fix_triton(python); _fix_matplotlib(python); _check_numpy_pillow(python)
    ok_moge = _check_import(python, "moge"); ok_da3 = _check_import(python, "depth_anything_3")
    # 权重: 默认只检查, 不自动下; --download-models 才下
    model_ok = _check_models()
    if args.download_models:
        _download_models(python)
        model_ok = _check_models()
    elif not all(model_ok.values()):
        _log("[MODEL] 有权重缺失, 但默认不自动下. 要下: venv_Easy3D\\Scripts\\python.exe Install_DA3_MoGe3.py --download-models")
    _log(f"[RESULT] MoGe3: {'OK' if ok_moge else 'FAIL'} | DA3: {'OK' if ok_da3 else 'FAIL'} | 权重: {sum(model_ok.values())}/{len(model_ok)} | log: {LOG_FILE}")
    if ok_moge and ok_da3:
        _log("[OK] 代码都装好了. 权重缺的先下, 再回 GUI 选 MoGe3-ViTL / DA3-B 再跑.")
        return 0
    _log("[FAIL] 有一个没装好, 把 install_da3_moge3.log 最后 30 行贴回来."); return 1

if __name__ == "__main__": raise SystemExit(main())
