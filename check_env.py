# Easy3D V10.8.1 - check_env.py
# 它只检查, 不装东西, 不改环境。先跑它, 把最后一行 [Easy3D-Check] 发回来, 不要只发 pip list。
# pip list cannot show import errors: 它只列名字, 不看 import 能不能成功, 也不看版本配不配。
import sys, importlib
print(f"[Easy3D-Check] Python: {sys.version.split()[0]}  exe={sys.executable}")
print("[Easy3D-Check] 注意: 必须用同一个 Python 跑 pip 和跑 GUI。")
print("[Easy3D-Check] 如果 pip list 里有 Pillow 但 GUI 报 No module named PIL, 就是两个 Python 不是同一个。")
# (名字, pip 包名, 必须? )
NEEDS = [
    ("PIL", "pillow", True), ("cv2", "opencv-python", True), ("numpy", "numpy", True),
    ("torch", "torch", True), ("torchvision", "torchvision", True),
    ("diffusers", "diffusers", True), ("transformers", "transformers", True),
    ("huggingface_hub", "huggingface_hub", True), ("safetensors", "safetensors", True),
    ("accelerate", "accelerate", True), ("peft", "peft", True),
    ("einops", "einops", True), ("timm", "timm", True), ("omegaconf", "omegaconf", True),
    ("yaml", "PyYAML", True), ("tqdm", "tqdm", True), ("scipy", "scipy", True),
    ("skimage", "scikit-image", True), ("matplotlib", "matplotlib", True),
    ("tkinterdnd2", "tkinterdnd2", True),
    ("psutil", "psutil", False), ("requests", "requests", False), ("pydub", "pydub", False),
]
missing = []
for mod, pip_name, required in NEEDS:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"[Easy3D-Check] OK   {mod} {v}  (pip: {pip_name})")
    except Exception as e:
        tag = "MISS" if required else "MISS(可选)"
        print(f"[Easy3D-Check] {tag} {mod} (pip: {pip_name}) -> {type(e).__name__}: {e}")
        if required: missing.append(pip_name)
# 版本搭配: transformers 5.x + diffusers 0.31 是已知会炸的组合
try:
    import transformers, diffusers
    tv, dv = transformers.__version__, diffusers.__version__
    print(f"[Easy3D-Check] transformers={tv} diffusers={dv}")
    if tv.split(".")[0] >= "5":
        print("[Easy3D-Check] !!! transformers 5.x + diffusers 0.31 不兼容, GenStereo import 大概率会炸。必须降到 transformers 4.x, 见 INSTALL_FIX.bat。")
        if "transformers" not in missing: missing.append("transformers==4.57.1")
except Exception as e:
    print(f"[Easy3D-Check] transformers/diffusers 版本读不到: {e}")
try:
    import torch
    print(f"[Easy3D-Check] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"[Easy3D-Check] torch 读不到: {e}")
if missing:
    print(f"[Easy3D-Check] 结论: 缺 / 错: {', '.join(missing)} -> 先跑 INSTALL_FIX.bat, 再跑 check_env.py, 都 OK 再开 GUI。")
else:
    print("[Easy3D-Check] 结论: 必需的包都 import 成功, 环境 OK, 可以开 GUI。如果 GUI 还报错, 把 GUI 弹出的那行 Traceback 原样发回来, 不要只发 pip list。")