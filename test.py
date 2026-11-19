# FIX v130: 锐化 -L 490
# FIX decoupled_warp_orig_v2: Warp 在原图上做, synth/mask 也放到原图尺寸再 Fusion -L 486
# FIX v71: inpaint 覆盖 +1% (base 0.55->0.5666)

# FIX v64: 宽图手 - 前景局部 dx(梯度) + 水平多扩 + 宽图 cap 100

# FIX v36: 均匀扩 -> 每个像素自适应扩 (base*0.55, 按 local |warped-fused| 在 0.5x..2.5x 浮动) 

# FIX v34: 拿掉 clamp + 羽化只 3px + 智能扩 (IMAGE_SIZE*scale*0.35, 5..40) -L 673 -L 722
# FIX v33: Tony hole 先扩 3px 再羽化 + V53 charmap 兜底

# FIX v29: warped==orig 时不混原图, diff 大处全用 fused
# FIX v28: decoupled 里 _apply 第二个参数必须是 warped, 不是原图左眼
# FIX v27: t 从 GENSTEREO_FUSION_STRENGTH 读 + mask 极性 + 羽化 -L 673 -L 722
# FIX v26: Fusion 权重图高斯羽化, <1 不再硬过渡

# FIX v25: Fast - Decoupled-L 473
from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import ssl
import os
import sys as _sys_stdout
try:
    _sys_stdout.stdout.reconfigure(encoding="utf-8", errors="ignore")
    _sys_stdout.stderr.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass  # FIX v77: Windows cp1252 下中文 print 会 UnicodeEncodeError, 强制 utf-8
import random
from extern.DAM2.depth_anything_v2.dpt import DepthAnythingV2
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import argparse

from GenStereo import GenStereo, AdaptiveFusionLayer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # v6.0 fix: 关掉 diffusers 的 scale FutureWarning，Log

# Easy3D v1.9: SD version from env (GUI) or --sd_version, default v1.5
SD_VERSION = os.environ.get("GENSTEREO_VERSION", "v1.5")
_ORIG_LONG = 0  # FIX v15: 原图长边, 在 infer_depth 打开原图时记下 
if SD_VERSION == "v1.5":
    IMAGE_SIZE = 512
    CHECKPOINT_NAME = 'genstereo-v1.5'
elif SD_VERSION == "v2.1":
    IMAGE_SIZE = 768
    CHECKPOINT_NAME = 'genstereo-v2.1'
else:
    raise ValueError(f"Unknown SD version: {SD_VERSION}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# v3.2 LOSSLESS: TF32 (3070 Tensor Core, no visible quality change)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False  #原值 True
        print("[Easy3D] TF32 ON (lossless speedup, 3070 Tensor Core)")
    except Exception as _e:
        print(f"[Easy3D] TF32 not enabled: {_e}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# MDE dispatch (v3.4 + FlashDepth): GUI passes GENSTEREO_MDE_MODEL + GENSTEREO_MDE_PATH.
#   DAM2:      GENSTEREO_MDE_MODEL=depth_anything_v2_vitl/vitb/vits, path=models/depth_anything_v2_*.pth
#   FlashDepth: GENSTEREO_MDE_MODEL=flashdepth-l/flashdepth-s, path=models/FlashDepth/configs/flashdepth-*/iter_*.pth
# All paths relative to Easy3D folder (this file's folder). No absolute paths.
# FlashDepth is lazy-loaded: DAM2 is NOT loaded when FlashDepth is selected (and vice versa).
_mde_model_env = os.environ.get("GENSTEREO_MDE_MODEL", "flashdepth-l").strip().lower()  # FIX v20: 恢复 .strip().lower(), V18 的 # 把它注释掉了
_mde_path_env = os.environ.get("GENSTEREO_MDE_PATH", "models/FlashDepth/configs/flashdepth-l/iter_10001.pth")  # FIX v18: 默认 MDE 路径 -> FlashDepth-L
IS_FLASHDEPTH = _mde_model_env in ("flashdepth-l", "flashdepth-s", "flashdepth")
# FIX v84: 新模型 (DA3 / MoGe) 不能走下面 DAM2 分支. DAM2 要的是 .pth 文件, DA3 要的是目录, 混了就会 torch.load('models/da3-large') -> PermissionError.
IS_NEW_MDE = _mde_model_env.startswith(("da3", "moge", "any_v3", "any-v3", "depth_anything_v3", "distill")) or _mde_model_env in ("da3-large", "da3_large", "da3-base", "da3-small", "moge-3-vitl", "moge-3-vitg", "moge3_vitl", "moge3_vitg")
# FlashDepth / 新模型都不走 DAM2, checkpoint_dir 强制回 models/ (别变成 models/da3-large)
if IS_FLASHDEPTH or IS_NEW_MDE:
    checkpoint_dir = "models"

# FIX v76: 新 MDE. DA3-S/B + MoGe3-ViTL 不是 FlashDepth, 也不是 DAM2, 必须单独分流, 否则会掉进 DAM2 分支炸掉.
# DA3 在 GUI 里传的是 models/da3-small|da3-base (文件夹), MoGe3 传的是 models/moge-3-vitl/model.pt (文件), 统一成文件夹供 wrapper 用
_NEW_MDE_DIR = {"da3-small": "models/da3-small", "da3-base": "models/da3-base", "moge-3-vitl": "models/moge-3-vitl", "moge-3-vitg": "models/moge-3-vitg", "da3-large": "models/da3-large", "distill_any_depth_vitb": "models/distill_any_depth_vitb.safetensors", "da3_large": "models/da3-large"}.get(_mde_model_env, _mde_path_env)
# Map GUI lowercase -> flashdepth_model.py model_type ("FlashDepth-L" / "FlashDepth-S")
FLASHDEPTH_TYPE = {"flashdepth-l": "FlashDepth-L", "flashdepth-s": "FlashDepth-S", "flashdepth": "FlashDepth"}.get(_mde_model_env, "FlashDepth-L")  # FIX v18: 兜底 FlashDepth-S -> FlashDepth-L
# checkpoint_dir = folder that holds sd-vae-ft-mse / image_encoder / genstereo-v1.5 = models/
checkpoint_dir = os.path.dirname(_mde_path_env) or "models"
# For FlashDepth, checkpoint_dir for GenStereo must stay models/ (not models/FlashDepth/...)
if IS_FLASHDEPTH or IS_NEW_MDE:
    checkpoint_dir = "models"  # FIX v76: 新 MDE 也一样, GenStereo 的 fusion 在 models/ 里, 不能跟 MDE 路径走
os.makedirs(checkpoint_dir, exist_ok=True)
dam2 = None
_flashdepth_wrapper = None
if not IS_FLASHDEPTH and not IS_NEW_MDE:
    # DAM2 only (FIX v76: DA3/MoGe3 不走这里, 走 _get_new_mde_wrapper)
    encoder = 'vitl'
    for _cand in ('vitl', 'vitb', 'vits'):
        if _cand in _mde_path_env.lower():
            encoder = _cand
            break
    encoder_size_map = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}
    dam2 = DepthAnythingV2(**model_configs[encoder])
    size_name = encoder_size_map[encoder]
    dam2_path = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{size_name}/resolve/main/depth_anything_v2_{encoder}.pth"
    dam2_checkpoint = _mde_path_env
    if not os.path.exists(dam2_checkpoint):
        print(f"[Easy3D] DAM2 model not found: {dam2_checkpoint} - put it in models/ (see download_models.py), will NOT wget on Windows")
    dam2.load_state_dict(torch.load(dam2_checkpoint, map_location='cpu'))
    dam2 = dam2.to(DEVICE).eval()
    print(f"[Easy3D] MDE: DepthAnythingV2 {encoder} ({dam2_checkpoint})")
else:
    # FIX v77: else 里有 FlashDepth 和新 MDE 两种, 不能都打 FlashDepth
    if IS_FLASHDEPTH:
        print(f"[Easy3D] MDE: FlashDepth {FLASHDEPTH_TYPE} ({_mde_path_env}) - lazy load on first image")
    elif IS_NEW_MDE:
        print(f"[Easy3D] MDE: New {_mde_model_env} ({_mde_path_env}) - lazy load on first image")
    else:
        print(f"[Easy3D] MDE: {_mde_model_env} ({_mde_path_env}) - lazy load on first image")

# --- FIX v82 Dilation: GUI 传 GENSTEREO_DILATION (0-6 整数, 默认 2), test.py 读它再传给各模型 ---
def _get_edge_dilation():
    try:
        import os as _os
        for _k in ("GENSTEREO_DILATION", "GENSTEREO_EDGE_DILATION", "EDGE_DILATION"):
            _v = _os.environ.get(_k, "")
            if str(_v).strip() != "":
                return max(0, min(50, int(float(_v))))
    except Exception:
        pass
    return 4
_EDGE_DILATION = _get_edge_dilation()
def _call_mde_with_dilation(wrapper, image, image_size=None, **kwargs):
    """统一调 wrapper.infer: 支持 edge_dilation 的传, 不支持的 (FlashDepth/DA2) 回退到不传, 不炸."""
    _ed = _get_edge_dilation()
    try:
        return wrapper.infer(image, edge_dilation=_ed, **kwargs)
    except TypeError as _e:
        if "edge_dilation" in str(_e) or "unexpected keyword" in str(_e) or "got an unexpected" in str(_e):
            return wrapper.infer(image, **kwargs)
        raise
    except Exception:
        try:
            return wrapper.infer(image, **kwargs)
        except Exception:
            raise

def _v53_correct_disparity_np(warped_np, fused_np, img_np):
    # V55 helper: numpy 版, 先把 fused 从 -1..1 归一到 0..1 再比, 否则 max 永远 1.0
    try:
        import numpy as _np
        import cv2 as _cv2
        _w = warped_np.astype(_np.float32)
        _f = fused_np.astype(_np.float32)
        _i = img_np.astype(_np.float32)
        _ws = _cv2.resize(_w, (224, 224), interpolation=_cv2.INTER_LINEAR)
        _fs = _cv2.resize(_f, (224, 224), interpolation=_cv2.INTER_LINEAR)
        _is = _cv2.resize(_i, (224, 224), interpolation=_cv2.INTER_LINEAR)
        _best_k = 1.0
        _best_err = float("inf")
        _err_k1 = -1.0
        for _k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            _wk = _is * (1.0 - _k) + _ws * _k
            _err = float(_np.abs(_wk - _fs).mean())
            if _k == 1.0:
                _err_k1 = _err
            if _err < _best_err:
                _best_err = _err
                _best_k = _k
        print("[Fusion-V53] disparity scale k=" + str(_best_k) + " err_k1=" + str(round(_err_k1, 4)) + " err_best=" + str(round(_best_err, 4)) + " (k<1 => warped disparity > fused, shrink)", flush=True)
        if _best_k < 0.99:
            _out = _i * (1.0 - _best_k) + _w * _best_k
            print("[Fusion-V53] warped disparity scaled by k=" + str(_best_k), flush=True)
            return _out.astype(_np.float32)
        return warped_np
    except Exception as _e:
        print("[Fusion-V53] align failed: " + str(_e), flush=True)
        return warped_np


def _get_flashdepth_wrapper(resolution=None):
    global _flashdepth_wrapper
    if _flashdepth_wrapper is not None:
        # Resolution can change per batch (GUI Resolution slider) -> update prep_lower_bound
        if resolution:
            try:
                _r = int(resolution)
                _r -= _r % 14
                _r = max(14, _r)
                _flashdepth_wrapper.model.prep_lower_bound = _r
            except Exception:
                pass
        return _flashdepth_wrapper
    # flashdepth_model.py is in the same folder as this test.py (relative import, no absolute path)
    try:
        from flashdepth_model import FlashDepthModelWrapper
    except Exception as e:
        raise RuntimeError(f"FlashDepth selected but flashdepth_model.py not found next to test.py: {e}")
    import torch as _torch
    _device = _torch.device(DEVICE)
    _w = FlashDepthModelWrapper(FLASHDEPTH_TYPE)
    _w.device = _device
    _w.load_model(FLASHDEPTH_TYPE, resolution=resolution, device=_device)  # FIX res2: first image must also use GUI Resolution, not default 518/392
    _flashdepth_wrapper = _w
    return _w

# ---- FIX v76: 新 MDE 懒加载 (DA3-S/B + MoGe3-ViTL), 失败返回 None, 上层回退 FD-L, 不崩 ----
_new_mde_wrapper = None
def _get_new_mde_wrapper(resolution=None):
    # FIX v81 DA3-L: 早返回, 不要掉进 DA_v3_bs (BS 只认 Small/Base, Large 会错)
    try:
        _mtype_l = str(_mde_model_env).strip().lower()  # FIX v86: 看模型名, 不是 resolution (resolution 是 768/2048 数字, 永远进不了 da3-large)
    except Exception:
        _mtype_l = ""
    if _mtype_l in ("da3-large", "da3_large", "any_v3_mono_large", "any_v3_mono_larg", "da3-l"):
        # FIX v87: DA3-L 必须调 load_model() 才有 self.model, 只建 _DA3LModel() 不加载, infer() 会报 has no attribute 'model'
        try:
            from models.DA_v3_l import DepthAnythingV3LargeModel as _DA3LModel
        except Exception:
            from DA_v3_l import DepthAnythingV3LargeModel as _DA3LModel
        import torch as _torch_l
        _device_l = _torch_l.device(DEVICE)
        _m_l = _DA3LModel("da3-large")  # FIX v87: 第一个参数是 model_type, 不是 resolution (原来传 resolution=2048, model_type 变成 2048)
        _m_l.load_model("da3-large", resolution=resolution, device=_device_l)
        _m_l.device = _device_l
        globals()["_new_mde_wrapper"] = _m_l  # 缓存, 别每张图重加载
        print(f"[Easy3D] MDE: DA3 da3-large ({_NEW_MDE_DIR}) - via DA_v3_l.py", flush=True)
        return _m_l
    global _new_mde_wrapper
    if _new_mde_wrapper is not None:
        return _new_mde_wrapper
    import torch as _torch
    _device = _torch.device(DEVICE)
    if _mde_model_env in ("da3-small", "da3-base"):
        # DA3: models/DA_v3_bs.py 里的 DepthAnythingV3BSModel, 权重在 models/da3-small|da3-base/
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (_here, os.path.join(_here, "models")):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from models.DA_v3_bs import DepthAnythingV3BSModel
        _mtype = "Any_V3_Mono_Small" if _mde_model_env == "da3-small" else "Any_V3_Mono_Base"
        _m = DepthAnythingV3BSModel(_mtype)
        _m.load_model(_mtype, resolution=resolution, device=_device)
        _m.device = _device
        _new_mde_wrapper = _m
        print(f"[Easy3D] MDE: DA3 {_mde_model_env} ({_NEW_MDE_DIR}) - via DA_v3_bs.py", flush=True)
        return _m
    if _mde_model_env == "moge-3-vitl":
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (_here, os.path.join(_here, "models")):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from models.moge_3_model import MoGe3Model
        _m = MoGe3Model("moge-3-vitl")
        _loaded = _m.load_model("moge-3-vitl", device=_device)
        # moge_3_model.load_model 返回 model 但没赋给 self.model, 这里补上 (不改原文件)
        _m.model = _loaded if _loaded is not None else getattr(_m, "model", None)
        _m.device = _device
        _new_mde_wrapper = _m
        print(f"[Easy3D] MDE: MoGe3-ViTL ({_NEW_MDE_DIR}) - via moge_3_model.py", flush=True)
        return _m
    if _mde_model_env == "moge-3-vitg":
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (_here, os.path.join(_here, "models")):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from models.moge_3_model import MoGe3Model
        _m = MoGe3Model("moge-3-vitg")
        _loaded = _m.load_model("moge-3-vitg", device=_device)
        # moge_3_model.load_model 返回 model 但没赋给 self.model, 这里补上 (不改原文件)
        _m.model = _loaded if _loaded is not None else getattr(_m, "model", None)
        _m.device = _device
        _new_mde_wrapper = _m
        print(f"[Easy3D] MDE: MoGe3-ViTG ({_NEW_MDE_DIR}) - via moge_3_model.py", flush=True)
        return _m
    if _mde_model_env in ("distill_any_depth_vitb", "distill", "distill-b", "distill_any_depth", "distill_any_depth_vitb".replace("_vitb", "")):
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in (_here, os.path.join(_here, "models")):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from models.Distill_model import DistillAnyDepthModel
        _m = DistillAnyDepthModel("distill_any_depth_vitb")
        _m.load_model("distill_any_depth_vitb", resolution=resolution, device=_device)
        _m.device = _device
        globals()["_new_mde_wrapper"] = _m
        print(f"[Distill] via Distill_model.py ({_NEW_MDE_DIR})", flush=True)
        return _m
    return None

def infer_depth_new(image_path: str, image_size: int):
    # 新 MDE: 输入 PIL RGB, 输出 [1,1,H,W] float on DEVICE, 和 FlashDepth/DAM2 路径同格式, 下游 Warp/Fusion 不用改
    image = Image.open(image_path).convert('RGB')
    global _ORIG_LONG
    _ow, _oh = image.size
    _ORIG_LONG = max(_ow, _oh)
    if (_ow, _oh) != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    try:
        _w = _get_new_mde_wrapper(resolution=image_size)
    except Exception as _e:
        print(f"[Easy3D] New MDE {_mde_model_env} load FAILED ({_e}), fallback to FD-L. Need: pip install git+https://github.com/microsoft/MoGe.git and git+https://github.com/ByteDance-Seed/Depth-Anything-3.git", flush=True)
        raise
    _ed = _get_edge_dilation()  # FIX v88: Edge Protection Level, 原来这里没传, DA3-B 永远 0, MoGe 永远 2
    print(f"[Easy3D] Edge Protection Level = {_ed} (GENSTEREO_DILATION)", flush=True)
    try:
        if str(_mde_model_env).lower().startswith("moge"):
            # FIX v89: MoGe 内部 dilate 是裸 except: pass, 炸了也看不见, 这里关掉内部那次, 出来后统一做一次
            depth = _w.infer(image, resolution=image_size, edge_dilation=0)  # FIX v91: 膨胀移到归一化之后, 这里先不膨
        else:
            depth = _w.infer(image, resolution=image_size, edge_dilation=_ed)

    except TypeError:
        depth = _w.infer(image, resolution=image_size)  # 旧 wrapper 不认 edge_dilation, 回退

    if not torch.is_tensor(depth):
        depth = torch.tensor(depth)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(0)
    # DA3/MoGe3 有的已归一化, 再归一化一次无害, 保证和 FD/DA2 同格式
    lo = depth.amin(dim=(-2, -1), keepdim=True); hi = depth.amax(dim=(-2, -1), keepdim=True)
    depth = (depth - lo) / (hi - lo + 1e-6)
    # FIX v91: MoGe 的膨胀必须在归一化之后, 否则膨胀完一归一化就被压回去. DA3 不走这里 (内部已做).
    if str(_mde_model_env).lower().startswith("moge"):
        depth = _apply_edge_protection(depth, "MoGe3")
    return image, depth.float().clamp(0, 1).to(DEVICE)

# GenStereo config - filled in __main__ from CLI args (steps/guidance/seed).
# Defaults here so `import test` still works.
genstereo_cfg = dict(
    pretrained_model_path=checkpoint_dir,
    checkpoint_name=CHECKPOINT_NAME,
    half_precision_weights=True if 'cuda' in DEVICE else False,
    num_inference_steps=50,
    guidance_scale=1.5,
)
genstereo_nvs = GenStereo(cfg=genstereo_cfg, device=DEVICE, sd_version=SD_VERSION)
# v3.2: xFormers on denoising_unet + reference_unet if installed, else SDPA/flash fallback
try:
    _ok = 0
    for _m in (getattr(genstereo_nvs, 'denoising_unet', None), getattr(genstereo_nvs, 'reference_unet', None)):
        if _m is not None and hasattr(_m, 'enable_xformers_memory_efficient_attention'):
            _m.enable_xformers_memory_efficient_attention(); _ok += 1
    if _ok:
        print(f"[Easy3D] xFormers attention ON ({_ok}/2 UNets)")
    else:
        print("[Easy3D] xFormers not available, using PyTorch SDPA/flash")
except Exception as _e:
    print(f"[Easy3D] xFormers not used, using SDPA/flash: {_e}")

fusion_model = AdaptiveFusionLayer()
fusion_checkpoint = join(checkpoint_dir, CHECKPOINT_NAME, 'fusion_layer.pth')
fusion_model.load_state_dict(torch.load(fusion_checkpoint, map_location='cpu'))  # V67: 先放 CPU, 避免 GPU OOM 静默退出
print("[Easy3D] Fusion loaded (V67 checkpoint)", flush=True)
try:
    import torch as _torch_v68
    _torch_v68.cuda.empty_cache()
    fusion_model = fusion_model.to(DEVICE).eval()
    print("[Easy3D] Fusion on " + str(DEVICE) + " (V68)", flush=True)
except Exception as _e_v68:
    print("[Easy3D] Fusion to " + str(DEVICE) + " failed (OOM?), keep on CPU: " + str(_e_v68), flush=True)
    fusion_model = fusion_model.to('cpu').eval()



def infer_depth_flashdepth(image_path: str, image_size: int):
    # FlashDepth: input is PIL RGB, output -> [1,1,H,W] float on DEVICE (same shape as DAM2 path)
    # flashdepth_model.py infer() already does preprocess + clean_shallow_dots internally.
    image = Image.open(image_path).convert('RGB')
    global _ORIG_LONG
    _ow, _oh = image.size
    _ORIG_LONG = max(_ow, _oh)  # FIX v15: 记原图长边
    if (_ow, _oh) != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        pass  # FIX v21: Log 
    wrapper = _get_flashdepth_wrapper(resolution=image_size)  # GUI Resolution -> FlashDepth prep
    try:
        print(f"[FlashDepth] prep_lower_bound={getattr(wrapper.model, 'prep_lower_bound', '?')} (GUI Resolution={image_size}, model={FLASHDEPTH_TYPE})")
    except Exception: pass
    # FlashDepth infer returns [H,W] or [B,H,W]; normalize to [1,1,H,W]
    depth = _call_mde_with_dilation(wrapper, image)
    if not torch.is_tensor(depth):
        depth = torch.tensor(depth)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(0)
    # FlashDepth depth is already in [0,1]-ish / relative range; keep as-is, generate_novel_view normalizes it
    return image, depth.float().to(DEVICE)

def infer_depth(image_path: str, image_size: int):
    # Unified entry: dissteps by GENSTEREO_MDE_MODEL
    if IS_NEW_MDE:
        try:
            return infer_depth_new(image_path, image_size)
        except Exception as _e_new:
            print(f"[Easy3D] New MDE {_mde_model_env} infer FAILED ({_e_new}), fallback to FlashDepth", flush=True)
            return infer_depth_flashdepth(image_path, image_size)
    if IS_FLASHDEPTH:
        return infer_depth_flashdepth(image_path, image_size)
    return infer_depth_dam2(image_path, image_size)

def _apply_edge_protection(depth, label):  # FIX v91: import 按文件路径找, 不靠模块名; 打印 depth changed
    _ed = _get_edge_dilation() if "_get_edge_dilation" in globals() else 2
    print(f"[Easy3D] Edge Protection Level = {_ed} (GENSTEREO_DILATION) -> {label}", flush=True)
    if not _ed:
        print(f"[Easy3D] Edge Protection skip {label}: level 0", flush=True)
        return depth
    try:
        import importlib.util as _ilu
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _root = _os.path.dirname(_here) if _os.path.basename(_here).lower() == "models" else _here
        _cands = [
            _os.path.join(_here, "dilation.py"), _os.path.join(_here, "edge_dilation.py"),
            _os.path.join(_here, "models", "dilation.py"), _os.path.join(_here, "models", "edge_dilation.py"),
            _os.path.join(_root, "models", "dilation.py"), _os.path.join(_root, "models", "edge_dilation.py"),
            _os.path.join(_root, "dilation.py"), _os.path.join(_root, "edge_dilation.py"),
        ]
        _dilate_edge = None
        _used = ""
        for _fp in _cands:
            if _os.path.isfile(_fp):
                try:
                    _spec = _ilu.spec_from_file_location("_easy3d_dilation_" + _os.path.basename(_fp).split(".")[0], _fp)
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    if hasattr(_mod, "dilate_edge"):
                        _dilate_edge = _mod.dilate_edge
                        _used = _fp
                        break
                except Exception:
                    continue
        if _dilate_edge is None:
            print(f"[Easy3D] Edge Protection FAILED {label}: 没找到 dilation.py / edge_dilation.py, 试了 {_cands}", flush=True)
            return depth
        _in_ndim = depth.ndim
        _d = depth
        if _d.ndim == 2:
            _d = _d.unsqueeze(0).unsqueeze(0)
        elif _d.ndim == 3:
            _d = _d.unsqueeze(0)
        _before = _d.detach().clone()
        _d = _dilate_edge(_d, _ed)
        try:
            _diff = (_d - _before).abs()
            print(f"[Easy3D] Edge Protection applied to {label}: level {_ed}, from {_used}, depth changed max={float(_diff.max()):.4f} mean={float(_diff.mean()):.4f}", flush=True)
        except Exception:
            print(f"[Easy3D] Edge Protection applied to {label}: level {_ed}, from {_used}", flush=True)
        if _in_ndim == 2:
            _d = _d.squeeze(0).squeeze(0)
        elif _in_ndim == 3:
            _d = _d.squeeze(0)
        return _d
    except Exception as _e:
        print(f"[Easy3D] Edge Protection FAILED {label}: {_e} (level {_ed} 没生效)", flush=True)
        return depth


def infer_depth_dam2(image_path: str, image_size: int):
    # v3.1 (SD3D-style): NEVER crop, NEVER letterbox.
    image = Image.open(image_path).convert('RGB')
    global _ORIG_LONG
    _ow, _oh = image.size
    _ORIG_LONG = max(_ow, _oh)  # FIX v15: 先记原图长边
    if (_ow, _oh) != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        pass  # FIX v21: resize 
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    depth_dam2 = dam2.infer_image(image_bgr)
    # FIX v89: DA2 原来这里直接 return, 从来没读过 Edge Protection Level, 在这里统一做一次
    _depth_da2 = torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    _depth_da2 = _apply_edge_protection(_depth_da2, "DA2")
    return image, _depth_da2

FUSION_STRENGTH = float(__import__("os").environ.get("GENSTEREO_FUSION_STRENGTH", "1.0") or 1.0)  # FIX v28: 1.0=老样子(纯AdaptiveFusion), 0.0=纯Warp, 0.5=一半一半
# FIX v29: Decoupled A/B/C 扩散分辨率, 在这里改
DECOUPLED_LOW_RES = {"decoupled_a": 896, "decoupled_b": 1280, "decoupled_c": 1344, "decoupled_d": 1568, "decoupled_e": 1792, "decoupled_f": 2016, "decoupled": 896}  # FIX v132: E=1792, F=2016

# FIX v130: 新图锐化, 在这里改, 最明显的位置
# 0 = 关, 0.4 = 轻微 (原值), 1.0 = 强. 只锐化新生成的右/左眼, 不锐化原图左眼
SHARPEN_NEW_VIEW = 0.6


def _sharpen_new_view(img_np, strength=None):
    """FIX v130: Unsharp Mask, Fusion 完、合 SBS 前、已缩到原图尺寸时做. img_np: HxWx3, 0..1 float."""
    try:
        _s = float(SHARPEN_NEW_VIEW) if strength is None else float(strength)
    except Exception:
        _s = 0.4
    if not _s or _s <= 0:
        return img_np
    try:
        import cv2 as _cv2
        _blur = _cv2.GaussianBlur(img_np.astype("float32"), (0, 0), sigmaX=1.0)
        _out = img_np.astype("float32") + _s * (img_np.astype("float32") - _blur)
        print(f"[Easy3D-Sharpen] 新图锐化 {_s} (Fusion 完、合 SBS 前、原图尺寸)", flush=True)
        return _out.clip(0, 1).astype("float32")
    except Exception as _e:
        print(f"[Easy3D-Sharpen] 锐化失败, 跳过: {_e}", flush=True)
        return img_np
  #适合的值有672,896,1120,1344 1568,1792,2016,2240,2464,2688 
def _fix_outer_edge(img_np, band=None):
    """FIX v134: 回退 v131/v133, 外圈不动."""
    return img_np  # FIX v134: 回退, 外圈原样返回, 不 REPLICATE 不 REFLECT
    try:
        import cv2 as _cv2
        _a = img_np.astype("float32")
        if _a.ndim != 3 or _a.shape[0] < 64 or _a.shape[1] < 64:
            return img_np
        _h, _w = _a.shape[:2]
        _b = int(band) if band else 48
        _b = min(_b, _w // 4, _h // 4)
        _inner = _a[_b:_h - _b, _b:_w - _b]
        # FIX v133: REPLICATE -> REFLECT_101, 拉长变镜像, 不出横向拖影
        _fixed = _cv2.copyMakeBorder(_inner, _b, _b, _b, _b, _cv2.BORDER_REFLECT_101)
        # FIX v133: 羽化, 外圈权重 1 -> 内侧交界 0, 余弦渐变, 不留硬缝
        import numpy as _np_v133
        _yy, _xx = _np_v133.mgrid[0:_h, 0:_w].astype("float32")
        _d = _np_v133.minimum(_np_v133.minimum(_xx, _w - 1 - _xx), _np_v133.minimum(_yy, _h - 1 - _yy))  # 到最近外边的距离
        _wgt = _np_v133.clip(1.0 - _d / max(1, _b), 0, 1)  # 外边=1, 交界=0
        _wgt = (0.5 - 0.5 * _np_v133.cos(_wgt * _np_v133.pi))[..., None]  # 余弦平滑
        _out = (_a * (1.0 - _wgt) + _fixed * _wgt).astype("float32")
        return _out.astype("float32")
    except Exception as _e:
        print(f"[Easy3D-OuterEdge] v131 失败, 跳过: {_e}", flush=True)
        return img_np

def _fix_warped_outer_black(warped_np, mask_np, band=48):
    # FIX v136: 同 v124, 只填 warped 里 mask=0 的洞. warped: HxWx3 0..1, mask: HxW 0..1 (1=有像素).
    try:
        import cv2 as _cv2
        import numpy as _np
        _w = warped_np.astype('float32')
        _m = mask_np.astype('float32')
        if _m.ndim == 3:
            _m = _m[..., 0]
        _h, _wdt = _w.shape[:2]
        if _w.ndim != 3 or _h < 64 or _wdt < 64:
            return warped_np
        _b = min(int(band), _wdt // 4, _h // 4)
        _yy, _xx = _np.mgrid[0:_h, 0:_wdt]
        _dist = _np.minimum(_np.minimum(_xx, _wdt - 1 - _xx), _np.minimum(_yy, _h - 1 - _yy))
        _hole_outer = (_m < 0.5) & (_dist < _b)
        if not _hole_outer.any():
            return warped_np
        _inner = _w[_b:_h - _b, _b:_wdt - _b]
        _fill = _cv2.copyMakeBorder(_inner, _b, _b, _b, _b, _cv2.BORDER_REPLICATE)
        _out = _w.copy()
        _out[_hole_outer] = _fill[_hole_outer]
        print(f'[Easy3D-OuterEdge] v136: warped 外圈洞 {int(_hole_outer.sum())}px 已用边缘填 (只填洞)', flush=True)
        return _out.astype('float32')
    except Exception as _e:
        print(f'[Easy3D-OuterEdge] v136 失败, 跳过: {_e}', flush=True)
        return warped_np

def _apply_fusion_strength(fused, warped, mask, t=None):
    # FIX v49: 先对位移, 再混, 不靠羽化遮丑
    # 原理: warped (FlashDepth 896, scale 0.0759) 与 fused (GenStereo 768 扩散) 是两套位移, 手差十几 px, 不对位移直接混就是双影, 羽化只是把硬边糊掉
    # V49: 在有效区搜最优全局 dx (-24..24), 让 warped 平移后与 fused 最像, 再用 t 诚实混, 不推 w, 不作弊
    import os as _os
    import torch as _torch
    import torch.nn.functional as _F
    _tt = None
    try:
        _tt = float(t) if t is not None else None
    except Exception:
        _tt = None
    _env = _os.environ.get("GENSTEREO_FUSION_STRENGTH", "") or _os.environ.get("FUSION_STRENGTH", "")
    _env_f = None
    try:
        _env_f = float(_env) if str(_env).strip() != "" else None
    except Exception:
        _env_f = None
    if _tt is None:
        _tt = _env_f if _env_f is not None else 1.0
        _t_src = "env" if _env_f is not None else "default"
    elif _tt == 1.0 and _env_f is not None and _env_f != 1.0:
        _tt = _env_f
        _t_src = "env-override-passed-1.0"
    else:
        _t_src = "passed"
    _tt = max(0.0, min(1.0, float(_tt)))
    try:
        _f = fused if isinstance(fused, _torch.Tensor) else _torch.as_tensor(fused)
        _w = warped if isinstance(warped, _torch.Tensor) else _torch.as_tensor(warped)
        _m = mask if isinstance(mask, _torch.Tensor) else _torch.as_tensor(mask)
        _was_3d = (_f.dim() == 3)
        if _was_3d:
            _f = _f.unsqueeze(0); _w = _w.unsqueeze(0)
        if _m.dim() == 2:
            _m = _m.unsqueeze(0).unsqueeze(0)
        elif _m.dim() == 3:
            _m = _m.unsqueeze(1) if _m.shape[0] == _f.shape[0] else _m.unsqueeze(0)
        if _m.shape[-2:] != _f.shape[-2:]:
            _m = _F.interpolate(_m.float(), size=_f.shape[-2:], mode="bilinear", align_corners=False)
        _m = _m.float().clamp(0, 1)
        _mask_mean = float(_m.mean())
        if _mask_mean > 0.5:
            _hole = 1.0 - _m
            _pol = "valid->hole=1-mask"
        else:
            _hole = _m
            _pol = "hole=mask"
        _hole_ratio = float(_hole.mean())
        # V49 对位移: 搜 dx, 让 warped 平移后与 fused 最像
        _best_dx = 0
        _best_err = float("inf")
        _err_dx0 = -1.0
        try:
            # 下采样到 224 加速, 只在有效区比
            _fs = _F.interpolate(_f.float(), size=(224, 224), mode="bilinear", align_corners=False)
            _ws = _F.interpolate(_w.float(), size=(224, 224), mode="bilinear", align_corners=False)
            _hs = _F.interpolate(_hole.float(), size=(224, 224), mode="bilinear", align_corners=False)
            _valid = (1.0 - _hs)
            # 灰度化, 避免颜色差干扰位移搜索
            _fg = _fs.mean(dim=1, keepdim=True)
            _wg = _ws.mean(dim=1, keepdim=True)
            for _dx in range(-24, 25):
                # _dx>0: warped 向右移 _dx, 补边用 reflect
                if _dx == 0:
                    _w_shift = _wg
                elif _dx > 0:
                    _pad = _wg[:, :, :, :1].expand(-1, -1, -1, _dx).clone()  # FIX v124: 平移空出来的外侧复制边缘, 不填黑
                    _w_shift = _torch.cat([_pad, _wg[:, :, :, :-_dx]], dim=3)
                else:
                    _pad = _wg[:, :, :, -1:].expand(-1, -1, -1, -_dx).clone()  # FIX v124
                    _w_shift = _torch.cat([_wg[:, :, :, -_dx:], _pad], dim=3)
                _err = float(((_w_shift - _fg).abs() * _valid).sum() / (_valid.sum() + 1e-6))
                if _dx == 0:
                    _err_dx0 = _err
                if _err < _best_err:
                    _best_err = _err
                    _best_dx = _dx
            # 把 224 尺度的 dx 换回原图尺度
            _scale_x = _f.shape[3] / 224.0
            _dx_full = int(round(_best_dx * _scale_x))
            if _dx_full != 0:
                if _dx_full > 0:
                    _pad = _w[:, :, :, :1].expand(-1, -1, -1, _dx_full).clone()  # FIX v124
                    _w = _torch.cat([_pad, _w[:, :, :, :-_dx_full]], dim=3)
                else:
                    _pad = _w[:, :, :, -1:].expand(-1, -1, -1, -_dx_full).clone()  # FIX v124
                    _w = _torch.cat([_w[:, :, :, -_dx_full:], _pad], dim=3)
                # 洞也要跟着移, 否则洞对不上
                if _dx_full > 0:
                    _pad_h = _hole[:, :, :, :1].expand(-1, -1, -1, _dx_full).clone()  # FIX v124
                    _hole = _torch.cat([_pad_h, _hole[:, :, :, :-_dx_full]], dim=3)
                else:
                    _pad_h = _hole[:, :, :, -1:].expand(-1, -1, -1, -_dx_full).clone()  # FIX v124
                    _hole = _torch.cat([_hole[:, :, :, -_dx_full:], _pad_h], dim=3)
        except Exception as _ae:
            _best_dx = 0
            _dx_full = 0
            _err_dx0 = -1.0
            print("[Fusion] v49 align failed: " + str(_ae), flush=True)
        # 对完位移, 诚实混: w = hole*1 + (1-hole)*t, 不推 w, 不 robust, 不作弊
        # V58 Rowan mask-only: valid 100% Warp, hole 100% fused, 只糊洞边缘

        # V66 safe: V64 包进 try, OOM/失败就 fallback 到 V59 (扩 3px), 能出图
        try:
            _scale_v64 = float(globals().get("_LAST_SCALE_FACTOR", 0.0) or 0.0) or 0.0759
            _img_w_v64 = int(_f.shape[3])  # FIX v73: Fusion 宽 = 原图宽
            _cap_v64 = 2000.0  # FIX v137b: 放开限制, 原 200/120 会把基准截断, HOLE_SCALE=10/20 也扩不上去. 改 2000, 想不限改 float('inf')
            _base_v64 = max(5.0, min(_cap_v64, _img_w_v64 * _scale_v64 * 5.0))  # FIX v108: 扩大修补范围, 系数 0.999 -> 2.0, cap 已同步翻倍
            # FIX v74: 调洞大小. GENSTEREO_HOLE_SCALE=1.0 现在的样子, 0.5 减半, 0.0 不扩张, 负数=腐蚀(洞比原始还小)
            try:
                _hole_scale_v74 = float(_os.environ.get("GENSTEREO_HOLE_SCALE", "1.0") or 1.0)
            except Exception:
                _hole_scale_v74 = 1.0
            _base_raw_v74 = float(_base_v64)
            _base_v64 = float(_base_v64) * abs(_hole_scale_v74) if _hole_scale_v74 != 0 else 0.0
            _diff_v64 = _F.avg_pool2d((_w - _f).abs().mean(dim=1, keepdim=True), kernel_size=9, stride=1, padding=4)
            _gx_v64 = _F.conv2d(_diff_v64, _torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=_diff_v64.dtype, device=_diff_v64.device), padding=1)
            _gy_v64 = _F.conv2d(_diff_v64, _torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], dtype=_diff_v64.dtype, device=_diff_v64.device), padding=1)
            _edge_v64 = ((_gx_v64.abs() + _gy_v64.abs()) / 8.0).clamp(0, 1)
            _need_v64 = _base_v64 * (0.5 + 2.0 * _diff_v64.clamp(0, 1) + 2.0 * _edge_v64)
            _need_x_v64 = _need_v64 * 1.5  # 视差水平, 水平多扩
            _need_y_v64 = _need_v64 * 0.8
            _dist_v64 = _torch.zeros_like(_hole)
            # V65: 内层 try 已删, 外层 try 会兜底, 不再套 try 导致 SyntaxError
            _hole_np_v64 = (1.0 - _hole[0, 0].detach().cpu().numpy() > 0.5).astype('uint8')
            _dist_np_v64 = cv2.distanceTransform(_hole_np_v64, cv2.DIST_L2, 3)
            _dist_v64 = _torch.as_tensor(_dist_np_v64, dtype=_hole.dtype, device=_hole.device).unsqueeze(0).unsqueeze(0)
            # 水平扩为主, 垂直扩为辅: 手的水平错位用水平扩盖, 细手指缝也靠水平扩
            _hole_dil = (((_dist_v64 < _need_x_v64).float() * 0.7 + (_dist_v64 < _need_y_v64).float() * 0.3).clamp(0, 1) * (1.0 - _hole) + _hole).clamp(0, 1)
            # FIX v74: HOLE_SCALE<0 时腐蚀洞 (洞比原始还小), HOLE_DEBUG=1 时打印+保存三联调试图
            if _hole_scale_v74 < 0:
                try:
                    _erode_px_v74 = abs(_hole_scale_v74) * _base_raw_v74
                    _hole_np0_v74 = (_hole[0, 0].detach().cpu().numpy() > 0.5).astype('uint8')
                    _dist_in_v74 = cv2.distanceTransform(_hole_np0_v74, cv2.DIST_L2, 3)
                    _keep_v74 = (_torch.as_tensor(_dist_in_v74, dtype=_hole.dtype, device=_hole.device) > float(_erode_px_v74)).float()
                    _hole = (_hole * _keep_v74.unsqueeze(0).unsqueeze(0)).clamp(0, 1)
                    _hole_dil = _hole.clone()
                except Exception as _e_erode_v74:
                    print('[Hole-Debug] erode failed: ' + str(_e_erode_v74), flush=True)
            if _os.environ.get('GENSTEREO_HOLE_DEBUG', '') == '1':
                try:
                    _hole_before_v74 = float(_hole.mean())
                    _hole_after_v74 = float(_hole_dil.mean())
                    print(f'[Hole-Debug] scale={_hole_scale_v74} base_raw={_base_raw_v74:.1f} base={float(_base_v64):.1f} need_mean={float(_need_v64.mean()):.1f} need_max={float(_need_v64.max()):.1f} hole_before={_hole_before_v74*100:.2f}% hole_after={_hole_after_v74*100:.2f}%', flush=True)
                    globals()['_HOLE_DEBUG_CNT'] = int(globals().get('_HOLE_DEBUG_CNT', 0) or 0) + 1
                    _dbg_dir_v74 = Path('./vis/_hole_debug'); _dbg_dir_v74.mkdir(parents=True, exist_ok=True)
                    _h0_v74 = (_hole[0, 0].detach().cpu().numpy() * 255).astype('uint8')
                    _h1_v74 = (_hole_dil[0, 0].detach().cpu().numpy() * 255).astype('uint8') if _hole_dil.dim() >= 3 else (_hole_dil.detach().cpu().numpy() * 255).astype('uint8')
                    _tri_v74 = cv2.hconcat([_h0_v74, _h1_v74, _h1_v74])
                    cv2.imwrite(str(_dbg_dir_v74 / f"hole_debug_{globals()['_HOLE_DEBUG_CNT']:03d}.png"), _tri_v74)
                except Exception as _e_dbg_v74:
                    print('[Hole-Debug] save failed: ' + str(_e_dbg_v74), flush=True)
        except Exception as _e_v66:
            print("[Fusion] V66 V64 failed, fallback to V59: " + str(_e_v66), flush=True)
            _hole_dil = _F.max_pool2d(_hole, kernel_size=7, stride=1, padding=3)
        _hole_f = _F.avg_pool2d(_hole_dil, kernel_size=1, stride=1, padding=0)  #  羽化洞的边缘 0px，原值为(_hole_dil, kernel_size=3, stride=1, padding=1)  ，这个值大会造成外侧渐变黑边
        _wmap = _hole_f * _tt
        if _wmap.shape[1] == 1 and _f.shape[1] > 1:
            _wmap = _wmap.repeat(1, _f.shape[1], 1, 1)
        _out = _w * (1.0 - _wmap) + _f * _wmap
        if _was_3d:
            _out = _out.squeeze(0)
        return _out
    except Exception as _e:
        print("[Fusion] V65 fallback to fused: " + str(_e), flush=True)
        return fused

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

_TILING_THRESHOLD_SINGLE = 768  # FIX v127: A=896 老路也切块 # 在这里开始模块化
_TILING_THRESHOLD_HEAVY = 1536
_TILING_STEP = 768 if SD_VERSION == "v2.1" else 512
_RESOLUTION_LIST = [512,672,768,896,1024,1088,1120,1216,1280,1344,1472,1568,1600,1664,1728,1792,1920,1984,2016,2048,2240,2304,2464,2560,2688,2816,2912,3072,3136,3328,3360,3456,3584,3808,3840,4032,4096,4224,4256,4352,4480,4608,4704,4864,4928,4992,5120,5152,5376]
def _nearest_resolution(orig):
    if not orig:
        return 0
    return min(_RESOLUTION_LIST, key=lambda x: abs(x - int(orig)))
def _limit_to_source(gui_res, orig_long, limit_on=True):
    try:
        _g = int(gui_res)
        _o = int(orig_long) if orig_long else 0
    except Exception:
        return int(gui_res)
    if not limit_on or _o < 768:
        return _g
    return min(_g, int(_nearest_resolution(_o)))
def _decoupled_low():
    try:
        _v = int(os.environ.get("GENSTEREO_DECOUPLED_LOW", "0") or 0)
    except Exception:
        _v = 0
    if _v > 0:
        return _v
    _mm = str(globals().get("_MODE", "quality") or "quality").strip().lower()
    return int(DECOUPLED_LOW_RES.get(_mm, 1024 if _mm in ("decoupled", "decoupled_b") else (768 if SD_VERSION == "v2.1" else 512)))
_MODE = os.environ.get("GENSTEREO_MODE", "quality").strip().lower()
_ORIG_LONG_CLI = 0
def _smart_step():
    _rounded = _nearest_resolution(_ORIG_LONG) if _ORIG_LONG else 0
    if _rounded and _rounded < 1024:
        return 512
    return 768 if SD_VERSION == "v2.1" else 512
_TILING_SEED_BASE = [0]
def _tiling_plan(image_size):
    s = int(image_size)
    if s <= _TILING_THRESHOLD_SINGLE:
        return ("single", _smart_step(), 0.0)
    if s <= _TILING_THRESHOLD_HEAVY:
        return ("light", _smart_step(), 0.25)
    return ("heavy", _smart_step(), 0.50)
def _step_seed(row, col, pass_id=0):
    import hashlib as _hl
    h = _hl.md5(f"{int(_TILING_SEED_BASE[0])}:{int(row)}:{int(col)}:{int(pass_id)}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2 ** 31)
def _set_step_seed(row, col, pass_id=0):
    s = _step_seed(row, col, pass_id)
    random.seed(s); np.random.seed(s % (2 ** 32)); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s
def _pad_reflect(img_np, patch, stride):
    h, w = img_np.shape[:2]
    import math as _m
    nw = int(_m.ceil((w - patch) / stride)) * stride + patch if w > patch else patch
    nh = int(_m.ceil((h - patch) / stride)) * stride + patch if h > patch else patch
    if (nw, nh) == (w, h):
        return img_np, (w, h)
    pad_r, pad_b = nw - w, nh - h
    out = np.pad(img_np, ((0, pad_b), (0, pad_r), (0, 0)), mode="edge")
    return out, (w, h)
def _depth_aware_seam_mask(left, right, disp_overlap, overlap, axis=1):
    import numpy as _np
    if axis == 0:
        left, right, disp_overlap = _np.transpose(left, (1, 0, 2)), _np.transpose(right, (1, 0, 2)), disp_overlap.T
    diff = _np.abs(left - right).mean(axis=2)
    if disp_overlap.shape != diff.shape:
        _h, _w = diff.shape
        _d = _np.zeros((_h, _w), dtype=_np.float32)
        _ch = min(_h, disp_overlap.shape[0]); _cw = min(_w, disp_overlap.shape[1])
        _d[:_ch, :_cw] = disp_overlap[:_ch, :_cw]
        if _ch < _h:
            _d[_ch:, :] = _d[_ch - 1:_ch, :]
        if _cw < _w:
            _d[:, _cw:] = _d[:, _cw - 1:_cw]
        disp_overlap = _d
    gx = _np.abs(_np.gradient(left.mean(axis=2), axis=1)) + _np.abs(_np.gradient(right.mean(axis=2), axis=1))
    dg = _np.abs(_np.gradient(disp_overlap, axis=1))
    cost = diff + 0.5 * gx + 2.0 * dg
    H, W = cost.shape
    cum = cost.copy(); back = _np.zeros((H, W), dtype=_np.int32)
    for x in range(1, W):
        prev = cum[:, x - 1]
        cand = _np.stack([_np.roll(prev, 1), prev, _np.roll(prev, -1)], axis=1)
        cand[0, 0] = 1e9; cand[-1, 2] = 1e9
        j = _np.argmin(cand, axis=1)
        cum[:, x] += cand[_np.arange(H), j]
        back[:, x] = j - 1
    seam = _np.zeros(H, dtype=_np.int32)
    seam[W - 1] = int(_np.argmin(cum[:, W - 1]))
    for _x in range(W - 1, 0, -1):
        cur_row = int(seam[_x])
        prev_col = cum[:, _x - 1]
        cands = []
        for _d in (-1, 0, 1):
            _r = cur_row + _d
            cands.append(prev_col[_r] if 0 <= _r < H else 1e18)
        best = int(_np.argmin(cands)) - 1
        seam[_x - 1] = int(_np.clip(cur_row + best, 0, H - 1))
    _k = 15
    _pad = _k // 2
    _padded = _np.pad(seam.astype(float), (_pad, _pad), mode='edge')
    _med = _np.array([_np.median(_padded[i:i + _k]) for i in range(H)])
    _smooth = _np.convolve(_med, _np.ones(_k) / _k, mode='same')
    seam = _np.clip(_smooth.astype(_np.int32), 0, W - 1)
    return _np.clip(seam, 0, W - 1)
def _seam_for_pair(left_full, right_patch, disp_full, overlap, axis=1):
    import numpy as _np
    if axis == 0:
        left_full, right_patch = _np.transpose(left_full, (1, 0, 2)), _np.transpose(right_patch, (1, 0, 2))
        disp_full = disp_full.T if disp_full is not None else None
    H, Wl, C = left_full.shape
    ov = int(overlap)
    l_ov, r_ov = left_full[:, Wl - ov:], right_patch[:, :ov]
    d_ov = disp_full[:H, Wl - ov:Wl] if disp_full is not None and disp_full.shape[1] >= Wl else _np.zeros((H, ov), dtype=_np.float32)
    return _depth_aware_seam_mask(l_ov, r_ov, d_ov, ov, axis=1)
def _stitch_pair(left_full, right_patch, disp_full, overlap, mode, axis=1, disp_step=None, seam_override=None):
    import numpy as _np
    if overlap <= 0:
        return _np.concatenate([left_full, right_patch], axis=axis)
    if axis == 0:
        left_full, right_patch = _np.transpose(left_full, (1, 0, 2)), _np.transpose(right_patch, (1, 0, 2))
        disp_full = disp_full.T if disp_full is not None else None
    H, Wl, C = left_full.shape; Wp = right_patch.shape[1]
    ov = int(overlap)
    l_ov, r_ov = left_full[:, Wl - ov:], right_patch[:, :ov]
    _d_raw = disp_full[:H, Wl - ov:Wl] if disp_full is not None and disp_full.shape[1] >= Wl else _np.zeros((H, ov), dtype=_np.float32)
    if _d_raw.shape != (H, ov):
        _tmp = _np.zeros((H, ov), dtype=_np.float32)
        _ch2 = min(H, _d_raw.shape[0]); _cw2 = min(ov, _d_raw.shape[1])
        _tmp[:_ch2, :_cw2] = _d_raw[:_ch2, :_cw2]
        if _ch2 < H:
            _tmp[_ch2:, :] = _tmp[_ch2 - 1:_ch2, :]
        if _cw2 < ov:
            _tmp[:, _cw2:] = _tmp[:, _cw2 - 1:_cw2]
        d_ov = _tmp
    else:
        d_ov = _d_raw
    try:
        _ml = l_ov.mean(axis=(0, 1), keepdims=True); _mr = r_ov.mean(axis=(0, 1), keepdims=True)
        _sl = l_ov.std(axis=(0, 1), keepdims=True) + 1e-6; _sr = r_ov.std(axis=(0, 1), keepdims=True) + 1e-6
        _gain = _np.clip(_sl / _sr, 0.8, 1.25)
        right_patch = _np.clip((right_patch - _mr) * _gain + _ml, 0, 1).astype(_np.float32)
        r_ov = right_patch[:, :ov]
    except Exception:
        pass
    seam = seam_override if seam_override is not None else _depth_aware_seam_mask(l_ov, r_ov, d_ov, ov, axis=1)
    ys, xs = _np.mgrid[0:H, 0:ov]
    _xs = _np.arange(ov, dtype=_np.float32)[None, :, None]
    _t = (_xs + 0.5) / float(max(1, ov))
    _w = 0.5 - 0.5 * _np.cos(_np.pi * _t)
    w = l_ov * (1 - _w) + r_ov * _w
    out = _np.concatenate([left_full[:, :Wl - ov], w, right_patch[:, ov:]], axis=1)
    if axis == 0:
        out = _np.transpose(out, (1, 0, 2))
    return out
def _multiband_blend(a, b, dist, levels=4):
    import numpy as _np
    w = (dist > 0).astype(_np.float32)
    def _lap(img):
        pyr, cur = [], img
        for _ in range(levels):
            down = cv2.pyrDown(cur)
            up = cv2.resize(cv2.pyrUp(down), (cur.shape[1], cur.shape[0]))
            pyr.append(cur - up); cur = down
        pyr.append(cur)
        return pyr
    la, lb = _lap(a), _lap(b)
    out_levels = []
    for i, (pa, pb) in enumerate(zip(la, lb)):
        h, wd = pa.shape[:2]
        wi = cv2.resize(w, (wd, h), interpolation=cv2.INTER_LINEAR)
        if i < levels - 1:
            wi = _np.clip((cv2.resize(dist, (wd, h)) + 2) / 4.0, 0, 1)
        else:
            wi = cv2.GaussianBlur(wi, (0, 0), sigmaX=max(16, wd // 8))
        out_levels.append(pa * (1 - wi[..., None]) + pb * wi[..., None])
    out = out_levels[-1]
    for i in range(levels - 1, -1, -1):
        out = cv2.resize(cv2.pyrUp(out), (out_levels[i].shape[1], out_levels[i].shape[0])) + out_levels[i]
    return _np.clip(out, 0, 1).astype(_np.float32)
def generate_novel_view_tiled(image, depth, output_dir, basename, scale_factor=0.15, output_name="generated_right.png", pass_id=0):
    import numpy as _np
    import torch as _t
    mode, patch, overlap_ratio = _tiling_plan(IMAGE_SIZE)
    if mode == "single":
        return generate_novel_view(image, depth, output_dir, basename, scale_factor, output_name)
    print(f"[Easy3D-Step] Detail {IMAGE_SIZE}", flush=True)  # FIX v100
    _cfg_h, _cfg_w = None, None
    try:
        _cfg_h = getattr(genstereo_nvs.cfg, "height", None); _cfg_w = getattr(genstereo_nvs.cfg, "width", None)
        genstereo_nvs.cfg.height = int(patch); genstereo_nvs.cfg.width = int(patch)
    except Exception as _e:
        print(f"[Easy3D-Step] Could not set cfg height/width: {_e}", flush=True)
    img_np = _np.array(image.convert("RGB")).astype(_np.float32) / 255.0
    disp = (normalize_disp(depth) * float(scale_factor) * IMAGE_SIZE)
    disp_np = disp.squeeze().cpu().numpy().astype(_np.float32)
    H0, W0 = img_np.shape[:2]
    stride = max(1, int(patch * (1 - overlap_ratio)))
    def _starts(size, patch, stride):
        if size <= patch:
            return [0]
        n = (size - patch + stride - 1) // stride + 1
        if n <= 1:
            return [0]
        return [round(i * (size - patch) / float(n - 1)) for i in range(n)]
    starts_x = _starts(W0, patch, stride)
    starts_y = _starts(H0, patch, stride)
    cols = len(starts_x); rows = len(starts_y)
    _step_x = (starts_x[-1] - starts_x[0]) / float(max(1, cols - 1)) if cols > 1 else float(patch)
    overlap = patch - int(round(_step_x))
    img_pad = img_np
    disp_pad = disp_np
    Hp, Wp = img_np.shape[:2]
    ow, oh = W0, H0
    _total_steps = rows * cols
    print(f"[Easy3D-Step] Total {_total_steps} steps", flush=True)  # FIX v100
    import time as _time
    _t0 = _time.time()
    warped_grid, synth_grid, mask_grid = [], [], []
    for r in range(rows):
        w_row, s_row, m_row = [], [], []
        for c in range(cols):
            y0, x0 = starts_y[r], starts_x[c]
            patch_img = Image.fromarray((img_pad[y0:y0 + patch, x0:x0 + patch] * 255).astype(_np.uint8))
            patch_disp = _t.tensor(disp_pad[y0:y0 + patch, x0:x0 + patch]).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            _step_idx = r * cols + c + 1
            print(f"[Easy3D-Step] Step {_step_idx}/{_total_steps} ... elapsed {_time.time() - _t0:.1f}s", flush=True)  # FIX v100
            _set_step_seed(r, c, pass_id)
            with _t.no_grad():
                renders = genstereo_nvs(src_image=patch_img, src_disparity=patch_disp, ratio=None)
            w_row.append(((renders["warped"] + 1) / 2)[0].cpu().numpy().transpose(1, 2, 0).astype(_np.float32))
            s_row.append(renders["synthesized"][0].cpu().numpy().transpose(1, 2, 0).astype(_np.float32))
            _m = morphological_opening(renders["mask"])
            m_row.append(_m[0, 0].cpu().numpy().astype(_np.float32)[..., None].repeat(3, axis=2))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        warped_grid.append(w_row); synth_grid.append(s_row); mask_grid.append(m_row)
    def _stitch_grids_shared_seam(w_grid, s_grid, m_grid):
        def _stitch_row(w_row_list, s_row_list, m_row_list, r_idx):
            disp_row = disp_pad[starts_y[r_idx]: starts_y[r_idx] + patch, :]
            cw, cs, cm = w_row_list[0], s_row_list[0], m_row_list[0]
            for c in range(1, cols):
                seam = _seam_for_pair(cw, w_row_list[c], disp_row, overlap, axis=1)
                cw = _stitch_pair(cw, w_row_list[c], disp_row, overlap, mode, axis=1, seam_override=seam)
                cs = _stitch_pair(cs, s_row_list[c], disp_row, overlap, mode, axis=1, seam_override=seam)
                cm = _stitch_pair(cm, m_row_list[c], disp_row, overlap, mode, axis=1, seam_override=seam)
            return cw, cs, cm
        stitched = [_stitch_row(w_grid[r], s_grid[r], m_grid[r], r) for r in range(rows)]
        ow_, os_, om_ = stitched[0]
        for r in range(1, rows):
            disp_vert = disp_pad[:ow_.shape[0], :]
            seam = _seam_for_pair(ow_, stitched[r][0], disp_vert, overlap, axis=0)
            ow_ = _stitch_pair(ow_, stitched[r][0], disp_vert, overlap, mode, axis=0, seam_override=seam)
            os_ = _stitch_pair(os_, stitched[r][1], disp_vert, overlap, mode, axis=0, seam_override=seam)
            om_ = _stitch_pair(om_, stitched[r][2], disp_vert, overlap, mode, axis=0, seam_override=seam)
        return ow_[:oh, :ow], os_[:oh, :ow], om_[:oh, :ow, 0]
    warped_full, synth_full, mask_full = _stitch_grids_shared_seam(warped_grid, synth_grid, mask_grid)
    warped_full = _fix_warped_outer_black(warped_full, mask_full)  # FIX v136: 同 v124, Fusion 前只填 warped 外圈洞
    # FIX v136: synth 不动 (v131 整圈换 synth 是错的)
    try:
        if _cfg_h is not None: genstereo_nvs.cfg.height = int(_cfg_h)
        if _cfg_w is not None: genstereo_nvs.cfg.width = int(_cfg_w)
    except Exception:
        pass
    print(f"[Easy3D-Step] All steps done, elapsed {_time.time() - _t0:.1f}s, start Fusion", flush=True)
    import torch as _t2
    with _t2.no_grad():
        mask_t = _t2.tensor(mask_full).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        synth_t = _t.tensor(synth_full.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        warped_t = _t.tensor(warped_full.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        fusion = fusion_model(synth_t, warped_t, mask_t)
        fusion = _apply_fusion_strength(fusion, warped_t, mask_t)
    fusion_np = _np.clip(fusion[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
    # FIX v136: 成品 fusion_np 不动 (v131 整圈换成品是错的)
    fusion_np = _sharpen_new_view(fusion_np)  # FIX v130: 新图, 合 SBS 前锐化
    output_path = join(output_dir, basename); os.makedirs(output_path, exist_ok=True)
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, "disp.png"), disp_vis)
    image.save(join(output_path, "left.png"))
    to_pil_image(_t2.tensor(warped_full.transpose(2, 0, 1))).save(join(output_path, "warped.png"))
    to_pil_image(_t2.tensor(fusion_np.transpose(2, 0, 1))).save(join(output_path, output_name))
    print(f"[Easy3D-Step] Saved {output_name}")  # FIX v100
def _warp_by_disparity(img_np, disp_np):
    import numpy as _np
    _h, _w = disp_np.shape[:2]
    _xs, _ys = _np.meshgrid(_np.arange(_w, dtype=_np.float32), _np.arange(_h, dtype=_np.float32))
    _map_x = (_xs + disp_np).astype(_np.float32)
    _map_y = _ys.astype(_np.float32)
    _warped = cv2.remap((img_np * 255.0).astype(_np.uint8), _map_x, _map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (_warped.astype(_np.float32) / 255.0)

def _load_orig_image_for_warp():
    """FIX decoupled_warp_orig_v2: 读原图 (--orig_img / GENSTEREO_ORIG_IMG). 返回 PIL 或 None."""
    try:
        import os as _os
        from PIL import Image as _PIL
        _p = str(globals().get("_ORIG_IMG_CLI", "") or "")
        if not _p:
            _p = str(_os.environ.get("GENSTEREO_ORIG_IMG", "") or "")
        if not _p or not Path(_p).exists():
            return None
        return _PIL.open(_p).convert("RGB")
    except Exception:
        return None


def _warp_on_original(orig_pil, depth, scale_factor):
    """FIX decoupled_warp_orig_v2: 深度图 bilinear 放大到原图 + 对原图 Warp. 深度放大非常快."""
    import numpy as _np
    import torch as _t
    _ow, _oh = orig_pil.size
    _disp_norm = normalize_disp(depth)
    _disp_orig_t = _t.nn.functional.interpolate(_disp_norm, size=(_oh, _ow), mode="bilinear", align_corners=False)
    _disp_orig = (_disp_orig_t * float(scale_factor) * float(_ow)).squeeze().cpu().numpy().astype(_np.float32)
    _img_np = _np.array(orig_pil).astype(_np.float32) / 255.0
    _warped = _warp_by_disparity(_img_np, _disp_orig)
    return _warped, _disp_orig, _img_np



def _CURRENT_ORIG():
    """FIX decoupled_warp_orig_v3: __main__ 循环每张图开始前设置的当前原图 PIL, 没有则 None."""
    try:
        return globals().get("_CURRENT_ORIG_PIL", None)
    except Exception:
        return None


def _resolve_orig_for_prep(prep_path):
    """FIX decoupled_warp_orig_v3: 按 prep 文件名找原图. prep = <stem>_prep<res>.jpg, 原图 = <orig_dir>/<stem>.*"""
    try:
        import os as _os
        from PIL import Image as _PIL
        _stem = Path(str(prep_path)).stem
        # 去掉 _prep<res> 后缀
        if "_prep" in _stem:
            _stem = _stem.split("_prep")[0]
        # 1) orig_dir 里找同 stem
        _odir = str(globals().get("_ORIG_DIR_CLI", "") or _os.environ.get("GENSTEREO_ORIG_DIR", "") or "")
        if _odir and Path(_odir).is_dir():
            for _ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"):
                _c = Path(_odir) / (_stem + _ext)
                if _c.exists():
                    return _PIL.open(_c).convert("RGB")
            # 大小写 / stem 带点: 遍历一次
            for _c in Path(_odir).iterdir():
                if _c.stem == _stem and _c.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"):
                    return _PIL.open(_c).convert("RGB")
        # 2) 单张 --orig_img 兜底 (单图批才准)
        _one = str(globals().get("_ORIG_IMG_CLI", "") or "")
        if _one and Path(_one).exists():
            return _PIL.open(_one).convert("RGB")
        # 3) _CURRENT_ORIG ( __main__ 每张图设置, 最准 )
        _cur = _CURRENT_ORIG()
        if _cur is not None:
            return _cur
    except Exception:
        pass
    return None


def _load_orig_image_for_warp(prep_path=None):
    """FIX decoupled_warp_orig_v3: 优先 _CURRENT_ORIG, 再按 prep 找, 再单张兜底."""
    try:
        _cur = _CURRENT_ORIG()
        if _cur is not None:
            return _cur
        if prep_path is not None:
            _r = _resolve_orig_for_prep(prep_path)
            if _r is not None:
                return _r
        import os as _os
        from PIL import Image as _PIL
        _p = str(globals().get("_ORIG_IMG_CLI", "") or _os.environ.get("GENSTEREO_ORIG_IMG", "") or "")
        if _p and Path(_p).exists():
            return _PIL.open(_p).convert("RGB")
    except Exception:
        pass
    return None


def _warp_on_original(orig_pil, depth, scale_factor):
    """FIX decoupled_warp_orig_v3: 深度图 bilinear 放大到原图 + 对原图 Warp."""
    import numpy as _np
    import torch as _t
    _ow, _oh = orig_pil.size
    _disp_norm = normalize_disp(depth)
    _disp_orig_t = _t.nn.functional.interpolate(_disp_norm, size=(_oh, _ow), mode="bilinear", align_corners=False)
    _disp_orig = (_disp_orig_t * float(scale_factor) * float(_ow)).squeeze().cpu().numpy().astype(_np.float32)
    _img_np = _np.array(orig_pil).astype(_np.float32) / 255.0
    _warped = _warp_by_disparity(_img_np, _disp_orig)
    return _warped, _disp_orig, _img_np


def generate_novel_view_decoupled(image, depth, output_dir, basename, scale_factor=0.15, output_name="generated_right.png", pass_id=0):
    import torch as _torch
    import numpy as _np
    import torch as _t
    import time as _time
    _t0 = _time.time()
    _low = int(_decoupled_low())
    print(f"[Easy3D-Decoupled] Warp full {IMAGE_SIZE}, diffusion low {_low}, 1 pass", flush=True)
    img_np = _np.array(image.convert("RGB")).astype(_np.float32) / 255.0
    try:
        _dh, _dw = int(depth.shape[-2]), int(depth.shape[-1])
        if (_dh, _dw) != (int(IMAGE_SIZE), int(IMAGE_SIZE)):
            depth = _torch.nn.functional.interpolate(depth.float(), size=(int(IMAGE_SIZE), int(IMAGE_SIZE)), mode="bilinear", align_corners=False)
            print(f"[Easy3D-Decoupled] Depth aligned {_dw} -> {IMAGE_SIZE} (FlashDepth rounds to 14x)", flush=True)
    except Exception as _e:
        print(f"[Easy3D-Decoupled] Depth align skipped: {_e}", flush=True)
    # FIX decoupled_warp_orig_v2: 优先对原图 Warp. 有原图 -> 深度放大到原图 + Warp 原图; 没有 -> 回退 Res 图.
    _orig_pil = _load_orig_image_for_warp()
    _warped_orig = None; _img_np_orig = None; _orig_wh = None
    if _orig_pil is not None:
        try:
            _warped_orig, _disp_orig, _img_np_orig = _warp_on_original(_orig_pil, depth, scale_factor)
            _orig_wh = (_orig_pil.size[0], _orig_pil.size[1])  # (W, H)
            print(f"[Easy3D-Decoupled] Warp on ORIGINAL {_orig_wh[0]}x{_orig_wh[1]} (depth upscaled from {IMAGE_SIZE}, fast)", flush=True)
        except Exception as _e:
            print(f"[Easy3D-Decoupled] Warp on original failed ({_e}), fallback to Res", flush=True)
            _warped_orig = None; _img_np_orig = None; _orig_wh = None
    disp_full = (normalize_disp(depth) * float(scale_factor) * IMAGE_SIZE).squeeze().cpu().numpy().astype(_np.float32)
    warped_full = _warped_orig if _warped_orig is not None else _warp_by_disparity(img_np, disp_full)  # FIX decoupled_warp_orig_v2
    # FIX v73 Bug 1: 原图模式下 warped 已是原图尺寸, 不要再拉回 1344 图.
    if _orig_wh is None and warped_full.shape[:2] != (int(IMAGE_SIZE), int(IMAGE_SIZE)):
        warped_full = cv2.resize(warped_full, (int(IMAGE_SIZE), int(IMAGE_SIZE)), interpolation=cv2.INTER_LINEAR).astype(_np.float32)
        print(f"[Easy3D-Decoupled] Warped aligned -> {IMAGE_SIZE}", flush=True)
    low_img = image.resize((_low, _low), Image.BILINEAR)
    disp_low_t = normalize_disp(depth) * float(scale_factor) * _low
    disp_low_np = _torch.nn.functional.interpolate(disp_low_t, size=(_low, _low), mode="bilinear", align_corners=False).squeeze().cpu().numpy().astype(_np.float32)
    patch_disp = _t.tensor(disp_low_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    _cfg_h, _cfg_w = None, None
    try:
        _cfg_h = getattr(genstereo_nvs.cfg, "height", None); _cfg_w = getattr(genstereo_nvs.cfg, "width", None)
        genstereo_nvs.cfg.height = int(_low); genstereo_nvs.cfg.width = int(_low)
    except Exception:
        pass
    try:
        random.seed(int(_TILING_SEED_BASE[0]) + int(pass_id)); _t.manual_seed(int(_TILING_SEED_BASE[0]) + int(pass_id))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(_TILING_SEED_BASE[0]) + int(pass_id))
        with _t.no_grad():
            renders = genstereo_nvs(src_image=low_img, src_disparity=patch_disp, ratio=None)
    finally:
        try:
            if _cfg_h is not None: genstereo_nvs.cfg.height = int(_cfg_h)
            if _cfg_w is not None: genstereo_nvs.cfg.width = int(_cfg_w)
        except Exception:
            pass
    synth_low = renders["synthesized"][0].cpu().numpy().transpose(1, 2, 0).astype(_np.float32)
    mask_low = morphological_opening(renders["mask"])[0, 0].cpu().numpy().astype(_np.float32)
    _fuse_wh = _orig_wh if _orig_wh is not None else (int(IMAGE_SIZE), int(IMAGE_SIZE))  # FIX decoupled_warp_orig_v2: Fusion 尺寸跟 Warp 走
    synth_full = cv2.resize(synth_low, _fuse_wh, interpolation=cv2.INTER_LINEAR).astype(_np.float32)
    mask_full = cv2.resize(mask_low, _fuse_wh, interpolation=cv2.INTER_LINEAR).astype(_np.float32)  # FIX decoupled_warp_orig_v2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Easy3D-Decoupled] Diffusion done, elapsed {_time.time() - _t0:.1f}s, start Fusion", flush=True)
    # FIX decoupled_warp_orig_v2: 原图模式下三张都是原图尺寸, 不再 assert Res 图
    if _orig_wh is None:
        assert synth_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"synth {synth_full.shape}"
    if _orig_wh is None:
        assert warped_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"warped {warped_full.shape}"  # FIX decoupled_warp_orig_v2
    if _orig_wh is None:
        assert mask_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"mask {mask_full.shape}"  # FIX decoupled_warp_orig_v2
    globals()["_LAST_SCALE_FACTOR"] = float(scale_factor)
    warped_full = _v53_correct_disparity_np(warped_full, synth_full, _img_np_orig if _img_np_orig is not None else img_np)  # FIX decoupled_warp_orig_v2
    with _t.no_grad():
        mask_t = _t.tensor(mask_full).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        fusion = fusion_model(_t.tensor(synth_full.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE), _t.tensor(warped_full.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE), mask_t)
        try:
            _dbg_max_wf = float(__import__('numpy').abs(warped_full - synth_full).max()) if 'warped_full' in locals() and 'synth_full' in locals() else -1.0
            _dbg_max_wi = float(__import__('numpy').abs(warped_full - img_np).max()) if 'warped_full' in locals() and 'img_np' in locals() else -1.0
            print('[Fusion-Debug-V45] max|warped-fused|=' + str(round(_dbg_max_wf, 4)) + ' max|warped-img|=' + str(round(_dbg_max_wi, 4)) + ' (max|warped-img|~0 means warped is orig)', flush=True)
        except Exception as _dbg_e:
            print('[Fusion-Debug-V45] failed: ' + str(_dbg_e), flush=True)
    fusion = _apply_fusion_strength(fusion, _t.tensor(warped_full.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE), mask_t)
    fusion_np = _np.clip(fusion[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
    fusion_np = _sharpen_new_view(fusion_np)  # FIX v130: 新图, 合 SBS 前锐化
    output_path = join(output_dir, basename); os.makedirs(output_path, exist_ok=True)
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, "disp.png"), disp_vis)
    image.save(join(output_path, "left.png"))
    to_pil_image(_t.tensor(warped_full.transpose(2, 0, 1))).save(join(output_path, "warped.png"))
    to_pil_image(_t.tensor(fusion_np.transpose(2, 0, 1))).save(join(output_path, output_name))
    print(f"[Easy3D-Decoupled] saved {output_name} (Warp {IMAGE_SIZE} + diffusion {_low})")

# ---- FIX v109: GNVD 路由 (decoupled 高分辨率 >960 才切块, 其余老路) ----
def _call_gnvd(img, depth, output_dir, basename, scale_factor, output_name, pass_id=0):
    """试 GNVD, 成功返回 True (已保存), 失败返回 False (上层回退老路). 兼容: 缺依赖就回退, 不炸."""
    try:
        from generate_novel_view_decoupled import generate_novel_view_decoupled_highres as _gnvd
    except Exception as _e:
        print(f"[Easy3D-GNVD] import 失败, 回退老路: {_e}", flush=True)
        return False
    try:
        def _save_gnvd(image=None, depth=None, warped_full=None, fusion_np=None,
                       output_dir=None, basename=None, output_name=None, normalize_disp=None, **_kw):
            import torch as _t_save
            _op = join(output_dir, basename); os.makedirs(_op, exist_ok=True)
            try:
                _dv = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imwrite(join(_op, "disp.png"), _dv)
            except Exception:
                pass
            try:
                image.save(join(_op, "left.png"))
            except Exception:
                pass
            try:
                to_pil_image(_t_save.tensor(warped_full.transpose(2, 0, 1))).save(join(_op, "warped.png"))
            except Exception:
                pass
            fusion_np = _sharpen_new_view(fusion_np)  # FIX v130: GNVD 新图, 合 SBS 前锐化
            to_pil_image(_t_save.tensor(fusion_np.transpose(2, 0, 1))).save(join(_op, output_name))
        _ctx = dict(
            low=int(_decoupled_low()),
            image_size=int(IMAGE_SIZE),
            genstereo_nvs=genstereo_nvs,
            fusion_model=fusion_model,
            normalize_disp=normalize_disp,
            morphological_opening=morphological_opening,
            apply_fusion_strength=_apply_fusion_strength,
            warp_on_original=_warp_on_original,
            load_orig_image_for_warp=_load_orig_image_for_warp,
            v53_correct=_v53_correct_disparity_np,
            warp_by_disparity=_warp_by_disparity,
            seed_base=int(_TILING_SEED_BASE[0]) if isinstance(_TILING_SEED_BASE, (list, tuple)) else int(_TILING_SEED_BASE),
            device=str(DEVICE),
            save_fn=_save_gnvd,
            orig_pil=globals().get("_CURRENT_ORIG_PIL"),
        )
        _gnvd(img, depth, output_dir, basename, scale_factor, output_name, pass_id=pass_id, ctx=_ctx)
        return True
    except Exception as _e:
        print(f"[Easy3D-GNVD] GNVD 失败, 回退老路: {_e}", flush=True)
        return False

def _generate_for_image(img, depth, output_dir, basename, scale_factor, output_name, pass_id=0):
    if str(_MODE).startswith("decoupled"):
        # FIX v109: decoupled 且扩散分辨率 >960 才走 GNVD 切块, <=960 走老路, GNVD 失败回退老路
        try:
            _low_v109 = int(_decoupled_low())
        except Exception:
            _low_v109 = 0
        if _low_v109 >= 896:  # FIX v127: A=896 也走 GNVD 2x2
            if _call_gnvd(img, depth, output_dir, basename, scale_factor, output_name, pass_id=pass_id):
                return None
            print("[Easy3D-GNVD] 回退老路 generate_novel_view_decoupled()", flush=True)
        return generate_novel_view_decoupled(img, depth, output_dir, basename, scale_factor, output_name, pass_id=pass_id)
    mode, _, _ = _tiling_plan(IMAGE_SIZE)
    if mode == "single":
        return generate_novel_view(img, depth, output_dir, basename, scale_factor, output_name)
    return generate_novel_view_tiled(img, depth, output_dir, basename, scale_factor, output_name, pass_id=pass_id)
def generate_novel_view(image, depth, output_dir, basename, scale_factor=0.15, output_name='generated_right.png'):
    import torch as _torch
    import torch as _t
    disparity = normalize_disp(depth) * scale_factor * IMAGE_SIZE
    output_path = join(output_dir, basename)
    os.makedirs(output_path, exist_ok=True)
    renders = genstereo_nvs(src_image=image, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    with torch.no_grad():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
        fusion_image = _apply_fusion_strength(fusion_image, warped, mask)
    # FIX v131: Quality Single, Fusion 后修成品 + warped 外圈
    try:
        import numpy as _np_v131
        _f_np_v131 = _np_v131.clip(fusion_image[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
        fusion_image_pil = to_pil_image(_t.tensor(_f_np_v131.transpose(2, 0, 1)))  # FIX v136: 成品不动
        _w_np_v131 = _np_v131.clip(warped[0].cpu().numpy().transpose(1, 2, 0), 0, 1)
        _m_np_v131 = mask[0, 0].cpu().numpy().astype('float32')  # FIX v136: 只填 warped 外圈洞, 用 mask 判断洞
        warped_pil = to_pil_image(_t.tensor(_fix_warped_outer_black(_w_np_v131, _m_np_v131).transpose(2, 0, 1)))
    except Exception as _e_v131:
        print(f"[Easy3D-OuterEdge] v131 Single 失败, 用原图: {_e_v131}", flush=True)
        warped_pil = to_pil_image(warped[0])
        fusion_image_pil = to_pil_image(fusion_image[0])
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, 'disp.png'), disp_vis)
    image.save(join(output_path, 'left.png'))
    warped_pil.save(join(output_path, 'warped.png'))
    fusion_image_pil.save(join(output_path, output_name))
print("[Easy3D] module loaded (V69)", flush=True)
if __name__ == '__main__':
    print("[Easy3D] __main__ start (V67)", flush=True)
    parser = argparse.ArgumentParser(description="Generate novel view from input image")
    parser.add_argument("image_path", nargs="+", help="v3.2: MULTIPLE files = load model ONCE, loop all (lossless). GUI passes all _tmp_square files in one call.")
    parser.add_argument("--output", default="./vis", help="Output directory (vis/<stem>/{left,generated_right,warped,disp}.png)")
    parser.add_argument("--scale_factor", type=float, default=0.15, help="Disparity scaling. GUI: 0.15 * baseline * depth_strength")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (GUI Steps)")
    parser.add_argument("--fusion_strength", type=float, default=1.0, help="FIX v28: 1.0=old AdaptiveFusion, 0.0=pure Warp")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale (GUI Guidance)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed, -1 = random")
    parser.add_argument("--gen_mode", default="orig_right", choices=["orig_right", "orig_left", "both"], help="v3.3")
    parser.add_argument("--image_size", type=int, default=0, help="STRETCH target square side.")
    parser.add_argument("--mode", default="quality", choices=["quality", "decoupled", "decoupled_a", "decoupled_b", "decoupled_c", "decoupled_d", "decoupled_e", "decoupled_f"], help="FIX v22")
    parser.add_argument("--decoupled_low", type=int, default=0, help="FIX v22")
    parser.add_argument("--orig_long", type=int, default=0, help="FIX v22")
    parser.add_argument("--orig_img", type=str, action="append", default=[], help="FIX decoupled_warp_orig_v3: 原图路径, 每张各传一次")
    parser.add_argument("--orig_dir", type=str, default="", help="FIX decoupled_warp_orig_v3: 输入文件夹, 每张按 stem 找原图")
    parser.add_argument("--limit_to_source", type=int, default=1, help="FIX v22")
    args = parser.parse_args()
    _seed = args.seed if args.seed is not None and args.seed >= 0 else random.randint(0, 2147483647)
    _TILING_SEED_BASE[0] = int(_seed)
    print(f"[Easy3D] seed={_seed} steps={args.steps} guidance={args.guidance} scale_factor={args.scale_factor} image_size={args.image_size or IMAGE_SIZE}")
    print(f"[Easy3D] MDE={_mde_model_env} path={_mde_path_env}")
    random.seed(_seed); np.random.seed(_seed % (2**32)); torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    try:
        genstereo_nvs.cfg.num_inference_steps = int(args.steps)
        genstereo_nvs.cfg.guidance_scale = float(args.guidance)
        if hasattr(genstereo_nvs, 'scheduler'):
            genstereo_nvs.scheduler.set_timesteps(int(args.steps), device=genstereo_nvs.device)
    except Exception as _e:
        print(f"[Easy3D] Could not apply steps/guidance to scheduler (will use defaults): {_e}")
    try:
        _fs = float(getattr(args, "fusion_strength", 1.0) if getattr(args, "fusion_strength", None) is not None else float(__import__("os").environ.get("GENSTEREO_FUSION_STRENGTH", "1.0") or 1.0))
        globals()["FUSION_STRENGTH"] = _fs
        print("[Easy3D] FusionStrength=%.2f (1.0=old AdaptiveFusion, 0.0=pure Warp)" % _fs)
    except Exception as _e:
        print(f"[Easy3D] FusionStrength not applied: {_e}")
    if args.image_size and args.image_size > 0:
        IMAGE_SIZE = int(args.image_size)
    try:
        globals()["_ORIG_IMG_CLI"] = str(getattr(args, "orig_img", "") or "")
        globals()["_ORIG_IMG_CLI"] = str((getattr(args, "orig_img", []) or [""])[0]) if getattr(args, "orig_img", None) else ""
        globals()["_ORIG_DIR_CLI"] = str(getattr(args, "orig_dir", "") or "")
        globals()["_ORIG_IMGS_CLI"] = list(getattr(args, "orig_img", []) or [])
        globals()["_MODE"] = str(getattr(args, "mode", "quality") or "quality").strip().lower()
        # FIX v104: Decoupled 时 IMAGE_SIZE 由 test.py 的 DECOUPLED_LOW_RES 定, GUI Resolution 顶不掉.
        if str(globals().get("_MODE", "quality") or "quality").strip().lower().startswith("decoupled"):
            _dec_low = int(_decoupled_low())
            if _dec_low > 0:
                IMAGE_SIZE = int(_dec_low)
                print(f"[Easy3D-Decoupled] Mode={globals().get('_MODE')} IMAGE_SIZE forced to {IMAGE_SIZE} (test.py DECOUPLED_LOW_RES, GUI Resolution ignored)", flush=True)
        if getattr(args, "decoupled_low", 0):
            os.environ["GENSTEREO_DECOUPLED_LOW"] = str(int(args.decoupled_low))
        _orig_cli = int(getattr(args, "orig_long", 0) or 0) or int(os.environ.get("GENSTEREO_ORIG_LONG", "0") or 0)
        globals()["_ORIG_LONG_CLI"] = _orig_cli
        _limit_on = int(getattr(args, "limit_to_source", 1) or 0) != 0
        if _orig_cli >= 768 and _limit_on and IMAGE_SIZE > _nearest_resolution(_orig_cli):
            _eff = _limit_to_source(IMAGE_SIZE, _orig_cli, True)
            print(f"[Easy3D-Limit] Orig {_orig_cli}px, Res {IMAGE_SIZE} -> {_eff} (Limit to Source, orig>=768)")
            IMAGE_SIZE = int(_eff)
        elif _orig_cli and _orig_cli < 768:
            print(f"[Easy3D-Limit] Orig {_orig_cli}px <768, no cap, Res {IMAGE_SIZE} (small orig, do not shrink)")
        print(f"[Easy3D] Mode={_MODE} LimitToSource={'ON' if _limit_on else 'OFF'}")
        # FIX v96: Fast-Decoupled 时 GUI Resolution 被忽略, 强制用 Decoupled 分辨率 (A=960 B=1280 C=1536).
        # --image_size 和 Limit to Source 都顶不掉. 这样 Warp 全图 = 扩散低分辨率, 不用缩放, 最清晰.
        _forced = DECOUPLED_LOW_RES.get(str(globals().get('_MODE', 'quality')).strip().lower())
        if _forced:
            print(f"[Easy3D-Decoupled] Mode={globals().get('_MODE')} GUI Resolution ignored, forced IMAGE_SIZE={_forced} (Warp = diffusion low)")
            IMAGE_SIZE = int(_forced)
    except Exception as _e:
        print(f"[Easy3D] Limit/Mode not applied: {_e}")
    try:
        genstereo_nvs.cfg.height = int(IMAGE_SIZE); genstereo_nvs.cfg.width = int(IMAGE_SIZE)
        print(f"[Easy3D] GenStereo running at {IMAGE_SIZE} (SD3D-style)")
    except Exception as _e:
        print(f"[Easy3D] Could not override GenStereo height/width: {_e}")
    _paths = args.image_path if isinstance(args.image_path, list) else [args.image_path]
    print(f"[Easy3D] Batch: {len(_paths)} image(s), model loaded ONCE, gen_mode={args.gen_mode}")
    for _i, _ip in enumerate(_paths, 1):
        base_name = splitext(basename(_ip))[0]
        print(f"[Easy3D] [{_i}/{len(_paths)}] Processing {_ip} ... gen_mode={args.gen_mode} scale={args.scale_factor}")
        globals()["_CURRENT_ORIG_PIL"] = _resolve_orig_for_prep(_ip)  # FIX decoupled_warp_orig_v3: 这张的原图
        img, depth = infer_depth(_ip, IMAGE_SIZE)
        if args.gen_mode == "both":
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * 0.5, 'generated_right.png', pass_id=0)
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * -0.5, 'generated_left.png', pass_id=1)
        elif args.gen_mode == "orig_left":
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * -1.0, 'generated_left.png', pass_id=1)
        else:
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor, 'generated_right.png', pass_id=0)
