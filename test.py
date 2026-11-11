# FIX v71: inpaint 覆盖 +1% (base 0.55->0.5666)

# FIX v64: 宽图手 - 前景局部 dx(梯度) + 水平多扩 + 宽图 cap 100

# FIX v36: 均匀扩 -> 每个像素自适应扩 (base*0.55, 按 local |warped-fused| 在 0.5x..2.5x 浮动) -L 362

# FIX v34: 拿掉 clamp + 羽化只 3px + 智能扩 (IMAGE_SIZE*scale*0.35, 5..40)
# FIX v33: Tony hole 先扩 3px 再羽化 + V53 charmap 兜底

# FIX v29: warped==orig 时不混原图, diff 大处全用 fused
# FIX v28: decoupled 里 _apply 第二个参数必须是 warped, 不是原图左眼
# FIX v27: t 从 GENSTEREO_FUSION_STRENGTH 读 + mask 极性 + 羽化
# FIX v26: Fusion 权重图高斯羽化, <1 不再硬过渡

# FIX v25: Fast - Decoupled-L 250
from os.path import basename, splitext, join
import numpy as np
import torch
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
import ssl
import os
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
# Map GUI lowercase -> flashdepth_model.py model_type ("FlashDepth-L" / "FlashDepth-S")
FLASHDEPTH_TYPE = {"flashdepth-l": "FlashDepth-L", "flashdepth-s": "FlashDepth-S", "flashdepth": "FlashDepth"}.get(_mde_model_env, "FlashDepth-L")  # FIX v18: 兜底 FlashDepth-S -> FlashDepth-L
# checkpoint_dir = folder that holds sd-vae-ft-mse / image_encoder / genstereo-v1.5 = models/
checkpoint_dir = os.path.dirname(_mde_path_env) or "models"
# For FlashDepth, checkpoint_dir for GenStereo must stay models/ (not models/FlashDepth/...)
if IS_FLASHDEPTH:
    checkpoint_dir = "models"
os.makedirs(checkpoint_dir, exist_ok=True)
dam2 = None
_flashdepth_wrapper = None
if not IS_FLASHDEPTH:
    # DAM2 only
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
    print(f"[Easy3D] MDE: FlashDepth {FLASHDEPTH_TYPE} ({_mde_path_env}) - lazy load on first image")

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
    depth = wrapper.infer(image)
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
    if IS_FLASHDEPTH:
        return infer_depth_flashdepth(image_path, image_size)
    return infer_depth_dam2(image_path, image_size)

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
    return image, torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

FUSION_STRENGTH = float(__import__("os").environ.get("GENSTEREO_FUSION_STRENGTH", "1.0") or 1.0)  # FIX v28: 1.0=老样子(纯AdaptiveFusion), 0.0=纯Warp, 0.5=一半一半
# FIX v29: Decoupled A/B/C 扩散分辨率, 在这里改
DECOUPLED_LOW_RES = {"decoupled_a": 736, "decoupled_b": 864, "decoupled_c": 1024, "decoupled": 896}  # FIX v29: A=640 B=896 C=1024
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
                    _pad = _torch.zeros_like(_wg[:, :, :, :_dx])
                    _w_shift = _torch.cat([_pad, _wg[:, :, :, :-_dx]], dim=3)
                else:
                    _pad = _torch.zeros_like(_wg[:, :, :, :-_dx])
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
                    _pad = _torch.zeros_like(_w[:, :, :, :_dx_full])
                    _w = _torch.cat([_pad, _w[:, :, :, :-_dx_full]], dim=3)
                else:
                    _pad = _torch.zeros_like(_w[:, :, :, :-_dx_full])
                    _w = _torch.cat([_w[:, :, :, -_dx_full:], _pad], dim=3)
                # 洞也要跟着移, 否则洞对不上
                if _dx_full > 0:
                    _pad_h = _torch.zeros_like(_hole[:, :, :, :_dx_full])
                    _hole = _torch.cat([_pad_h, _hole[:, :, :, :-_dx_full]], dim=3)
                else:
                    _pad_h = _torch.zeros_like(_hole[:, :, :, :-_dx_full])
                    _hole = _torch.cat([_hole[:, :, :, -_dx_full:], _pad_h], dim=3)
        except Exception as _ae:
            _best_dx = 0
            _dx_full = 0
            _err_dx0 = -1.0
            print("[Fusion] v49 align failed: " + str(_ae), flush=True)
        # 对完位移, 诚实混: w = hole*1 + (1-hole)*t, 不推 w, 不 robust, 不作弊
        # V58 Rowan mask-only: valid 100% Warp, hole 100% fused, 只糊洞边缘

        # V66 safe: V64 包进 try, OOM/失败就 fallback 到 V59 (扩 3px), 保证能出图
        try:
            _scale_v64 = float(globals().get("_LAST_SCALE_FACTOR", 0.0) or 0.0) or 0.0759
            _img_w_v64 = int(globals().get("IMAGE_SIZE", 0) or _f.shape[3])
            _cap_v64 = 100.0 if _img_w_v64 >= 1536 else 60.0  # 宽图手 cap 放开
            _base_v64 = max(5.0, min(_cap_v64, _img_w_v64 * _scale_v64 * 0.966))  #修改inpaint范围比例，再要0.96 +1%：-> 0.97
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
        except Exception as _e_v66:
            print("[Fusion] V66 V64 failed, fallback to V59: " + str(_e_v66), flush=True)
            _hole_dil = _F.max_pool2d(_hole, kernel_size=7, stride=1, padding=3)
        _hole_f = _F.avg_pool2d(_hole_dil, kernel_size=5, stride=1, padding=0)  # V66: fallback 羽化 0px
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

_TILING_THRESHOLD_SINGLE = 900 # 在这里开始模块化
_TILING_THRESHOLD_HEAVY = 1536
_TILING_STEP = 768 if SD_VERSION == "v2.1" else 512
_RESOLUTION_LIST = [512,768,896,1024,1088,1216,1280,1344,1472,1600,1664,1728,1792,1920,1984,2048,2240,2304,2560,2688,2816,3072,3136,3328,3456,3584,3840,4096,4224,4352,4480,4608,4864,4928,4992,5120]
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
    print(f"[Easy3D-Step] mode={mode} step={patch} overlap={overlap_ratio:.2f} size={IMAGE_SIZE}", flush=True)
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
    print(f"[Easy3D-Step] Total {_total_steps} steps ({rows}x{cols}), each {patch}", flush=True)
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
            print(f"[Easy3D-Step] Step [{r},{c}] {_step_idx}/{_total_steps} ... elapsed {_time.time() - _t0:.1f}s", flush=True)
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
    output_path = join(output_dir, basename); os.makedirs(output_path, exist_ok=True)
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, "disp.png"), disp_vis)
    image.save(join(output_path, "left.png"))
    to_pil_image(_t2.tensor(warped_full.transpose(2, 0, 1))).save(join(output_path, "warped.png"))
    to_pil_image(_t2.tensor(fusion_np.transpose(2, 0, 1))).save(join(output_path, output_name))
    print(f"[Easy3D-Step] saved {output_name} ({mode}, {rows}x{cols} steps)")
def _warp_by_disparity(img_np, disp_np):
    import numpy as _np
    _h, _w = disp_np.shape[:2]
    _xs, _ys = _np.meshgrid(_np.arange(_w, dtype=_np.float32), _np.arange(_h, dtype=_np.float32))
    _map_x = (_xs + disp_np).astype(_np.float32)
    _map_y = _ys.astype(_np.float32)
    _warped = cv2.remap((img_np * 255.0).astype(_np.uint8), _map_x, _map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (_warped.astype(_np.float32) / 255.0)
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
    disp_full = (normalize_disp(depth) * float(scale_factor) * IMAGE_SIZE).squeeze().cpu().numpy().astype(_np.float32)
    warped_full = _warp_by_disparity(img_np, disp_full)
    if warped_full.shape[:2] != (int(IMAGE_SIZE), int(IMAGE_SIZE)):
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
    synth_full = cv2.resize(synth_low, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(_np.float32)
    mask_full = cv2.resize(mask_low, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(_np.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Easy3D-Decoupled] Diffusion done, elapsed {_time.time() - _t0:.1f}s, start Fusion", flush=True)
    assert synth_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"synth {synth_full.shape}"
    assert warped_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"warped {warped_full.shape}"
    assert mask_full.shape[:2] == (int(IMAGE_SIZE), int(IMAGE_SIZE)), f"mask {mask_full.shape}"
    globals()["_LAST_SCALE_FACTOR"] = float(scale_factor)
    warped_full = _v53_correct_disparity_np(warped_full, synth_full, img_np)
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
    output_path = join(output_dir, basename); os.makedirs(output_path, exist_ok=True)
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, "disp.png"), disp_vis)
    image.save(join(output_path, "left.png"))
    to_pil_image(_t.tensor(warped_full.transpose(2, 0, 1))).save(join(output_path, "warped.png"))
    to_pil_image(_t.tensor(fusion_np.transpose(2, 0, 1))).save(join(output_path, output_name))
    print(f"[Easy3D-Decoupled] saved {output_name} (Warp {IMAGE_SIZE} + diffusion {_low})")
def _generate_for_image(img, depth, output_dir, basename, scale_factor, output_name, pass_id=0):
    if str(_MODE).startswith("decoupled"):
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
    parser.add_argument("--mode", default="quality", choices=["quality", "decoupled", "decoupled_a", "decoupled_b", "decoupled_c"], help="FIX v22")
    parser.add_argument("--decoupled_low", type=int, default=0, help="FIX v22")
    parser.add_argument("--orig_long", type=int, default=0, help="FIX v22")
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
        globals()["_MODE"] = str(getattr(args, "mode", "quality") or "quality").strip().lower()
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
        img, depth = infer_depth(_ip, IMAGE_SIZE)
        if args.gen_mode == "both":
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * 0.5, 'generated_right.png', pass_id=0)
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * -0.5, 'generated_left.png', pass_id=1)
        elif args.gen_mode == "orig_left":
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor * -1.0, 'generated_left.png', pass_id=1)
        else:
            _generate_for_image(img, depth, args.output, base_name, args.scale_factor, 'generated_right.png', pass_id=0)
