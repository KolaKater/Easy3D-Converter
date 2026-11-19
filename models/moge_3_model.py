
import os
from os import path
import torch
import torch.nn.functional as F
from importlib import import_module
# FIX v76: Easy3D 相对路径, models_DIR 原来没定义
from pathlib import Path as _Path
try:
    models_DIR = str(_Path(__file__).resolve().parent)
except Exception:
    models_DIR = "models"
try:
    from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR
except Exception:
    try:
        from base_depth_model import BaseDepthModel, HUB_MODEL_DIR
    except Exception:
        class BaseDepthModel:
            def __init__(self, *a, **k): pass
        HUB_MODEL_DIR = models_DIR
try:
    try:
        from .edge_dilation import dilate_edge  # FIX v86: indent 跟着原来 try: 走, 不再硬写顶格
    except Exception:
        try:
            from edge_dilation import dilate_edge
        except Exception:
            from .dilation import dilate_edge
except Exception:
    try:
        from dilation import dilate_edge
    except Exception:
        def dilate_edge(x, *a, **k): return x  # FIX v76

HAS_VIRTUAL_CAM = False

# ===== MoGe-3 视频稳定修复版 - 哈哈，之前 EMA 反而更不稳，改回最稳的固定范围 =====
MOGE3_GAMMA = 1.6
MOGE3_MAX_DEPTH = 20.0
MOGE3_MIN_DEPTH = 0.1
MOGE3_VIDEO_USE_FIXED_RANGE = True  # 视频用固定 0.1-20.0，不用分位数，绝对不闪
MOGE3_IMAGE_USE_PERCENTILE = True   # 图片用分位数，细节更好
MOGE3_GAMMA_VIDEO = 1.0
MOGE3_GAMMA_IMAGE = 1.6  # 原来你1.6太夸张，改回1.0


# ===== 前景边缘内侧小亮点清理 - VeloDepth同款，metric闭运算，norm开运算 =====
DOT_K = 5
DOT_THRESH = 0.08
VIDEO_SKIP_DOT_CLEAN = True

def _dilate_dot(x, k=3):
    if isinstance(k, (list, tuple)):
        pad = (k[0]//2, k[1]//2)
    else:
        pad = k//2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

def _erode_dot(x, k=3):
    if isinstance(k, (list, tuple)):
        pad = (k[0]//2, k[1]//2)
    else:
        pad = k//2
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)

def clean_shallow_dots(depth, k=None, diff_thresh=None):
    try:
        if k is None:
            k = int(globals().get('DOT_K', 5))
        if diff_thresh is None:
            diff_thresh = float(globals().get('DOT_THRESH', 0.08))
        orig_shape = depth.shape
        squeeze_needed = False
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
            squeeze_needed = True
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
            squeeze_needed = True
        is_metric = depth.max().item() > 2.0
        if is_metric:
            dilated = _dilate_dot(depth, k)
            closed = _erode_dot(dilated, k)
            diff = closed - depth
            mask = (diff > diff_thresh) & (depth < 10.0) & (diff < 2.0)
            mask_f = mask.float()
            mask_eroded = _erode_dot(mask_f, 3)
            dot_mask = mask_f * (1.0 - mask_eroded)
            dot_mask = _dilate_dot(dot_mask, 3)
            dot_mask = F.avg_pool2d(dot_mask, 3, 1, 1).clamp(0,1)
            cleaned = depth * (1-dot_mask) + closed * dot_mask
        else:
            eroded = _erode_dot(depth, k)
            opened = _dilate_dot(eroded, k)
            diff = depth - eroded
            mask = (diff > 0.04) & (depth > 0.6) & (eroded > 0.35) & (diff < 0.25)
            mask_f = mask.float()
            mask_eroded = _erode_dot(mask_f, 3)
            dot_mask = mask_f * (1.0 - mask_eroded)
            dot_mask = _dilate_dot(dot_mask, 3)
            dot_mask = F.avg_pool2d(dot_mask, 3, 1, 1).clamp(0,1)
            cleaned = depth * (1-dot_mask) + opened * dot_mask
        if squeeze_needed:
            if len(orig_shape) == 3:
                cleaned = cleaned.squeeze(1)
            elif len(orig_shape) == 2:
                cleaned = cleaned.squeeze(0).squeeze(0)
        return cleaned
    except Exception as e:
        print(f"[DotClean] failed {e}")
        return depth


MODEL_FILES = {
    "MoGe-3 ViTL": path.join(models_DIR, "moge", "moge-3-vitl", "model.pt"),
    "MoGe3_ViTL": path.join(models_DIR, "moge-3-vitl", "model.pt"),
    "MoGe3_ViTG": path.join(models_DIR, "moge-3-vitg", "model.pt"),
    "moge-3-vitl": path.join(models_DIR, "moge-3-vitl", "model.pt"),
    "moge-3-vitg": path.join(models_DIR, "moge-3-vitg", "model.pt"),
}
CUSTOM_PATHS = [str(_Path(models_DIR) / "moge-3-vitl" / "model.pt"), str(_Path(models_DIR) / "moge" / "moge-3-vitl" / "model.pt")]  # FIX v76: 相对 models/, 不是 cwd

def _resolve_model_path(model_type):
    p = MODEL_FILES.get(model_type)
    if p and path.isfile(p): return p
    for cp in CUSTOM_PATHS:
        if path.isfile(cp): return cp
    return p

def _norm_dilation(ed):
    if ed is None: return (0,0)
    try: v=int(float(ed)); return (v,v)
    except: return (0,0)

def _dilation_to_arg(ed_tuple):
    x,y=ed_tuple
    if x==0 and y==0: return 0
    if x==y: return x
    return (x,y)

def _resolution_to_level(res): return 9

def _import_fixer_module(base_name: str):
    candidates = [f".models.{base_name}", f".{base_name}", f"models.{base_name}", base_name]
    for mod_path in candidates:
        try:
            if mod_path.startswith("."):
                mod = import_module(mod_path, package=__package__)
            else:
                mod = import_module(mod_path)
            return mod, mod_path
        except: continue
    return None, None

def _get_hair_fixer_func(model_type: str):
    mt = str(model_type).lower()
    base_name = "hair_guard_fixer_moge3_vitg" if "vitg" in mt else "hair_guard_fixer_moge3_vitl"
    try:
        mod, real_path = _import_fixer_module(base_name)
        if mod is None: return None
        for fname in ["fix", "apply", "hair_guard_fix"]:
            if hasattr(mod, fname):
                fn = getattr(mod, fname)
                if callable(fn):
                    print(f"[MoGe3] Loaded hair fixer: {real_path}.{fname}")
                    return fn
        return mod if callable(mod) else None
    except: return None

def _try_apply_fixer(fixer_fn, disparity, image=None, **kwargs):
    if fixer_fn is None: return disparity
    try:
        res = fixer_fn(disparity, rgb=image) if image is not None else fixer_fn(disparity)
        if isinstance(res, (tuple, list)): d = res[0] if torch.is_tensor(res[0]) else None
        else: d = res if torch.is_tensor(res) else None
        if d is not None and d.shape == disparity.shape:
            return torch.maximum(d, disparity)  # 只扩张不收缩
        return disparity
    except Exception as e:
        print(f"[MoGe3] Hair fixer fail: {e}")
        return disparity

def _moge3_image_to_disparity(depth, mask=None, device=None):
    """图片模式：用 2%/98% 分位数，细节多，但视频会闪"""
    device = device or depth.device
    depth = torch.nan_to_num(depth, nan=MOGE3_MAX_DEPTH, posinf=MOGE3_MAX_DEPTH, neginf=MOGE3_MIN_DEPTH)
    depth = depth.clamp(MOGE3_MIN_DEPTH, MOGE3_MAX_DEPTH)
    if mask is not None:
        valid = (mask > 0.5) if mask.dtype != torch.bool else mask
        depth_valid = depth[valid] if valid.sum() > 100 else depth[depth > 0.01]
    else:
        depth_valid = depth[depth > 0.01]
    if depth_valid.numel() < 100: depth_valid = depth.reshape(-1)
    try:
        lo = torch.quantile(depth_valid.float(), 0.02)
        hi = torch.quantile(depth_valid.float(), 0.98)
        if (hi - lo).abs() < 1e-3:
            lo = torch.tensor(MOGE3_MIN_DEPTH, device=device)
            hi = torch.tensor(MOGE3_MAX_DEPTH, device=device)
        depth_clipped = depth.clamp(lo, hi)
        disp = (hi - depth_clipped) / (hi - lo + 1e-8)
        if mask is not None:
            valid_mask = (mask > 0.5) if mask.dtype != torch.bool else mask
            disp = torch.where(valid_mask, disp, torch.zeros_like(disp))
    except:
        disp = (MOGE3_MAX_DEPTH - depth) / (MOGE3_MAX_DEPTH - MOGE3_MIN_DEPTH + 1e-8)
    return disp.clamp(0,1)

def _moge3_video_to_disparity(depth, mask=None, device=None):
    """视频模式：固定 0.1-20m，不用分位数，绝对不闪 - 这才是真稳定"""
    device = device or depth.device
    depth = torch.nan_to_num(depth, nan=MOGE3_MAX_DEPTH, posinf=MOGE3_MAX_DEPTH, neginf=MOGE3_MIN_DEPTH)
    depth = depth.clamp(MOGE3_MIN_DEPTH, MOGE3_MAX_DEPTH)
    # 固定范围，不随帧变
    disp = (MOGE3_MAX_DEPTH - depth) / (MOGE3_MAX_DEPTH - MOGE3_MIN_DEPTH + 1e-8)
    if mask is not None:
        valid_mask = (mask > 0.5) if mask.dtype != torch.bool else mask
        disp = torch.where(valid_mask, disp, torch.zeros_like(disp))
    return disp.clamp(0,1)

try:
    from moge.model.v3 import MoGeModel
except ImportError:
    try: from moge.model import MoGeModel
    except: MoGeModel = None

@torch.inference_mode()
# FIX v79: test.py 传进来的是 PIL Image, 不是 Tensor. 先转成 Tensor [3,H,W] float 0..1.
def _v79_to_tensor(x):
    import torch as _t
    if _t.is_tensor(x):
        return x.float() if x.dtype != _t.float32 else x
    try:
        from PIL import Image as _PILImage
        if isinstance(x, _PILImage.Image):
            import numpy as _np
            x = _np.asarray(x.convert('RGB'))
    except Exception:
        pass
    import numpy as _np2
    import torch as _t2
    arr = _np2.asarray(x)
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, axis=2)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        pass  # 已经是 [C,H,W]
    elif arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
    ten = _t2.from_numpy(arr).float()
    if ten.max() > 1.5:  # 0..255 -> 0..1
        ten = ten / 255.0
    return ten

def batch_infer(model, images, resolution=None, refine_steps=6, enable_amp=True, edge_dilation=4, output_device="cpu", device=None, model_type=None, **kwargs):
    # FIX v80 防线: 哪怕上游又把 resolution 塞进 kwargs, 也删掉, 免得 duplicate.
    kwargs.pop('resolution', None); kwargs.pop('model_type', None) if model_type is not None else None
    # FIX v79: PIL / numpy / list 先转 Tensor, 再算 dim()
    images = _v79_to_tensor(images) if not __import__('torch').is_tensor(images) else images
    # 判断是不是视频：iw3 传视频时 B>1 或者 kwargs 里有 is_video
    is_video = (images.dim() == 4 and images.shape[0] > 1) or kwargs.get('is_video', False) or kwargs.get('video_mode', False)
    if is_video:
        print(f"[MoGe3] Video mode detected B={images.shape[0] if images.dim()==4 else 1}, using FIXED range {MOGE3_MIN_DEPTH}-{MOGE3_MAX_DEPTH} for stability")

    if images.dim() == 3:
        images = images.unsqueeze(0)
        batch_mode = False
    else:
        batch_mode = True
    B = images.shape[0]
    if device is None: device = images.device
    out_depths = []
    res_level = _resolution_to_level(resolution)
    hair_fixer_fn = _get_hair_fixer_func(model_type) if model_type else None

    for i in range(B):
        x = images[i]
        orig_h, orig_w = x.shape[1], x.shape[2]
        x_in = x.to(device)

        with torch.autocast(device_type=device.type, enabled=enable_amp and device.type == "cuda"):
            try:
                out_dict = model.infer(x_in, resolution_level=res_level, refine_steps=refine_steps, use_fp16=True)
            except TypeError:
                try:
                    out_dict = model.infer(x_in, num_tokens=2500, refine_steps=refine_steps, use_fp16=True)
                except:
                    out_dict = model.infer(x_in, num_tokens=2500, refine_steps=refine_steps)

        depth = out_dict["depth"] if "depth" in out_dict else out_dict.get("depth_pred", out_dict.get("points"))
        mask = out_dict.get("mask", None)
        if not torch.is_tensor(depth):
            import numpy as np
            depth = torch.from_numpy(depth)
        depth = depth.to(torch.float32)
        if mask is not None and not torch.is_tensor(mask):
            import numpy as np
            mask = torch.from_numpy(mask)
        if mask is not None and mask.dtype != torch.bool:
            mask = mask > 0.5

        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, min=0.0)
        if depth.dim() == 3 and depth.shape[0] == 3: depth = depth[2]
        if depth.dim() > 2: depth = depth.squeeze()
        if depth.shape[0] != orig_h or depth.shape[1] != orig_w:
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if mask is not None and mask.shape != (orig_h, orig_w):
                mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode="nearest").squeeze(0).squeeze(0).bool()

        # 关键：视频用固定范围，图片用分位数
        if is_video and MOGE3_VIDEO_USE_FIXED_RANGE:
            disparity = _moge3_video_to_disparity(depth, mask=mask, device=depth.device)
        else:
            disparity = _moge3_image_to_disparity(depth, mask=mask, device=depth.device)

        # Gamma 视频强制1.0
        gamma_val = float(kwargs.get('gamma', MOGE3_GAMMA_IMAGE if not is_video else MOGE3_GAMMA_VIDEO))
        if abs(gamma_val - 1.0) > 1e-6:
            if is_video:
                print(f"[MoGe3] Video gamma {gamma_val} ignored, forced 1.0")
            else:
                d_min, d_max = disparity.min(), disparity.max()
                if d_max > d_min:
                    norm = (disparity - d_min) / (d_max - d_min + 1e-8)
                    disparity = torch.pow(norm.clamp(0,1), gamma_val) * (d_max - d_min) + d_min

        # Hair Guard V17 只扩张
        enable_hair_guard = kwargs.get('enable_hair_guard', True)
        if enable_hair_guard and hair_fixer_fn is not None:
            disparity = _try_apply_fixer(hair_fixer_fn, disparity, image=x)

        # 清理前景内侧小亮点
        try:
            skip = globals().get('VIDEO_SKIP_DOT_CLEAN', True) and is_video
            if not skip:
                disparity = clean_shallow_dots(disparity)
        except Exception as e:
            print(f"[MoGe3] dot clean skip {e}")

        ed_tuple = _norm_dilation(edge_dilation)
        ed_val = _dilation_to_arg(ed_tuple)
        if ed_val != 0 and ed_val != (0,0):
            disparity = disparity.unsqueeze(0).unsqueeze(0)
            try:
                disparity = dilate_edge(disparity, ed_val)
            except Exception as _e:
                print(f"[MoGe3] dilate_edge FAILED {_e} (ed={ed_val})", flush=True)  # FIX v89: 原来裸 except: pass, 炸了也看不见

            disparity = disparity.squeeze(0).squeeze(0)

        out_depths.append(disparity)

    out = torch.stack(out_depths, dim=0).unsqueeze(1).to(output_device)
    if not batch_mode: out = out.squeeze(0)
    return out

class MoGe3Model(BaseDepthModel):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.force_disparity = True
        self.refine_steps = 0
        self.model = None
        self.device = None
        self._last_resolution = None
        self._model_type = model_type
    def load_model(self, model_type, resolution=None, device=None, **kwargs):
        assert model_type in MODEL_FILES
        if MoGeModel is None: raise ImportError("moge not installed")
        model_path = _resolve_model_path(model_type)
        print(f"[MoGe3] Loading {model_path}")
        model = MoGeModel.from_pretrained(model_path if path.isfile(model_path) else "Ruicheng/moge-3-vitl")
        model = model.to(device).eval()
        self._last_resolution = resolution
        self._model_type = model_type
        return model
    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=4, **kwargs):
        # FIX v80: test.py 传 resolution=768 进 infer(), infer() 里又写 resolution=res, 再把带 resolution 的 **kwargs
        # 一起传给 batch_infer, resolution 传两次 -> got multiple values. 这里先 pop 掉, 只用显式 res.
        kwargs.pop('resolution', None); kwargs.pop('model_type', None)
        kwargs.pop('tta', None); kwargs.pop('low_vram', None)
        res = kwargs.get("resolution", None) or getattr(self, "_last_resolution", None)
        mt = kwargs.get("model_type", None) or getattr(self, "_model_type", None)
        # FIX v79: x 可能是 PIL, 没有 .device. 先转 Tensor, output_device 回退到 self.device
        _x = _v79_to_tensor(x)
        _out_dev = getattr(self, 'device', None) or getattr(_x, 'device', None) or 'cpu'
        return batch_infer(self.model, _x, resolution=res, refine_steps=self.refine_steps, enable_amp=enable_amp, edge_dilation=edge_dilation, output_device=_out_dev, device=self.device, model_type=mt, **kwargs)
    @classmethod
    def get_name(cls): return "moge-3"
    @classmethod
    def get_model_path(cls, model_type): return _resolve_model_path(model_type)
    def is_metric(self): return False
    def is_video_supported(self): return True
    def is_image_supported(self): return True
    @classmethod
    def supported(cls, model_type): return model_type in MODEL_FILES
    @classmethod
    def has_checkpoint_file(cls, model_type):
        p=_resolve_model_path(model_type)
        return path.isfile(p) if p else False
    @classmethod
    def multi_gpu_supported(cls, model_type): return False
    def force_update(self, *a,**k): return False
    def update(self, *a,**k): return False
    def needs_reload(self, *a,**k): return False
