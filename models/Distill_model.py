# Distill_model.py - Distill Any Depth ViTB - Easy3D wrapper
# 放法: D:\3D\Easy3D\models\Distill_model.py
# 权重: D:\3D\Easy3D\models\distill_any_depth_vitb.safetensors
# 全相对路径, 无绝对路径. GUI 里选 Distill-B 即可.
# Edge Protection Level: 由 test.py 传 edge_dilation 进来 (GUI 的 GENSTEREO_DILATION),
#   0 = 不膨, >0 = 按 level 膨. 不再像旧版那样写死 2px.
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from PIL import Image
import cv2

if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
    torch._dynamo.config.suppress_errors = True
if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'config'):
    try:
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.triton.max_autotune = False
    except Exception:
        pass
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass

# ---- 相对路径: 不管 Distill_model.py 在 Easy3D 根目录还是 models/ 里, 都能找到根目录 ----
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name.lower() == "models" else _HERE
# 权重只认这一处 (用户实际存放位置)
MODEL_PATH = _ROOT / "models" / "distill_any_depth_vitb.safetensors"
# 兜底: 旧路径 models/da2/... / 文件旁边, 找不到主路径时才试
_FALLBACKS = [
    _ROOT / "models" / "da2" / "distill_any_depth_vitb.safetensors",
    _HERE / "distill_any_depth_vitb.safetensors",
]
HUB_CACHE_CKPT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "distill_any_depth_vitb.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def _resolve_checkpoint():
    for _c in [MODEL_PATH, *_FALLBACKS]:
        if _c.exists():
            return _c
    return MODEL_PATH  # 报错时报主路径, 让用户一眼看懂该放哪


def ensure_checkpoint():
    ckpt = _resolve_checkpoint()
    if not ckpt.exists():
        print(f"[Distill] ERROR: {ckpt} not found")
        return False
    HUB_CACHE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not HUB_CACHE_CKPT.exists() or HUB_CACHE_CKPT.stat().st_size != ckpt.stat().st_size:
        shutil.copy2(str(ckpt), str(HUB_CACHE_CKPT))
    return True


def _load_dilate_edge():
    """找 dilation.py 里的 dilate_edge, 找不到返回 None (上层回退 cv2)."""
    import importlib.util as _ilu
    for _fp in (_ROOT / "models" / "dilation.py", _ROOT / "dilation.py",
                _HERE / "dilation.py", _HERE / "models" / "dilation.py"):
        if _fp.is_file():
            try:
                _spec = _ilu.spec_from_file_location("_easy3d_dilation_distill", str(_fp))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                if hasattr(_mod, "dilate_edge"):
                    return _mod.dilate_edge
            except Exception:
                continue
    return None


def _apply_edge_protection_tensor(depth_t, edge_dilation):
    """depth_t: [1,1,H,W] 0..1 tensor, 在归一化之后膨 (和 MoGe v91 同理, 否则归一化会把膨压回去)."""
    try:
        _ed = int(float(edge_dilation or 0))
    except Exception:
        _ed = 0
    if _ed <= 0:
        return depth_t
    _dilate = _load_dilate_edge()
    if _dilate is not None:
        try:
            return _dilate(depth_t, _ed)
        except Exception as _e:
            print(f"[Distill] dilate_edge failed ({_e}), fallback to cv2", flush=True)
    # 兜底: 旧版行为的 parametric 版, kernel 3x3, iterations = level
    try:
        _d = depth_t.detach().cpu().numpy()[0, 0]
        _u8 = (_d * 255).astype(np.uint8)
        _dil = cv2.dilate(_u8, np.ones((3, 3), np.uint8), iterations=max(1, _ed))
        return torch.tensor(_dil.astype(np.float32) / 255.0, dtype=depth_t.dtype, device=depth_t.device).unsqueeze(0).unsqueeze(0)
    except Exception:
        return depth_t


def load_model():
    global _model
    if _model is not None:
        return _model
    ensure_checkpoint()
    model = torch.hub.load("nagadomi/Depth-Anything_iw3:main", "DistillAnyDepth", encoder='v2_vitb', verbose=False, trust_repo=True)
    model.to(device).eval()
    _model = model
    return _model


def batch_preprocess(x, lower_bound=700):
    # Distill 官方 preprocess: lower_bound 封顶 700, 超了也不往上加, 保持它原来的画质.
    if lower_bound is None:
        lower_bound = 700
    lower_bound = int(lower_bound)
    if lower_bound > 700:
        lower_bound = 700
    if lower_bound < 14:
        lower_bound = 14
    B, C, H, W = x.shape
    scale = lower_bound / min(H, W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    new_h -= new_h % 14
    new_w -= new_w % 14
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


@torch.inference_mode()
def get_depth(image: Image.Image, resolution=None, edge_dilation=0):
    """image: PIL RGB. 返回 [1,1,H,W] float 0..1 tensor (和 DA3-L/MoGe wrapper 同格式, test.py 不用改)."""
    model = load_model()
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device)
    # resolution 是 test.py 传来的 image_size, Distill 内部封顶 700 (见 batch_preprocess)
    x = batch_preprocess(x, lower_bound=int(resolution) if resolution else 700)
    if device == "cuda":
        ctx = torch.autocast(device_type='cuda', enabled=True)
    else:
        ctx = torch.autocast(device_type='cpu', enabled=False)
    with ctx:
        depth = model(x)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h, w), mode="bicubic", align_corners=False)
        depth = depth.squeeze().cpu().numpy()
    d_min = np.percentile(depth, 1)
    d_max = np.percentile(depth, 99.9)
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_norm = np.power(depth_norm, 0.5)
    depth_norm = np.clip(depth_norm * 0.92 + 0.0001, 0, 1)
    # Edge Protection: 在归一化/ tone mapping 之后做, level 由 GUI 传, 0 = 不膨
    depth_t = torch.tensor(depth_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    depth_t = _apply_edge_protection_tensor(depth_t, edge_dilation)
    depth_norm = depth_t.squeeze().cpu().numpy()
    if str(os.environ.get("GENSTEREO_DISTILL_DEBUG", "")).strip() == "1":
        try:
            debug_path = _ROOT / "output" / "debug_depth.png"
            debug_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_path), (depth_norm * 255).astype(np.uint8))
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return depth_t.to(device)


class DistillAnyDepthModel:
    """test.py _get_new_mde_wrapper 用的 wrapper, 接口和 DA_v3_l / MoGe3Model 对齐."""
    def __init__(self, model_type="distill_any_depth_vitb", device=None):
        self.model_type = model_type
        self.device = torch.device(device) if device is not None else torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, model_type="distill_any_depth_vitb", resolution=None, device=None, **kwargs):
        if device is not None:
            self.device = torch.device(device)
        globals()['device'] = str(self.device)
        self.model = load_model()
        print(f"[Distill] loaded distill_any_depth_vitb ({_resolve_checkpoint()})", flush=True)
        return self.model

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False, resolution=None, **kwargs):
        if not isinstance(x, Image.Image):
            from torchvision.transforms.functional import to_pil_image
            if torch.is_tensor(x):
                _t = x.detach().cpu()
                if _t.ndim == 4:
                    _t = _t[0]
                x = to_pil_image(_t.clamp(0, 1))
            else:
                x = Image.fromarray(np.array(x).astype(np.uint8))
        return get_depth(x.convert("RGB"), resolution=resolution, edge_dilation=edge_dilation)

    @classmethod
    def get_name(cls):
        return "DistillAnyDepthViTB"
