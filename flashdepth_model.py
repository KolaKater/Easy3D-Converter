import os, sys
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# Robust imports for both iw3 and SSL3D
try:
    from nunif.device import device_is_mps, device_is_xpu
except:
    try:
        from nunif.nunif.device import device_is_mps, device_is_xpu
    except:
        def device_is_mps(d): return False
        def device_is_xpu(d): return False

try:
    from .dilation import dilate_edge, edge_dilation_is_enabled
except:
    try:
        from .SSL3D_edge import dilate_edge, edge_dilation_is_enabled
    except:
        try:
            from SSL3D_edge import dilate_edge, edge_dilation_is_enabled
        except:
            def dilate_edge(x, e): return x
            def edge_dilation_is_enabled(e): return False

try:
    from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR
except:
    try:
        from .SSL3D__depth_model import BaseDepthModel, HUB_MODEL_DIR
    except:
        try:
            from SSL3D__depth_model import BaseDepthModel, HUB_MODEL_DIR
        except:
            class BaseDepthModel:
                def __init__(self, model_type):
                    self.model_type=model_type
                    self.model=None
                    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Relative: <Easy3D>/models (flashdepth_model.py lives in <Easy3D>/, so dirname(__file__)/models)
            HUB_MODEL_DIR = path.join(path.dirname(path.abspath(__file__)), "models")

try:
    from .models import DepthAA
except:
    try:
        from nunif.iw3.models import DepthAA
    except:
        DepthAA = None

FLASHDEPTH_ENABLE_HAIR_GUARD = True
FLASHDEPTH_HAIR_EXPAND_DEFAULT = 0.35
DOT_K = 5
DOT_THRESH = 0.08
VIDEO_SKIP_DOT_CLEAN = True

def _dilate_dot(x, k=3):
    pad = (k[0]//2, k[1]//2) if isinstance(k, (list, tuple)) else k//2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
def _erode_dot(x, k=3):
    pad = (k[0]//2, k[1]//2) if isinstance(k, (list, tuple)) else k//2
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)

def clean_shallow_dots(depth, k=None, diff_thresh=None):
    try:
        if k is None: k = int(globals().get('DOT_K', 5))
        if diff_thresh is None: diff_thresh = float(globals().get('DOT_THRESH', 0.08))
        orig_shape = depth.shape
        squeeze_needed = False
        if depth.ndim == 3:
            depth = depth.unsqueeze(1); squeeze_needed = True
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0); squeeze_needed = True
        is_metric = depth.max().item() > 2.0
        if is_metric:
            dilated = _dilate_dot(depth, k); closed = _erode_dot(dilated, k)
            diff = closed - depth
            mask = (diff > diff_thresh) & (depth < 10.0) & (diff < 2.0)
            mask_f = mask.float()
            mask_eroded = _erode_dot(mask_f, 3)
            dot_mask = mask_f * (1.0 - mask_eroded)
            dot_mask = _dilate_dot(dot_mask, 3)
            dot_mask = F.avg_pool2d(dot_mask, 3, 1, 1).clamp(0,1)
            cleaned = depth * (1-dot_mask) + closed * dot_mask
        else:
            eroded = _erode_dot(depth, k); opened = _dilate_dot(eroded, k)
            diff = depth - eroded
            mask = (diff > 0.04) & (depth > 0.6) & (eroded > 0.35) & (diff < 0.25)
            mask_f = mask.float()
            mask_eroded = _erode_dot(mask_f, 3)
            dot_mask = mask_f * (1.0 - mask_eroded)
            dot_mask = _dilate_dot(dot_mask, 3)
            dot_mask = F.avg_pool2d(dot_mask, 3, 1, 1).clamp(0,1)
            cleaned = depth * (1-dot_mask) + opened * dot_mask
        if squeeze_needed:
            if orig_shape != cleaned.shape:
                if len(orig_shape) == 3: cleaned = cleaned.squeeze(1)
                elif len(orig_shape) == 2: cleaned = cleaned.squeeze(0).squeeze(0)
        return cleaned
    except Exception as e:
        return depth

def find_flashdepth_root():
    # Relative only: <Easy3D>/models/FlashDepth (+ checkpoints fallback)
    _base = path.join(path.dirname(path.abspath(__file__)), "models")
    candidates = [
        path.join(_base, "FlashDepth"),
        path.join(_base, "FlashDepth", "flashdepth"),
        path.join(HUB_MODEL_DIR, "checkpoints", "FlashDepth"),
        path.join(HUB_MODEL_DIR, "checkpoints", "flashdepth"),
    ]
    for root in candidates:
        if not root: continue
        if path.exists(path.join(root, "flashdepth", "model.py")) or path.exists(path.join(root, "model.py")): return root
        if path.exists(path.join(root, "configs")): return root
    return None

FLASH_ROOT = find_flashdepth_root()
OfficialFlashDepth = None
if FLASH_ROOT:
    for p in [FLASH_ROOT, path.join(FLASH_ROOT, "..")]:
        if p not in sys.path: sys.path.insert(0, p)
    try:
        from flashdepth.model import FlashDepth as OfficialFlashDepth
    except: pass

MIN_RESOLUTION = 224
def batch_preprocess(x, lower_bound=392, max_aspect_ratio=4, limit_resolution=False, upper_bound=512):
    B,C,H,W = x.shape
    ensure_multiple_of = 14
    if limit_resolution and lower_bound > min(W,H):
        lower_bound = min(W,H)
        lower_bound -= lower_bound % ensure_multiple_of
        lower_bound = max(lower_bound, MIN_RESOLUTION)
    if W < H: scale_factor = lower_bound / W
    else: scale_factor = lower_bound / H
    new_h = int(H*scale_factor); new_w = int(W*scale_factor)
    if new_h > upper_bound or new_w > upper_bound:
        scale = upper_bound / max(new_h,new_w)
        new_h = int(new_h*scale); new_w = int(new_w*scale)
    if new_h < new_w: new_w = min(new_w, int(max_aspect_ratio*new_h))
    else: new_h = min(new_h, int(max_aspect_ratio*new_w))
    new_h -= new_h % ensure_multiple_of
    new_w -= new_w % ensure_multiple_of
    new_h = max(new_h, ensure_multiple_of); new_w = max(new_w, ensure_multiple_of)
    antialias = not (device_is_mps(x.device) or device_is_xpu(x.device))
    dtype = x.dtype
    x = F.interpolate(x, size=(new_h,new_w), mode="bilinear", align_corners=False, antialias=antialias)
    if x.dtype != dtype: x = x.to(dtype)
    x.clamp_(0,1)
    mean = torch.tensor([0.485,0.456,0.406], dtype=x.dtype, device=x.device).reshape(1,3,1,1)
    stdv = torch.tensor([0.229,0.224,0.225], dtype=x.dtype, device=x.device).reshape(1,3,1,1)
    x.sub_(mean).div_(stdv)
    return x

def find_config_dir(model_type):
    base = HUB_MODEL_DIR
    name_map = {"FlashDepth":"flashdepth","FlashDepth-L":"flashdepth-l","FlashDepth-S":"flashdepth-s"}
    short = name_map.get(model_type,"flashdepth-s")
    ssl3d_base = path.join(path.dirname(path.abspath(__file__)), "models", "FlashDepth")  # relative
    cands = [
        path.join(ssl3d_base, "configs", short),
        path.join(ssl3d_base, "flashdepth", "configs", short),
        path.join(base,"checkpoints","FlashDepth","configs",short),
        path.join(FLASH_ROOT,"configs",short) if FLASH_ROOT else None,
    ]
    for c in cands:
        if c and path.exists(c):
            if path.exists(path.join(c,"config.yaml")) or any(f.endswith(".pth") for f in os.listdir(c)):
                return c
    for fp in [path.join(ssl3d_base, "configs", short, "iter_14001.pth"), path.join(ssl3d_base, "configs", short, "iter_10001.pth"), path.join(ssl3d_base, "configs", short, "iter_43002.pth")]:
        if path.exists(fp):
            return path.dirname(fp)
    return None

def find_checkpoint_file(config_dir):
    if not config_dir or not path.exists(config_dir): return None
    for f in os.listdir(config_dir):
        if f.endswith(".pth"): return path.join(config_dir,f)
    return None

MODEL_CONFIGS = {
    "FlashDepth":{"short":"flashdepth","default_res":518},
    "FlashDepth-L":{"short":"flashdepth-l","default_res":518},  # default only; test.py overrides with GUI Resolution
    "FlashDepth-S":{"short":"flashdepth-s","default_res":392},  # default only; test.py overrides with GUI Resolution
}

class FlashDepthModelWrapper(BaseDepthModel):
    MODEL_FILES = {
        "FlashDepth":"flashdepth/iter_43002.pth",
        "FlashDepth-L":"flashdepth-l/iter_10001.pth",
        "FlashDepth-S":"flashdepth-s/iter_14001.pth",
    }
    def __init__(self, model_type):
        super().__init__(model_type)
        self.metric_depth=False
        self.depth_aa=None
        self.is_official=False

    def reset_state(self):
        if hasattr(self,'model') and self.model is not None:
            if hasattr(self.model,'mamba') and hasattr(self.model.mamba,'start_new_sequence'):
                try: self.model.mamba.start_new_sequence()
                except: pass
            if hasattr(self.model,'reset_memory'):
                try: self.model.reset_memory()
                except: pass

    def load_model(self, model_type, resolution=None, device=None):
        if hasattr(self,'model') and self.model is not None:
            if resolution is not None:
                try:
                    self.model.prep_lower_bound = int(resolution)
                    if self.model.prep_lower_bound % 14 != 0:
                        self.model.prep_lower_bound += (14 - self.model.prep_lower_bound % 14)
                except: pass
            return self.model
        is_cpu = device.type=='cpu' if device is not None else False
        cfg_info = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["FlashDepth-S"])
        config_dir = find_config_dir(model_type)
        ckpt_path = find_checkpoint_file(config_dir)
        print(f"[FlashDepth] Config dir: {config_dir}, ckpt: {ckpt_path}")
        if OfficialFlashDepth is not None and config_dir and ckpt_path:
            try:
                from omegaconf import OmegaConf
                yaml_path = path.join(config_dir,"config.yaml")
                if path.exists(yaml_path):
                    yaml_cfg = OmegaConf.load(yaml_path)
                    model = OfficialFlashDepth(batch_size=1, hybrid_configs=None, training=False, **yaml_cfg.model)
                else:
                    if model_type=="FlashDepth-L":
                        model = OfficialFlashDepth(batch_size=1, hybrid_configs=None, training=False, patch_size=14, encoder='vitl', features=256, out_channels=[256,512,1024,1024], use_mamba=True, mamba_in_dpt_layer=[3], d_state=256)
                    else:
                        enc='vits'; feats=64; out_ch=[48,96,192,384]; ml=[3] if model_type=="FlashDepth-S" else [0]
                        model = OfficialFlashDepth(batch_size=1, hybrid_configs=None, training=False, patch_size=14, encoder=enc, features=feats, out_channels=out_ch, use_mamba=True, mamba_in_dpt_layer=ml, d_state=256)
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                sd = checkpoint["model"] if isinstance(checkpoint,dict) and "model" in checkpoint else checkpoint
                model.load_state_dict(sd, strict=False)
                model = model.to(device).eval()
                if not is_cpu: model = model.half()
                model.device=device
                model.prep_lower_bound = resolution or cfg_info["default_res"]
                model.patch_size=14
                self.model=model
                self.is_official=True
                if DepthAA is not None:
                    try:
                        self.depth_aa = DepthAA().load().eval().to(device).float()
                    except:
                        self.depth_aa = None
                print(f"[FlashDepth] Official model loaded: {model_type}")
                return model
            except Exception as e:
                print(f"[FlashDepth] Official load failed: {e}")
                import traceback; traceback.print_exc()
        raise FileNotFoundError(f"Failed to load official FlashDepth. config_dir={config_dir} ckpt={ckpt_path}")

    @torch.inference_mode()
    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=2, depth_aa=False, limit_resolution=None, **kwargs):
        is_cpu = self.device.type=='cpu'
        if not torch.is_tensor(x):
            x = TF.to_tensor(x).to(self.device)
        if x.ndim==3: x=x.unsqueeze(0)
        if low_vram: limit_resolution=224
        elif limit_resolution is None: limit_resolution=True
        if is_cpu: x=x.float()
        else: x=x.half()
        prep_res = 224 if low_vram else getattr(self.model,'prep_lower_bound',392)
        effective_upper = max(prep_res * 2, 1024)
        x_norm = batch_preprocess(x, prep_res, limit_resolution=limit_resolution, upper_bound=effective_upper)
        B,C,H,W = x_norm.shape
        patch_h, patch_w = H//self.model.patch_size, W//self.model.patch_size
        amp_enabled = enable_amp and not is_cpu
        if tta:
            self.reset_state()
            with torch.autocast(device_type='cuda' if not is_cpu else 'cpu', dtype=torch.float16, enabled=amp_enabled):
                dpt_features = self.model.get_dpt_features(x_norm, input_shape=(B,C,H,W))
                out_orig = self.model.final_head(dpt_features, patch_h, patch_w)
            self.reset_state()
            x_flip = torch.flip(x_norm, dims=[3])
            with torch.autocast(device_type='cuda' if not is_cpu else 'cpu', dtype=torch.float16, enabled=amp_enabled):
                dpt_features_flip = self.model.get_dpt_features(x_flip, input_shape=(B,C,H,W))
                out_flip = self.model.final_head(dpt_features_flip, patch_h, patch_w)
            depth = (out_orig + torch.flip(out_flip, dims=[3]))*0.5
        else:
            with torch.autocast(device_type='cuda' if not is_cpu else 'cpu', dtype=torch.float16, enabled=amp_enabled):
                dpt_features = self.model.get_dpt_features(x_norm, input_shape=(B,C,H,W))
                depth = self.model.final_head(dpt_features, patch_h, patch_w)
        depth = depth.unsqueeze(1).float()
        if edge_dilation_is_enabled(edge_dilation):
            depth = dilate_edge(depth, edge_dilation)
        try:
            is_video = kwargs.get('is_video', False) or (depth.shape[0] > 1)
            skip = globals().get('VIDEO_SKIP_DOT_CLEAN', True) and is_video
            if not skip:
                depth = clean_shallow_dots(depth)
        except: pass
        return depth.squeeze(1)

    def compile(self): pass
    def clear_compiled_model(self): pass
    @classmethod
    def get_name(cls): return "FlashDepth"
    @classmethod
    def supported(cls, model_type): return model_type in cls.MODEL_FILES or model_type in MODEL_CONFIGS
    @classmethod
    def has_checkpoint_file(cls, model_type):
        d=find_config_dir(model_type); p=find_checkpoint_file(d); return p is not None and path.exists(p)
    @classmethod
    def get_model_path(cls, model_type):
        d=find_config_dir(model_type); p=find_checkpoint_file(d); return p or path.join(HUB_MODEL_DIR,"checkpoints",cls.MODEL_FILES.get(model_type,"flashdepth-s/iter_14001.pth"))
    def is_metric(self): return False
    @classmethod
    def multi_gpu_supported(cls, model_type): return False
    @classmethod
    def force_update(cls): pass

FlashDepthModel = FlashDepthModelWrapper
