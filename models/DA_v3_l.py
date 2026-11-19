# DA_v3_l.py -- Depth-Anything-3 Large for Easy3D (clean rewrite, replaces broken iw3 fragment)
# 附件那份 DA_v3_l.py 是残缺的 iw3 片段: models_DIR 没定义, autocast/batch_preprocess/BaseDepthModel
# 都没 import, batch_infer 里 z 没定义, load_model 还去 torch.hub.load("nagadomi/...").
# 这份按 DA_v3_bs.py (DA3-B/S 现在能跑的那套) 重写: 走 depth_anything_3 官方 API,
# 只是换成 Large. 不依赖 iw3, 不依赖 nagadomi.
#
# 权重查找顺序 (相对路径, 文件在 models/ 里):
#   1) models/da3-large/config.json 存在 -> 用 models/da3-large/ 本地目录 (Install_DA3_L 下的就是这个)
#   2) models/da3-large/model.safetensors 存在, 但缺 config.json -> 走 HF depth-anything/DA3-LARGE
#   3) 只有 models/da3mono-large.safetensors (你手动下的那个) -> 它 alone 缺 config.json,
#      官方 API 没法直接从单个 .safetensors 加载, 所以同样走 HF DA3-LARGE, 不会静默用错文件.
#      想离线用本地: 先跑 Install_DA3_L.py, 它会把完整 DA3-LARGE (含 config.json) 下到 models/da3-large/
#   4) 都没有 -> 走 HF depth-anything/DA3-LARGE, 首次运行会自动下载到 HF 缓存
#
# test.py / GUI 接入: 见 patch_add_da3_large_gui_test.py, GUI 里选 DA3-L, test.py 会 import 这个文件.
import os
from os import path
from pathlib import Path

try:
    from .edge_dilation import dilate_edge, edge_dilation_is_enabled  # FIX v86: indent 跟着原来 try: 走, 不再硬写顶格
except Exception:
    try:
        from edge_dilation import dilate_edge, edge_dilation_is_enabled
    except Exception:
        from .dilation import dilate_edge, edge_dilation_is_enabled

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# --- models_DIR: 文件在 models/ 里, models_DIR = 文件所在 models/ 的绝对路径, 绝不用 D:\ 写死 ---
try:
    _HERE = Path(__file__).resolve().parent
    models_DIR = str(_HERE) if _HERE.name.lower() == "models" else str(_HERE / "models")
except Exception:
    models_DIR = "models"

# --- BaseDepthModel: 优先用 Easy3D 自带的, 没有就用最小 stub, 保证 import 不炸 ---
try:
    from .base_depth_model import BaseDepthModel
except Exception:
    try:
        from base_depth_model import BaseDepthModel
    except Exception:
        try:
            from models.base_depth_model import BaseDepthModel
        except Exception:
            class BaseDepthModel(torch.nn.Module):  # fallback stub
                def __init__(self, model_type=None, **kwargs):
                    super().__init__()
                    self.model_type = model_type
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.raw_output = kwargs.get("raw_output", False)
                    self.limit_resolution = kwargs.get("limit_resolution", False)

NAME_MAP_L = {
    "Any_V3_Mono_Large": "da3-large",
    "Any_V3_Mono_Larg": "da3-large",
    "da3-large": "da3-large",
    "da3_large": "da3-large",
    "DA3-L": "da3-large",
}
MODEL_FILES_L = {
    "Any_V3_Mono_Large": path.join(models_DIR, "da3-large", "model.safetensors"),
    "Any_V3_Mono_Larg": path.join(models_DIR, "da3-large", "model.safetensors"),
    "da3-large": path.join(models_DIR, "da3-large", "model.safetensors"),
    "da3_large": path.join(models_DIR, "da3-large", "model.safetensors"),
    "DA3-L": path.join(models_DIR, "da3-large", "model.safetensors"),
}
# 你手动下的那个文件, 也认, 但只认作“有文件”, 正式加载仍走下面 _resolve_model_dir() 的逻辑
MONO_LARGE_CANDIDATES = [
    path.join(models_DIR, "da3mono-large.safetensors"),
    path.join(models_DIR, "da3-large", "da3mono-large.safetensors"),
    path.join(models_DIR, "da3-large", "model.safetensors"),
]

def _find_da3_l_checkpoint(model_type="da3-large"):
    """返回优先的 checkpoint 路径 (仅用于显示/检查, 正式加载用 _resolve_model_dir)."""
    for cand in MONO_LARGE_CANDIDATES:
        if path.exists(cand):
            return cand
    return path.join(models_DIR, "da3-large", "model.safetensors")

def _resolve_model_dir():
    """决定 DepthAnything3.from_pretrained() 用本地目录还是 HF id."""
    local_dir = path.join(models_DIR, "da3-large")
    if path.exists(path.join(local_dir, "config.json")):
        print(f"[DA3-L] Using local dir (has config.json): {local_dir}")
        return local_dir
    for cand in MONO_LARGE_CANDIDATES:
        if path.exists(cand):
            print(f"[DA3-L] Found {cand} but no config.json in da3-large/, falling back to HF depth-anything/DA3-LARGE (run Install_DA3_L.py to download full local copy)")
            break
    return "depth-anything/DA3-LARGE"

@torch._dynamo.disable
def _forward_l(model, x, enable_amp, sky_thresh=0.30, raw_output=False):
    # 和 DA_v3_bs._forward_bs 同逻辑: model 是 OfficialWrapper, x 已做 ImageNet 归一化
    x = x.unsqueeze(1)
    try:
        from nunif.device import autocast as autocast_ctx
        ctx = autocast_ctx(device=x.device, dtype=torch.bfloat16) if enable_amp else torch.no_grad()
    except Exception:
        ctx = torch.no_grad()
    with ctx:
        out = model(x)
    depths = out["depth"]
    sky_probs = out["sky"]
    sky_masks = sky_probs > sky_thresh
    sky_weights = (sky_probs.clamp(sky_thresh, 1.0) - sky_thresh) / (1.0 - sky_thresh)
    disparity_maps = []
    for depth, sky_mask, sky_weight in zip(depths, sky_masks, sky_weights):
        if not raw_output:
            non_sky = sky_mask.numel() - sky_mask.sum()
            if non_sky < 10:
                d_out = torch.zeros_like(depth)
            else:
                d = depth.float()
                d = 1.0 / (d + 0.2)
                d = d * (1 - sky_weight)
                d_out = d
        else:
            try:
                max_rel = torch.quantile(depth[~sky_mask], 0.99)
                d_out = (depth * (1 - sky_weight) + sky_weight * max_rel).clamp(max=max_rel)
            except Exception:
                d_out = depth
        disparity_maps.append(d_out)
    return torch.stack(disparity_maps) if len(disparity_maps) > 1 else disparity_maps[0]

@torch._dynamo.disable
def batch_infer_l(model, x, enable_amp=True, sky_thresh=0.30, raw_output=False):
    batch = True
    if x.ndim == 3:
        x = x.unsqueeze(0); batch = False
    elif x.ndim == 5 and x.shape[1] == 1:
        x = x.squeeze(1); batch = True
    out = _forward_l(model, x, enable_amp, sky_thresh=sky_thresh, raw_output=raw_output)
    if not batch:
        out = out.squeeze(0)
    try:
        if out.ndim == 3 and not raw_output:
            d_min = out.amin(dim=(-2, -1), keepdim=True)
            d_max = out.amax(dim=(-2, -1), keepdim=True)
            out = (out - d_min) / (d_max - d_min + 1e-6)
            out = out.clamp(0, 1)
    except Exception as e:
        print(f"[DA3-L] scaler failed {e}")
    print(f"[DA3-L] batch_infer res {out.shape}")
    return out

class DepthAnythingV3LargeModel(BaseDepthModel):
    """DA3-L wrapper, 接口和 DA_v3_bs.DepthAnythingV3BSModel 一致: load_model / load / infer."""
    def __init__(self, model_type="da3-large", resolution=None, device=None, raw_output=False):
        try:
            super().__init__(model_type)
        except Exception:
            super().__init__()
            self.model_type = model_type
        self.model_type = model_type
        self.raw_output = raw_output
        self.limit_resolution = False
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def load_model(self, model_type="da3-large", resolution=None, device=None, raw_output=False):
        ckpt = _find_da3_l_checkpoint(model_type)
        print(f"[DA3-L] checkpoint: {ckpt}")
        self.raw_output = raw_output
        from depth_anything_3.api import DepthAnything3
        model_dir = _resolve_model_dir()
        print(f"[DA3-L] Trying official API from {model_dir}")
        try:
            da3_model = DepthAnything3.from_pretrained(model_dir)
        except Exception as e:
            print(f"[DA3-L] Official API failed: {e}")
            raise FileNotFoundError(f"DA3-L failed to load {model_type} from {ckpt} (model_dir={model_dir})")
        da3_model = da3_model.to(device=device if device is not None else "cpu")
        da3_model.eval()
        # OfficialWrapper: 和 DA_v3_bs 一样, forward 里把归一化后的 x 反归一化 -> PIL -> da3_model.inference()
        class OfficialWrapper(torch.nn.Module):
            def __init__(self, da3_model):
                super().__init__()
                self.da3_model = da3_model
                self.prep_lower_bound = resolution if resolution else 700
                self.current_process_res = None
            def set_process_res(self, res):
                self.current_process_res = res
                self.prep_lower_bound = res if res else 700
            @torch._dynamo.disable
            def forward(self, x):
                import numpy as np
                from PIL import Image
                if x.ndim == 5:
                    x = x.squeeze(1)
                x_f = x.float()
                mean = torch.tensor([0.485, 0.456, 0.406], device=x_f.device, dtype=torch.float32).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x_f.device, dtype=torch.float32).view(1, 3, 1, 1)
                x_denorm = (x_f * std + mean).clamp(0, 1)
                images = []
                for i in range(x_denorm.shape[0]):
                    img_np = (x_denorm[i].permute(1, 2, 0).contiguous().detach().cpu().numpy() * 255).astype(np.uint8)
                    images.append(Image.fromarray(img_np))
                with torch.no_grad():
                    res = getattr(self, "current_process_res", None)
                    try:
                        if res is None:
                            pred = self.da3_model.inference(images)
                        else:
                            pred = self.da3_model.inference(images, process_res=res, process_res_method="lower_bound_resize")
                    except TypeError:
                        try:
                            pred = self.da3_model.inference(images, process_res=res) if res is not None else self.da3_model.inference(images)
                        except Exception:
                            pred = self.da3_model.inference(images)
                    depths = pred.depth
                    if isinstance(depths, np.ndarray):
                        depth_t = torch.from_numpy(depths).float().to(x.device)
                    else:
                        depth_t = torch.tensor(depths).float().to(x.device)
                    if depth_t.ndim == 2:
                        depth_t = depth_t.unsqueeze(0)
                    if depth_t.ndim == 3 and depth_t.shape[0] != x_denorm.shape[0]:
                        depth_t = depth_t.unsqueeze(0)
                    if depth_t.ndim == 4:
                        depth_t = depth_t.squeeze(1)
                    B, H, W = depth_t.shape[0], depth_t.shape[1], depth_t.shape[2]
                    sky = torch.zeros((B, H, W), device=x.device, dtype=depth_t.dtype)
                    return {"depth": [depth_t], "sky": sky}
        self.model = OfficialWrapper(da3_model)
        print(f"[DA3-L] Official API loaded {model_type}")
        return self.model

    def load(self, gpu=None, resolution=None, limit_resolution=False, **kwargs):
        if gpu is None:
            device = self.device
        else:
            gpu_id = gpu[0] if isinstance(gpu, (list, tuple)) and len(gpu) > 0 else gpu if isinstance(gpu, int) else 0
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.load_model(self.model_type, resolution=resolution, device=device, raw_output=self.raw_output)
        try:
            self.model = self.model.to(device)
        except Exception:
            pass
        self.model.eval()
        self.device = device

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False, resolution=None, **kwargs):
        # 和 DA_v3_bs.infer 同逻辑: PIL -> Tensor, 直接 resize 到 lb, ImageNet 归一化
        if not torch.is_tensor(x):
            from PIL import Image
            if isinstance(x, Image.Image) and x.mode == "L":
                x = x.convert("RGB")
            x = TF.to_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass
        try:
            target_device = self.device
        except AttributeError:
            target_device = next(self.model.parameters()).device
        x = x.to(target_device)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1).clone()
        if resolution is None:
            resolution = kwargs.get("lower_bound", None)
        if resolution is None:
            resolution = getattr(self.model, "prep_lower_bound", 700)
        lb = resolution if resolution else 700
        if lb < 280:
            lb = 280
        lb = (int(lb) // 14) * 14
        try:
            self.model.set_process_res(lb)
        except Exception:
            pass
        print(f"[DA3-L] SINGLE infer customer lb={lb}")
        try:
            B, C, H, W = x.shape
            short = min(H, W)
            scale = lb / short
            new_h = int(round(H * scale / 14) * 14)
            new_w = int(round(W * scale / 14) * 14)
            x_in = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x_in.device, dtype=x_in.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x_in.device, dtype=x_in.dtype).view(1, 3, 1, 1)
            x_in = (x_in - mean) / std
            print(f"[DA3-L] direct resize {H} -> {new_h} lb={lb}")
        except Exception as e:
            print(f"[DA3-L] direct resize fallback {e}")
            x_in = x
        if not low_vram:
            if tta:
                x_in = torch.cat([x_in, torch.flip(x_in, dims=[3])], dim=0)
            out = batch_infer_l(self.model, x_in, enable_amp, raw_output=self.raw_output)
        else:
            out = batch_infer_l(self.model, x_in, enable_amp, raw_output=self.raw_output)
            if tta:
                out2 = batch_infer_l(self.model, torch.flip(x_in, dims=[3]), enable_amp, raw_output=self.raw_output)
                out = torch.cat([out, out2], dim=0)
        # FIX v82 DA3-L: 之前收了 edge_dilation 但没干活, 这里真正扩边缘 (和 DA3-B/S 一样)
        try:
            if edge_dilation is not None and int(edge_dilation) != 0:
                _ed = max(0, min(50, int(edge_dilation)))
                _o = out.unsqueeze(0).unsqueeze(0) if out.ndim == 3 else (out.unsqueeze(0) if out.ndim == 2 else out)
                _o = dilate_edge(_o, _ed)
                out = _o.squeeze(0).squeeze(0) if _o.ndim == 4 else _o
        except Exception as _e:
            print(f"[DA3-L] dilate_edge skip {_e}")
        if out.ndim == 2:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.ndim == 3:
            out = out.unsqueeze(1)
        # FIX v88: DA3-L 原来 infer() 里根本没调 dilate_edge, Edge Protection Level 传了也白传, 这里按 DA3-B 同款补上
        if edge_dilation_is_enabled(edge_dilation):
            if out.ndim == 2:
                out = out.unsqueeze(0).unsqueeze(0)
            elif out.ndim == 3:
                out = out.unsqueeze(1)
            if out.ndim == 4:
                out = dilate_edge(out, edge_dilation)
                out = out.squeeze(1) if out.ndim == 4 else out
        return out

    @classmethod
    def get_name(cls):
        return "DepthAnythingV3Large"

    @classmethod
    def has_checkpoint_file(cls, model_type="da3-large"):
        return path.exists(_find_da3_l_checkpoint(model_type))

    @classmethod
    def supported(cls, model_type):
        low = str(model_type).lower()
        return ("large" in low or "larg" in low or model_type in MODEL_FILES_L)

    @classmethod
    def get_model_path(cls, model_type="da3-large"):
        return _find_da3_l_checkpoint(model_type)

    def is_metric(self):
        return False

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return True

# 兼容旧名字: test.py / 别的 wrapper 可能 import 下面这些名字
DepthAnythingV3MonoModel = DepthAnythingV3LargeModel
Any_V3_Mono_Large = DepthAnythingV3LargeModel
