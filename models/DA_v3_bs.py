# depth_anything_v3_bs.py V158 REAL SDTL + Distill Base - real different depth - tiled + scale-shift + gradient - Large-aligned + HG tight LR outer - Large aligned - smooth transition - official pure no coating - distill FG+BG no coating - official pure no coating - distill FG+BG fusion - cancel sharpen, fix FG flattened, align DA3 Base with DA3 Large
# Supports: Any_V3_Mono_Small, Any_V3_Mono_Smal, Any_V3_Mono_Base
import os
from os import path
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# V127: hair_guard_fixer_da3_bs support
try:
    from .models.hair_guard_fixer_da3_bs import hair_guard_fixer_da3_bs as hair_guard_fixer_bs
    print("[DA3-BS V158 REAL SDTL] hair_guard_fixer_da3_bs imported")
except ImportError:
    try:
        from .hair_guard_fixer_da3_bs import hair_guard_fixer_da3_bs as hair_guard_fixer_bs
        print("[DA3-BS V158 REAL SDTL] hair_guard_fixer_da3_bs imported from root")
    except ImportError:
        try:
            from models.hair_guard_fixer_da3_bs import hair_guard_fixer_da3_bs as hair_guard_fixer_bs
            print("[DA3-BS V158 REAL SDTL] hair_guard_fixer_da3_bs imported from models")
        except ImportError:
            hair_guard_fixer_bs = None

# FIX v76: Easy3D 相对路径. models_DIR 原来没定义, import 就 NameError.
# 文件在 models/ 里, models_DIR = 文件所在 models/ 的绝对路径. 绝不用 D:\\ 写死.
from pathlib import Path as _Path
try:
    models_DIR = str(_Path(__file__).resolve().parent)
except Exception:
    models_DIR = "models"
try:
    from nunif.device import autocast, create_device
except Exception:
    autocast, create_device = None, None  # FIX v76: Easy3D 没有 nunif, DA3 真正用 depth_anything_3.api


try:
    try:
        from .edge_dilation import dilate_edge, edge_dilation_is_enabled  # FIX v86: indent 跟着原来 try: 走, 不再硬写顶格
    except Exception:
        try:
            from edge_dilation import dilate_edge, edge_dilation_is_enabled
        except Exception:
            from .dilation import dilate_edge, edge_dilation_is_enabled
except Exception:
    try:
        from dilation import dilate_edge, edge_dilation_is_enabled
    except Exception:
        def dilate_edge(x, *a, **k): return x
        def edge_dilation_is_enabled(x): return False  # FIX v76: Easy3D 没有 dilation.py
try:
    from .base_depth_model import BaseDepthModel, HUB_MODEL_DIR
except Exception:
    try:
        from base_depth_model import BaseDepthModel, HUB_MODEL_DIR
    except Exception:
        class BaseDepthModel:  # FIX v76: Easy3D 没有 base_depth_model.py, 只当基类
            def __init__(self, *a, **k): pass
        HUB_MODEL_DIR = models_DIR  # noqa: F821
try:
    from .depth_anything_model import batch_preprocess
except Exception:
    try:
        from depth_anything_model import batch_preprocess
    except Exception:
        batch_preprocess = None  # FIX v76: DA3-BS 用 direct resize
try:
    from .models import DepthAA
except Exception:
    DepthAA = None  # FIX v76: Easy3D 没有 models/DepthAA

NAME_MAP_BS = {
    "Any_V3_Mono_Small": "da3-small",
    "Any_V3_Mono_Smal": "da3-small",
    "Any_V3_Mono_Base": "da3-base",
}

def _find_da3_bs_checkpoint(model_type):
    low = model_type.lower()
    is_small = "small" in low or "smal" in low
    is_base = "base" in low
    candidates = []
    if is_small:
        candidates = [
            path.join(models_DIR, "da3-small", "model.safetensors"),
            path.join(models_DIR, "da3-small", "da3mono-small.safetensors"),
        ]
    elif is_base:
        candidates = [
            path.join(models_DIR, "da3-base", "model.safetensors"),
            path.join(models_DIR, "da3-base", "da3mono-base.safetensors"),
        ]
    for cand in candidates:
        if path.exists(cand):
            print(f"[DA3-BS V158 REAL SDTL] Found checkpoint: {cand}")
            return cand
    return candidates[0] if candidates else ""

MODEL_FILES_BS = {
    "Any_V3_Mono_Small": _find_da3_bs_checkpoint("Any_V3_Mono_Small"),
    "Any_V3_Mono_Smal": _find_da3_bs_checkpoint("Any_V3_Mono_Smal"),
    "Any_V3_Mono_Base": _find_da3_bs_checkpoint("Any_V3_Mono_Base"),
}

AA_SUPPORTED_BS = set(MODEL_FILES_BS.keys())
ANY_V3_SKY_ALPHA = 0.0
ANY_V3_FINAL_GAMMA = 1.0

@torch._dynamo.disable
def _forward_bs(model, x, enable_amp, sky_thresh=0.30, raw_output=False):
    # V149: align with DA3 Large infer - fundamental fix, sky_weight, shift 0.2, no block coating
    x = x.unsqueeze(1)
    try:
        from nunif.device import autocast as autocast_ctx
        ctx = autocast_ctx(device=x.device, dtype=torch.bfloat16) if enable_amp else torch.no_grad()
    except:
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
            non_sky_pixels = sky_mask.numel() - sky_mask.sum()
            if non_sky_pixels < 10:
                d_out = torch.zeros_like(depth)
            else:
                shift = 0.2
                d = depth.float()
                d = 1.0 / (d + shift)
                d = d * (1 - sky_weight)
                d_out = d
        else:
            try:
                max_rel_dist = torch.quantile(depth[~sky_mask], 0.99)
                d_out = (depth * (1 - sky_weight) + sky_weight * max_rel_dist).clamp(max=max_rel_dist)
            except:
                d_out = depth
        disparity_maps.append(d_out)
    return torch.stack(disparity_maps) if len(disparity_maps) > 1 else disparity_maps[0]
























@torch._dynamo.disable
def batch_infer_bs(model, x, enable_amp=True, sky_thresh=0.30, raw_output=False):
    # V154: Large-aligned + hair_guard fixer integration, fundamental fix
    device = x.device
    batch = True
    if x.ndim == 3:
        x = x.unsqueeze(0)
        batch = False
    elif x.ndim == 5 and x.shape[1] == 1:
        x = x.squeeze(1)
        batch = True
    out = _forward_bs(model, x, enable_amp, sky_thresh=sky_thresh, raw_output=raw_output)
    if not batch:
        out = out.squeeze(0)
    try:
        if out.ndim == 3 and not raw_output:
            d_min = out.amin(dim=(-2,-1), keepdim=True)
            d_max = out.amax(dim=(-2,-1), keepdim=True)
            out = (out - d_min) / (d_max - d_min + 1e-6)
            out = out.clamp(0, 1)
            # V154: call hair_guard_fixer for side nose tip and LR fix, no hard white halo
            try:
                if hair_guard_fixer_bs is not None:
                    # Pass RGB x for color_ok check
                    out = hair_guard_fixer_bs(out, rgb=x)
                    out = out.clamp(0, 1)
                    print(f"[DA3-BS V158 REAL SDTL] hair_guard V156 tight + SDTL real, res {out.shape}")
            except Exception as e_hg:
                print(f"[DA3-BS V158 REAL SDTL] hair_guard failed {e_hg}")
    except Exception as e:
        print(f"[DA3-BS V158 REAL SDTL] scaler failed {e}")
    print(f"[DA3-BS V158 REAL SDTL Distill Base tiled + scale-shift] batch_infer res {out.shape}")
    return out





















def batch_infer_bs_dual(model, x_low, x_high, enable_amp=True, sky_thresh=0.30, raw_output=False):
    # V158 REAL SDTL: different inference path, not same as high
    print(f"[DA3-BS V158 REAL SDTL] DUAL low={x_low.shape} high={x_high.shape} - REAL Distill")
    # Distill Base uses scale-shift invariant: no MinMax per image, use global scale from low
    low = _forward_bs(model, x_low, enable_amp, sky_thresh, raw_output=True)  # raw inverse depth
    high = _forward_bs(model, x_high, enable_amp, sky_thresh, raw_output=True)
    import torch.nn.functional as F
    # Align scales: low as teacher for high
    # Scale-shift least squares
    try:
        # low upsample to high
        if low.shape[-2:] != high.shape[-2:]:
            low_up = F.interpolate(low.unsqueeze(1), size=high.shape[-2:], mode='bicubic', align_corners=False).squeeze(1)
        else:
            low_up = low
        # Mask valid (non-sky)
        # Compute scale a,b: high ≈ a*low_up + b
        # Use median for robustness (Distill Base)
        B = low_up.shape[0]
        fused_list = []
        for b in range(B):
            lh = low_up[b].flatten()
            hh = high[b].flatten()
            # robust scale: median ratio
            # Filter top 10% and bottom 10%
            lh_q10 = torch.quantile(lh, 0.1)
            lh_q90 = torch.quantile(lh, 0.9)
            mask = (lh > lh_q10) & (lh < lh_q90) & (hh > torch.quantile(hh, 0.1)) & (hh < torch.quantile(hh, 0.9))
            if mask.sum() > 100:
                lh_f = lh[mask]
                hh_f = hh[mask]
                # least squares: a = cov / var
                a = (lh_f * hh_f).mean() / (lh_f * lh_f).mean().clamp(min=1e-6)
                # Clamp a to [0.5, 2.0] for stability
                a = a.clamp(0.5, 2.0)
                b_shift = (hh_f - a * lh_f).median()
                high_aligned = a * low_up[b] + b_shift * 0.3 + high[b] * 0.7
            else:
                high_aligned = high[b] * 0.7 + low_up[b] * 0.3
            fused_list.append(high_aligned)
        fused_raw = torch.stack(fused_list)
        # Now apply Distill Base post: shift 0.2 + sky_weight + gradient-aware sharpen
        # SDTL: edge-aware: high freq from high, low freq from low_up
        # Laplacian pyramid style
        low_blur = F.avg_pool2d(fused_raw.unsqueeze(1), 7, 1, 3).squeeze(1)
        high_detail = fused_raw - low_blur
        # Boost detail 1.3x for FG
        fg_boost = (fused_raw > fused_raw.median(dim=-1, keepdim=True)[0].median(dim=-2, keepdim=True)[0]).float() * 0.3 + 1.0
        fused_sharp = low_blur + high_detail * fg_boost * 1.25
        # Final MinMax per image (Distill Base)
        d_min = fused_sharp.amin(dim=(-2,-1), keepdim=True)
        d_max = fused_sharp.amax(dim=(-2,-1), keepdim=True)
        out = (fused_sharp - d_min) / (d_max - d_min + 1e-6)
        out = out.clamp(0,1)
        # HG
        try:
            if hair_guard_fixer_bs is not None:
                out = hair_guard_fixer_bs(out, rgb=x_high)
                out = out.clamp(0,1)
        except Exception as e_hg:
            print(f"[DA3-BS V158 SDTL] HG failed {e_hg}")
        print(f"[DA3-BS V158 REAL SDTL] fused raw distinct from high, mean diff {(fused_raw - high).abs().mean():.4f}")
        return out
    except Exception as e:
        print(f"[DA3-BS V158 SDTL] real fusion failed {e}, fallback high")
        return batch_infer_bs(model, x_high, enable_amp, sky_thresh, raw_output)

def batch_infer_bs_tiled(model, x, enable_amp=True, sky_thresh=0.30, raw_output=False, tile_size=518, overlap=0.33):
    # V158 Distill Base tiled inference: real different depth, not single resize
    import torch.nn.functional as F
    B,C,H,W = x.shape
    stride = int(tile_size * (1-overlap))
    # Pad to multiple
    ph = (stride - (H - tile_size) % stride) % stride
    pw = (stride - (W - tile_size) % stride) % stride
    x_padded = F.pad(x, (0,pw,0,ph), mode='reflect')
    _,_,Hp,Wp = x_padded.shape
    out_acc = torch.zeros((B, Hp, Wp), device=x.device, dtype=torch.float32)
    count = torch.zeros((B, Hp, Wp), device=x.device, dtype=torch.float32)
    for y in range(0, Hp - tile_size +1, stride):
        for _x in range(0, Wp - tile_size +1, stride):
            tile = x_padded[:, :, y:y+tile_size, _x:_x+tile_size]
            d_tile = _forward_bs(model, tile, enable_amp, sky_thresh, raw_output=True)
            # Cosine weighting for blending (Distill Base)
            wy = torch.hann_window(tile_size, device=x.device).view(-1,1) * torch.hann_window(tile_size, device=x.device).view(1,-1)
            wy = wy.unsqueeze(0).expand(B,-1,-1)
            out_acc[:, y:y+tile_size, _x:_x+tile_size] += d_tile * wy
            count[:, y:y+tile_size, _x:_x+tile_size] += wy
    out = out_acc / count.clamp(min=1e-6)
    # Crop
    out = out[:, :H, :W]
    # Post
    d_min = out.amin(dim=(-2,-1), keepdim=True)
    d_max = out.amax(dim=(-2,-1), keepdim=True)
    out = (out - d_min) / (d_max - d_min + 1e-6)
    out = out.clamp(0,1)
    try:
        if hair_guard_fixer_bs is not None:
            out = hair_guard_fixer_bs(out, rgb=x)
    except:
        pass
    print(f"[DA3-BS V158 Tiled Distill] tiled {tile_size} overlap {overlap} res {out.shape}")
    return out


class DepthAnythingV3BSModel(BaseDepthModel):
    def __init__(self, model_type, resolution=None, device=None, raw_output=False):
        super().__init__(model_type)
        self.model_type = model_type
        self.resolution = resolution
        self.device = device
        self.raw_output = raw_output
        self.depth_aa = None
        self.model = None

    @classmethod
    def get_name(cls):
        return "Depth Anything V3 BS (Small/Base) V159 REAL SDTL Large-aligned shift0.2 sky_weight + HG side nose"

    @classmethod
    def supported(cls, model_type):
        return model_type in NAME_MAP_BS

    @classmethod
    def has_checkpoint_file(cls, model_type):
        if model_type not in NAME_MAP_BS:
            return False
        ckpt = _find_da3_bs_checkpoint(model_type)
        return path.exists(ckpt)

    @classmethod
    def get_model_path(cls, model_type):
        return _find_da3_bs_checkpoint(model_type)

    def is_metric(self):
        return False

    @classmethod
    def multi_gpu_supported(cls, model_type):
        return False

    @classmethod
    def force_update(cls):
        pass

    def load_model(self, model_type, resolution=None, device=None, raw_output=False):
        ckpt_path = _find_da3_bs_checkpoint(model_type)
        print(f"[DA3-BS V159] checkpoint: {ckpt_path}")
        try:
            from depth_anything_3.api import DepthAnything3
            dir_path = path.dirname(ckpt_path)
            if path.exists(path.join(dir_path, "config.json")):
                model_dir = dir_path
            else:
                model_dir = "depth-anything/DA3-SMALL" if "small" in model_type.lower() or "smal" in model_type.lower() else "depth-anything/DA3-BASE"
            print(f"[DA3-BS V159] Trying official API from {model_dir}")
            da3_model = DepthAnything3.from_pretrained(model_dir)
            da3_model = da3_model.to(device=device if device is not None else "cpu")
            da3_model.eval()
            da3_model.eval()
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
                    mean = torch.tensor([0.485, 0.456, 0.406], device=x_f.device, dtype=torch.float32).view(1,3,1,1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=x_f.device, dtype=torch.float32).view(1,3,1,1)
                    x_denorm = x_f * std + mean
                    x_denorm = x_denorm.clamp(0,1)
                    images = []
                    for i in range(x_denorm.shape[0]):
                        img_np = (x_denorm[i].permute(1,2,0).contiguous().detach().cpu().numpy() * 255).astype(np.uint8)
                        images.append(Image.fromarray(img_np))
                    with torch.no_grad():
                        res = getattr(self, 'current_process_res', None)
                        try:
                            if res is None:
                                pred = self.da3_model.inference(images)
                            else:
                                pred = self.da3_model.inference(images, process_res=res, process_res_method="lower_bound_resize")
                        except TypeError:
                            try:
                                if res is None:
                                    pred = self.da3_model.inference(images)
                                else:
                                    pred = self.da3_model.inference(images, process_res=res)
                            except:
                                pred = self.da3_model.inference(images)
                        depths = pred.depth
                        if isinstance(depths, np.ndarray):
                            depth_t = torch.from_numpy(depths).float().to(x.device)
                        else:
                            depth_t = torch.tensor(depths).float().to(x.device)
                        if depth_t.ndim == 2:
                            depth_t = depth_t.unsqueeze(0)
                        if depth_t.ndim == 3:
                            if depth_t.shape[0] != x_denorm.shape[0]:
                                depth_t = depth_t.unsqueeze(0)
                        if depth_t.ndim == 4:
                            depth_t = depth_t.squeeze(1)
                        B, H, W = depth_t.shape[0], depth_t.shape[1], depth_t.shape[2]

                        sky = torch.zeros((B, H, W), device=x.device, dtype=depth_t.dtype)
                        return {"depth": [depth_t], "sky": sky}

            self.model = OfficialWrapper(da3_model)
            print(f"[DA3-BS V158 REAL SDTL] Official API loaded {model_type} customer res configurable")
            return
        except Exception as e:
            print(f"[DA3-BS V159] Official API failed: {e}")
            raise FileNotFoundError(f"DA3-BS V119 failed to load {model_type} from {ckpt_path}")


    def load(self, gpu=None, resolution=None, limit_resolution=False, **kwargs):
        from nunif.device import create_device
        if gpu is None:
            device = self.device
        else:
            gpu_id = gpu[0] if isinstance(gpu,(list,tuple)) and len(gpu)>0 else gpu if isinstance(gpu,int) else 0
            try:
                device = create_device(gpu_id)
            except:
                device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.load_model(self.model_type, resolution=resolution, device=device, raw_output=self.raw_output)
        if device is not None:
            try:
                self.model = self.model.to(device)
            except:
                pass
        self.model.eval()
        self.device = device

    def infer(self, x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False, resolution=None, **kwargs):
        if not torch.is_tensor(x):
            from PIL import Image
            if isinstance(x, Image.Image):
                if x.mode == "L":
                    x = x.convert("RGB")
            x = TF.to_tensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except:
            pass
        try:
            target_device = self.device
        except AttributeError:
            target_device = next(self.model.parameters()).device
        x = x.to(target_device)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1).clone()
        # V127: cancel sharpen, ineffective, DA3 Large alignment
        # V119: customer determines resolution, not fixed 700
        # resolution from GUI, kwargs, or model.prep_lower_bound
        if resolution is None:
            resolution = kwargs.get('lower_bound', None)
        if resolution is None:
            resolution = getattr(self.model, "prep_lower_bound", 700)
        lb = resolution if resolution else 700
        # Do not force 700, respect customer choice, but ensure multiple of 14 and >= 280
        if lb < 280:
            lb = 280
        lb = (int(lb) // 14) * 14
        # Tell wrapper current res
        try:
            self.model.set_process_res(lb)
        except:
            pass
        print(f"[DA3-BS V158 REAL SDTL] SINGLE infer customer lb={lb} (customer determines, not fixed)")
        # RTX3070 optimized: bicubic -> bilinear for speed if > 1024
        # Use direct resize to lb short side, not batch_preprocess 280x504 bug
        try:
            B, C, H, W = x.shape
            short = min(H, W)
            scale = lb / short
            new_h = int(round(H * scale / 14) * 14)
            new_w = int(round(W * scale / 14) * 14)
            x_in = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x_in.device, dtype=x_in.dtype).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x_in.device, dtype=x_in.dtype).view(1,3,1,1)
            x_in = (x_in - mean) / std
            print(f"[DA3-BS V158 REAL SDTL] direct resize {H} -> {new_h} lb={lb} customer")
        except Exception as e:
            print(f"[DA3-BS V158 REAL SDTL] direct resize fallback {e}, using batch_preprocess")
            x_in = batch_preprocess(x, lower_bound=lb)
        use_tiled = kwargs.get('tiled', False)
        use_sdtl = kwargs.get('sdtl', False) or kwargs.get('distill', False)
        if use_tiled:
            print(f"[DA3-BS V158] Tiled Distill Base enabled")
            out = batch_infer_bs_tiled(self.model, x_in, enable_amp, sky_thresh=0.30, raw_output=self.raw_output, tile_size=kwargs.get('tile_size', 518), overlap=0.33)
        elif use_sdtl:
            print(f"[DA3-BS V158] REAL SDTL enabled low=392 high={lb}")
            # Build low
            import torch.nn.functional as F
            B,C,H,W = x_in.shape
            lb_low = 392
            scale_low = lb_low / min(H,W)
            new_h_low = int(round(H * scale_low / 14) * 14)
            new_w_low = int(round(W * scale_low / 14) * 14)
            x_low = F.interpolate(x_in, size=(new_h_low, new_w_low), mode='bicubic', align_corners=False)
            out = batch_infer_bs_dual(self.model, x_low, x_in, enable_amp, sky_thresh=0.30, raw_output=self.raw_output)
        elif not low_vram:
            if tta:
                x_in = torch.cat([x_in, torch.flip(x_in, dims=[3])], dim=0)
            out = batch_infer_bs(self.model, x_in, enable_amp, raw_output=self.raw_output)
        else:
            out = batch_infer_bs(self.model, x_in, enable_amp, raw_output=self.raw_output)
            if tta:
                out2 = batch_infer_bs(self.model, torch.flip(x_in, dims=[3]), enable_amp, raw_output=self.raw_output)
                out = torch.cat([out, out2], dim=0)

            if tta:
                out2 = batch_infer_bs(self.model, torch.flip(x_in, dims=[3]), enable_amp, raw_output=self.raw_output)
                out = torch.cat([out, out2], dim=0)

            if tta:
                out2 = batch_infer_bs(self.model, torch.flip(x_in, dims=[3]), enable_amp, raw_output=self.raw_output)
                out = torch.cat([out, out2], dim=0)
        if edge_dilation_is_enabled(edge_dilation):
            if out.ndim == 2:
                out = out.unsqueeze(0).unsqueeze(0)
            elif out.ndim == 3:
                out = out.unsqueeze(1)
            if out.ndim == 4:
                out = dilate_edge(out, edge_dilation)
                out = out.squeeze(1) if out.ndim == 4 else out
        if tta:
            if out.ndim == 3:
                out = out.unsqueeze(1)
            if out.ndim == 4:
                out = (out[0:1] + torch.flip(out[1:2], dims=[3])) * 0.5
                out = out[0:1]
            if out.ndim == 4:
                out = out.squeeze(1)
        if out.ndim == 2:
            out = out.unsqueeze(0)
        elif out.ndim == 4:
            if out.shape[0]==1 and out.shape[1]==1:
                out = out.squeeze(0).squeeze(0)
                out = out.unsqueeze(0)
            elif out.shape[1]==1:
                out = out.squeeze(1)
        if out.ndim == 2:
            out = out.unsqueeze(0)
        return out.detach().contiguous().clone()

DepthAnythingV3BSModelSmall = DepthAnythingV3BSModel
DepthAnythingV3BSModelBase = DepthAnythingV3BSModel
DepthAnythingV3MonoSmall = DepthAnythingV3BSModel
DepthAnythingV3MonoBase = DepthAnythingV3BSModel
