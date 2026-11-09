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
_mde_model_env = os.environ.get("GENSTEREO_MDE_MODEL", "depth_anything_v2_vitl").strip().lower()
_mde_path_env = os.environ.get("GENSTEREO_MDE_PATH", "models/depth_anything_v2_vitl.pth")
IS_FLASHDEPTH = _mde_model_env in ("flashdepth-l", "flashdepth-s", "flashdepth")
# Map GUI lowercase -> flashdepth_model.py model_type ("FlashDepth-L" / "FlashDepth-S")
FLASHDEPTH_TYPE = {"flashdepth-l": "FlashDepth-L", "flashdepth-s": "FlashDepth-S", "flashdepth": "FlashDepth"}.get(_mde_model_env, "FlashDepth-S")
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
fusion_model.load_state_dict(torch.load(fusion_checkpoint))
fusion_model = fusion_model.to(DEVICE).eval()


def infer_depth_flashdepth(image_path: str, image_size: int):
    # FlashDepth: input is PIL RGB, output -> [1,1,H,W] float on DEVICE (same shape as DAM2 path)
    # flashdepth_model.py infer() already does stretch/preprocess + clean_shallow_dots internally.
    image = Image.open(image_path).convert('RGB')
    _ow, _oh = image.size
    if (_ow, _oh) != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        print(f"[Easy3D] Stretch (no crop): {_ow}x{_oh} -> {image_size}x{image_size} square")
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
    # Unified entry: dispatches by GENSTEREO_MDE_MODEL
    if IS_FLASHDEPTH:
        return infer_depth_flashdepth(image_path, image_size)
    return infer_depth_dam2(image_path, image_size)

def infer_depth_dam2(image_path: str, image_size: int):
    # v3.1 STRETCH ONLY (SD3D-style): NEVER crop, NEVER letterbox, NEVER patch.
    image = Image.open(image_path).convert('RGB')
    _ow, _oh = image.size
    if (_ow, _oh) != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        print(f"[Easy3D] Stretch (no crop): {_ow}x{_oh} -> {image_size}x{image_size} square")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    depth_dam2 = dam2.infer_image(image_bgr)
    return image, torch.tensor(depth_dam2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

def normalize_disp(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())

def morphological_opening(mask_tensor, kernel_size=7):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.tensor(cleaned_mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

def generate_novel_view(image, depth, output_dir, basename, scale_factor=0.15, output_name='generated_right.png'):
    # v3.3: scale sign = direction. + = generate RIGHT, - = generate LEFT.
    disparity = normalize_disp(depth) * scale_factor * IMAGE_SIZE
    output_path = join(output_dir, basename)
    os.makedirs(output_path, exist_ok=True)
    renders = genstereo_nvs(src_image=image, src_disparity=disparity, ratio=None)
    warped = (renders['warped'] + 1) / 2
    mask = morphological_opening(renders['mask'])
    with torch.no_grad():
        fusion_image = fusion_model(renders['synthesized'].float(), warped.float(), mask.float())
    warped_pil = to_pil_image(warped[0])
    fusion_image_pil = to_pil_image(fusion_image[0])
    disp_vis = cv2.applyColorMap((normalize_disp(depth.squeeze().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(join(output_path, 'disp.png'), disp_vis)
    image.save(join(output_path, 'left.png'))
    warped_pil.save(join(output_path, 'warped.png'))
    fusion_image_pil.save(join(output_path, output_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate novel view from input image")
    parser.add_argument("image_path", nargs="+", help="v3.2: MULTIPLE files = load model ONCE, loop all (lossless). GUI passes all _tmp_square files in one call.")
    parser.add_argument("--output", default="./vis", help="Output directory (vis/<stem>/{left,generated_right,warped,disp}.png)")
    parser.add_argument("--scale_factor", type=float, default=0.15, help="Disparity scaling. GUI: 0.15 * baseline * depth_strength")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (GUI Steps)")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale (GUI Guidance)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed, -1 = random")
    parser.add_argument("--gen_mode", default="orig_right", choices=["orig_right", "orig_left", "both"],
                        help="v3.3: orig_right=Left=orig, Right=AI (+scale). orig_left=Left=AI (-scale), Right=orig. both=Left=AI (-scale/2), Right=AI (+scale/2), orig is center.")
    parser.add_argument("--image_size", type=int, default=0, help="STRETCH target square side. Input stretched to image_size x image_size, UNet runs at that size. GUI stretches back to original aspect.")
    args = parser.parse_args()
    # Seed: must be set BEFORE any torch.randn in GenStereo (latents are random)
    _seed = args.seed if args.seed is not None and args.seed >= 0 else random.randint(0, 2147483647)
    print(f"[Easy3D] seed={_seed} steps={args.steps} guidance={args.guidance} scale_factor={args.scale_factor} image_size={args.image_size or IMAGE_SIZE}")
    print(f"[Easy3D] MDE={_mde_model_env} path={_mde_path_env}")
    random.seed(_seed); np.random.seed(_seed % (2**32)); torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    # Steps / guidance: GenStereo Config holds them; scheduler was already built with defaults,
    # so update both cfg and scheduler if the attributes exist.
    try:
        genstereo_nvs.cfg.num_inference_steps = int(args.steps)
        genstereo_nvs.cfg.guidance_scale = float(args.guidance)
        if hasattr(genstereo_nvs, 'scheduler'):
            genstereo_nvs.scheduler.set_timesteps(int(args.steps), device=genstereo_nvs.device)
    except Exception as _e:
        print(f"[Easy3D] Could not apply steps/guidance to scheduler (will use defaults): {_e}")
    if args.image_size and args.image_size > 0:
        IMAGE_SIZE = int(args.image_size)
    # v3.1: UNet runs AT image_size (stretch target), not fixed 512/768
    try:
        genstereo_nvs.cfg.height = int(IMAGE_SIZE); genstereo_nvs.cfg.width = int(IMAGE_SIZE)
        print(f"[Easy3D] GenStereo running at {IMAGE_SIZE}x{IMAGE_SIZE} (stretch target, SD3D-style)")
    except Exception as _e:
        print(f"[Easy3D] Could not override GenStereo height/width: {_e}")
    _paths = args.image_path if isinstance(args.image_path, list) else [args.image_path]
    print(f"[Easy3D] Batch: {len(_paths)} image(s), model loaded ONCE, gen_mode={args.gen_mode}")
    for _i, _ip in enumerate(_paths, 1):
        base_name = splitext(basename(_ip))[0]
        print(f"[Easy3D] [{_i}/{len(_paths)}] Processing {_ip} ... gen_mode={args.gen_mode} scale={args.scale_factor}")
        img, depth = infer_depth(_ip, IMAGE_SIZE)
        if args.gen_mode == "both":
            generate_novel_view(img, depth, args.output, base_name, args.scale_factor * 0.5, output_name='generated_right.png')
            generate_novel_view(img, depth, args.output, base_name, args.scale_factor * -0.5, output_name='generated_left.png')
        elif args.gen_mode == "orig_left":
            generate_novel_view(img, depth, args.output, base_name, args.scale_factor * -1.0, output_name='generated_left.png')
        else:
            generate_novel_view(img, depth, args.output, base_name, args.scale_factor, output_name='generated_right.png')
