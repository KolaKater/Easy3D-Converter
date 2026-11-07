"""
GFPGAN_face - Easy3D edition (v3.6)
For Easy3D GUI: Face Restore AFTER SBS merge, BEFORE Output Type.

Changes vs StereoSpace version:
1. No hardcoded D:/3D/StereoSpace_Tony as primary path. Search order:
   Easy3D/gfpgan/weights -> Easy3D/models -> Easy3D/upscale_models -> StereoSpace (fallback only)
2. bg_upsampler OFF by default for SBS: we only need faces, background stays original.
   (Old: RealESRGAN x2 on 4096-wide SBS -> 8192 wide, 3070 8GB OOM / slow)
3. New restore_face_sbs(): split SBS Left|Right, restore each eye separately, re-join.
   Wide SBS (4096+) fed directly to GFPGAN misses faces and is slow. Per-eye is safer.
4. init_face_restorer() is idempotent: second call returns True without reloading.
5. Device: cuda if available, else cpu (old: hardcoded cuda).
"""
import pathlib
from PIL import Image
import numpy as np
import cv2

face_restorer = None
face_upsampler = None  # always None in Easy3D (no bg upsample)
_face_loaded = False
_last_error = ""  # GUI can read via get_last_error()

def _find_gfpgan_weights():
    here = pathlib.Path(__file__).parent
    candidates = [
        here / "gfpgan" / "weights" / "GFPGANv1.4.pth",
        here / "gfpgan" / "weights" / "GFPGANv1.3.pth",
        here / "models" / "GFPGANv1.4.pth",
        here / "models" / "GFPGANv1.3.pth",
        here / "upscale_models" / "GFPGANv1.4.pth",
        here / "GFPGANv1.4.pth",
        # No absolute paths: only relative to this file.
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# GFPGAN official releases (v1.3.0) - auto-download if not found locally
_GFPGAN_URLS = {
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "parsing_parsenet.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/parsing_parsenet.pth",
    "detection_Resnet50_Final.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/detection_Resnet50_Final.pth",
}

def _download_file(url, dest, desc=""):
    """Download with progress, no re-download if already exists. Returns True on success."""
    import urllib.request, ssl
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[FaceRestore] {desc or dest.name} already exists, skip download", flush=True)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FaceRestore] Downloading {desc or dest.name} ...", flush=True)
    print(f"[FaceRestore]   URL: {url}", flush=True)
    print(f"[FaceRestore]   To: {dest}", flush=True)
    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0) or 0)
            done = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk: break
                f.write(chunk); done += len(chunk)
                if total > 0:
                    print(f"[FaceRestore]   {desc or dest.name}: {done/1024/1024:.1f} / {total/1024/1024:.1f} MB ({done*100//total}%)", flush=True)
                else:
                    print(f"[FaceRestore]   {desc or dest.name}: {done/1024/1024:.1f} MB", flush=True)
        print(f"[FaceRestore] Downloaded OK: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"[FaceRestore] Download FAILED {dest.name}: {e}", flush=True)
        try:
            if dest.exists(): dest.unlink()
        except Exception: pass
        return False

def _ensure_gfpgan_weights():
    """Ensure all 3 GFPGAN files exist in Easy3D/gfpgan/weights, auto-download missing ones."""
    here_weights = pathlib.Path(__file__).parent / "gfpgan" / "weights"
    here_weights.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in _GFPGAN_URLS.items():
        dest = here_weights / name
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"[FaceRestore] {name} not found in {here_weights}, auto-downloading ...", flush=True)
            if not _download_file(url, dest, desc=name):
                ok = False
        else:
            print(f"[FaceRestore] {name} found ({dest.stat().st_size/1024/1024:.1f} MB), skip download", flush=True)
    return ok

def init_face_restorer(restore_type="none", fidelity=0.7):
    global face_restorer, face_upsampler, _face_loaded, _last_error
    _last_error = ""
    if restore_type == "none":
        print(f"[FaceRestore] None - skip")
        return True
    if _face_loaded and face_restorer is not None:
        print(f"[FaceRestore] Already loaded, reuse")
        return True
    try:
        from gfpgan import GFPGANer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FaceRestore] Loading GFPGAN {restore_type} fidelity={fidelity} device={device} (Easy3D: no bg upsampler, faces only)", flush=True)
        # v3.6: auto-download missing weights (GFPGANv1.4 + parsing + detection)
        _ensure_gfpgan_weights()
        model_path = _find_gfpgan_weights()
        # Log the two helper files explicitly - user has them in Easy3D/gfpgan/weights
        for _h in ("parsing_parsenet.pth", "detection_Resnet50_Final.pth"):
            _hp = pathlib.Path(__file__).parent / "gfpgan" / "weights" / _h
            print(f"[FaceRestore] Helper {_h}: {'FOUND ' + str(_hp) if _hp.exists() else 'MISSING - will download'}", flush=True)
        if not model_path:
            model_path = pathlib.Path(__file__).parent / "GFPGANv1.4.pth"
            print(f"[FaceRestore] No local weights found, will try download to {model_path}", flush=True)
        # Easy3D: GFPGAN lib looks for parsing/detection in its OWN site-packages/gfpgan/weights,
        # not in <Easy3D>/gfpgan/weights (relative to this file). If user already has them in Easy3D, copy them over
        # so GFPGAN does not re-download them.
        try:
            import gfpgan as _gfpgan_pkg, shutil
            _lib_weights = pathlib.Path(_gfpgan_pkg.__file__).parent / "weights"
            _lib_weights.mkdir(parents=True, exist_ok=True)
            for _name in ("parsing_parsenet.pth", "detection_Resnet50_Final.pth"):
                _src = pathlib.Path(__file__).parent / "gfpgan" / "weights" / _name
                _dst = _lib_weights / _name
                if _src.exists() and not _dst.exists():
                    shutil.copy2(str(_src), str(_dst))
                    print(f"[FaceRestore] Copied {_name} to GFPGAN lib weights (reuse, no download)", flush=True)
                elif _dst.exists():
                    print(f"[FaceRestore] {_name} already in GFPGAN lib weights", flush=True)
                elif _src.exists():
                    print(f"[FaceRestore] {_name} found in Easy3D/gfpgan/weights", flush=True)
        except Exception as _ce:
            print(f"[FaceRestore] Could not copy parsing/detection weights (will download if needed): {_ce}", flush=True)
        # Easy3D: bg_upsampler=None -> background untouched, only faces restored, saves VRAM/time on SBS
        face_upsampler = None
        face_restorer = GFPGANer(
            model_path=str(model_path) if model_path and model_path.exists() else "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _face_loaded = True
        print(f"[FaceRestore] GFPGAN loaded OK {model_path}", flush=True)
        return True
    except Exception as e:
        import traceback
        _last_error = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
        print(f"[FaceRestore] Init failed {e}, fallback no-op", flush=True)
        traceback.print_exc()
        face_restorer = None
        _face_loaded = False
        return False

def restore_face(pil_img, blend=1.0, feather=15):
    """
    V75.25 FACE FIX - center 100% opaque, edge 15px only feather, center not affected
    Single image (kept for compat). For SBS use restore_face_sbs() instead.
    """
    global face_restorer
    if face_restorer is None:
        return pil_img
    try:
        img_np = np.array(pil_img)[:, :, ::-1]
        _, _, restored = face_restorer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True, weight=blend)
        if restored is None:
            return pil_img
        restored_pil = Image.fromarray(restored[:, :, ::-1])
        if restored_pil.size != pil_img.size:
            restored_pil = restored_pil.resize(pil_img.size, Image.LANCZOS)
        try:
            orig_np = np.array(pil_img).astype(np.float32)
            rest_np = np.array(restored_pil).astype(np.float32)
            diff = np.abs(orig_np - rest_np).mean(axis=2)
            mask = (diff > 5).astype(np.uint8) * 255
            if mask.sum() < 50:
                if blend < 0.99:
                    blended = orig_np*(1-blend) + rest_np*blend
                    return Image.fromarray(np.clip(blended,0,255).astype(np.uint8))
                return restored_pil
            k_core = 8
            kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_core, k_core))
            core = cv2.erode(mask, kernel_core, iterations=1)
            edge = cv2.subtract(mask, core)
            edge_blur = cv2.GaussianBlur(edge, (15, 15), 0)
            mask_f = core.astype(np.float32) / 255.0 + edge_blur.astype(np.float32) / 255.0 * 0.9
            mask_f = np.clip(mask_f, 0, 1.0)
            mask_f = mask_f[:, :, None]
            if blend < 0.99:
                mask_f = mask_f * blend
            blended = orig_np * (1 - mask_f) + rest_np * mask_f
            print(f"[FaceRestore] V75.25 core {k_core}px opaque edge 15px feather only center not affected blend={blend}", flush=True)
            return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        except Exception as e:
            print(f"[FaceRestore] feather failed {e}", flush=True)
            import traceback; traceback.print_exc()
            if blend < 0.99:
                orig_np = np.array(pil_img).astype(np.float32)
                rest_np = np.array(restored_pil).astype(np.float32)
                blended = orig_np*(1-blend) + rest_np*blend
                return Image.fromarray(np.clip(blended,0,255).astype(np.uint8))
            return restored_pil
    except Exception as e:
        print(f"[FaceRestore] restore failed {e}", flush=True)
        return pil_img

def restore_face_sbs(sbs_img, blend=1.0, feather=15):
    """Easy3D: SBS (Left|Right) -> split, restore each eye, re-join. Both Eyes, same treatment."""
    if face_restorer is None:
        return sbs_img
    w, h = sbs_img.size
    if w < 2 or w % 2 != 0:
        # Not a clean SBS (odd width) - fall back to whole-image restore
        print(f"[FaceRestore] SBS width {w} not even, falling back to whole-image restore", flush=True)
        return restore_face(sbs_img, blend=blend, feather=feather)
    half = w // 2
    left = sbs_img.crop((0, 0, half, h))
    right = sbs_img.crop((half, 0, w, h))
    print(f"[FaceRestore] SBS split: Left {left.size} + Right {right.size} -> restore each eye", flush=True)
    left_r = restore_face(left, blend=blend, feather=feather)
    right_r = restore_face(right, blend=blend, feather=feather)
    out = Image.new("RGB", (w, h))
    out.paste(left_r, (0, 0))
    out.paste(right_r, (half, 0))
    return out


def get_last_error():
    """GUI: why did init fail? Empty if none."""
    return _last_error
