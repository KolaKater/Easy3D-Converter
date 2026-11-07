"""
output_type.py - Output Type separated from GUI and inference
V75.30 - Extracted from SD3D_GUI-V74

Handles:
- Parallel (Default) - Left|Right correct for VR
- Cross-Eyed (XD) - Right|Left swapped
- Auto-Align Parallel / Cross-Eyed / Both

Usage in GUI:
    from output_type import OUTPUT_TYPE_OPTIONS, process_sbs_file

Usage in inference_bat_final.py (optional):
    from output_type import get_output_type_suffix, create_sbs_image
"""

from pathlib import Path
import shutil
import subprocess
import sys
from PIL import Image

OUTPUT_TYPE_OPTIONS = {
    "1. Parallel (Default)": "parallel",
    "2. Cross-Eyed (XD Swap)": "xd",
    "3. Auto-Align Parallel": "align_parallel",
    "4. Auto-Align Cross-Eyed (XD)": "align_xd",
    "5. Auto-Align Parallel+XD (Both - 2 Files)": "align_both"
}

def make_sbs(left_img, right_img):
    """Create side-by-side image from left and right"""
    out = Image.new("RGB", (left_img.width + right_img.width, left_img.height))
    out.paste(left_img, (0, 0))
    out.paste(right_img, (left_img.width, 0))
    return out

def get_output_type_suffix(output_type):
    """Get suffix for output type"""
    if "both" in output_type:
        return "_Align and _Align_XD"
    elif "xd" in output_type and "align" in output_type:
        return "_Align_XD"
    elif "align" in output_type:
        return "_Align"
    elif "xd" in output_type:
        return "_XD"
    else:
        return ""

def process_sbs_file(sbs_path, output_type, log_func=None, align_exe_path=None):
    """
    Process a single SBS file to desired output type
    - sbs_path: Path to SBS jpg
    - output_type: one of OUTPUT_TYPE_OPTIONS values (parallel, xd, align_parallel, align_xd, align_both)
    - log_func: optional log function
    - align_exe_path: optional path to StereoAutoAlign.exe, auto-detected if None
    Returns: Path or list of Paths for align_both
    """
    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print(msg, flush=True)

    try:
        sbs_path = Path(sbs_path)
        if not sbs_path.exists():
            log(f"[OutputType] File not found: {sbs_path}")
            return None

        # Parse base name and suffix
        # Example: IMG_..._R_GA1.60_GB0.65_TEXT1.0.3D_SBS_B0.15_S123.jpg
        # base_name = part before 3D_SBS
        name = sbs_path.stem
        # Find suffix after 3D_SBS
        if "3D_SBS" in name:
            # Keep everything after 3D_SBS as suffix for output
            idx = name.find("3D_SBS")
            base_name = name[:idx].rstrip("_")
            suffix = name[idx+len("3D_SBS"):]
        else:
            base_name = name
            suffix = ""

        # Open SBS and split left/right
        sbs_img = Image.open(sbs_path)
        w, h = sbs_img.size
        mid = w // 2
        left = sbs_img.crop((0, 0, mid, h))
        right = sbs_img.crop((mid, 0, w, h))

        needs_align = "align" in output_type
        is_xd = "xd" in output_type
        is_both = "both" in output_type

        # Auto-Align with StereoAutoAlign.exe if needed
        if needs_align:
            if align_exe_path is None:
                # Try to find exe in current dir and tools
                candidates = [
                    Path.cwd() / "StereoAutoAlign.exe",
                    Path.cwd() / "tools" / "StereoAutoAlign.exe",
                    Path(__file__).parent / "StereoAutoAlign.exe",
                    Path(__file__).parent / "tools" / "StereoAutoAlign.exe",
                    # User reported path: ffmpeg\bin\stereoautoalign030\StereoAutoAlign.exe
                    Path(__file__).parent / "ffmpeg" / "bin" / "stereoautoalign030" / "StereoAutoAlign.exe",
                    Path.cwd() / "ffmpeg" / "bin" / "stereoautoalign030" / "StereoAutoAlign.exe",
                    Path(__file__).parent.parent / "ffmpeg" / "bin" / "stereoautoalign030" / "StereoAutoAlign.exe",
                ]
                for cand in candidates:
                    if cand.exists():
                        align_exe_path = cand
                        break

            if align_exe_path and Path(align_exe_path).exists():
                align_exe = Path(align_exe_path)
                log(f"[Auto-Align] Using {align_exe}")
                try:
                    temp_dir = sbs_path.parent / f"_temp_align_{sbs_path.stem}"
                    temp_dir.mkdir(exist_ok=True)
                    left_temp = temp_dir / f"{sbs_path.stem}_L_raw.jpg"
                    right_temp = temp_dir / f"{sbs_path.stem}_R_raw.jpg"
                    left.save(left_temp)
                    right.save(right_temp)

                    cmd = [str(align_exe), str(left_temp), str(right_temp), "10"]
                    si = None
                    creationflags = 0
                    if sys.platform == "win32":
                        try:
                            si = subprocess.STARTUPINFO()
                            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                            si.wShowWindow = subprocess.SW_HIDE
                            creationflags = subprocess.CREATE_NO_WINDOW
                        except:
                            si = None
                            creationflags = 0
                    subprocess.run(cmd, cwd=str(temp_dir), timeout=30,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   startupinfo=si, creationflags=creationflags)

                    aligned_l = None
                    aligned_r = None
                    for f in temp_dir.glob("*Align_l.*"):
                        aligned_l = f
                        break
                    for f in temp_dir.glob("*Align_r.*"):
                        aligned_r = f
                        break
                    # Fallback search in parent
                    if not aligned_l:
                        for f in sbs_path.parent.glob("*_L_raw_Align_l.*"):
                            aligned_l = f
                            break
                    if not aligned_r:
                        for f in sbs_path.parent.glob("*_L_raw_Align_r.*"):
                            aligned_r = f
                            break

                    if aligned_l and aligned_r:
                        left_aligned = Image.open(aligned_l)
                        right_aligned = Image.open(aligned_r)
                        lw, lh = left_aligned.size
                        left_cropped = left_aligned.crop((1, 0, lw-1, lh))
                        rw, rh = right_aligned.size
                        right_cropped = right_aligned.crop((1, 0, rw-1, rh))
                        left = left_cropped
                        right = right_cropped
                        if left.size != right.size:
                            right = right.resize(left.size, Image.LANCZOS)
                        log(f"[Auto-Align] Aligned success")
                    else:
                        log(f"[Auto-Align] WARNING: Aligned files not found, using original")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    for f in Path.cwd().glob("*_L_raw_Align_*"):
                        try:
                            f.unlink()
                        except:
                            pass
                except Exception as e:
                    log(f"[Auto-Align] Failed: {e}")
            else:
                log(f"[Auto-Align] StereoAutoAlign.exe not found, adding Align suffix only")

        # V58.2 FIX: Base SBS from inference is ORIG|GEN which is actually Cross-Eyed in VR
        # So Parallel needs No-Swap (left|right), XD needs Swap (right|left)
        # But earlier version had it reversed, fixed here

        if is_both:
            final_parallel = make_sbs(left, right)  # Parallel = no swap
            name_parallel = f"{base_name}_Align{suffix}.jpg"
            path_parallel = sbs_path.parent / name_parallel
            final_parallel.save(path_parallel, "JPEG", quality=95, subsampling=0)
            log(f"[OutputType] Saved {output_type} Parallel: {name_parallel}")

            final_xd = make_sbs(right, left)  # XD = swap
            name_xd = f"{base_name}_Align_XD{suffix}.jpg"
            path_xd = sbs_path.parent / name_xd
            final_xd.save(path_xd, "JPEG", quality=95, subsampling=0)
            log(f"[OutputType] Saved {output_type} XD: {name_xd}")

            try:
                if sbs_path.exists():
                    sbs_path.unlink()
                    log(f"[OutputType] Removed original: {sbs_path.name}")
            except:
                pass
            return [path_parallel, path_xd]

        elif is_xd and needs_align:
            final = make_sbs(right, left)  # XD = swap
            new_name = f"{base_name}_Align_XD{suffix}.jpg"
        elif needs_align:
            final = make_sbs(left, right)  # Parallel = no swap
            new_name = f"{base_name}_Align{suffix}.jpg"
        elif is_xd:
            final = make_sbs(right, left)  # XD = swap
            new_name = f"{base_name}_XD{suffix}.jpg"
        else:
            # Parallel default - fix if original was XD
            final = make_sbs(left, right)
            new_name = f"{base_name}{suffix}.jpg"
            new_path = sbs_path.parent / new_name
            if new_path != sbs_path:
                final.save(new_path, "JPEG", quality=95, subsampling=0)
                try:
                    sbs_path.unlink()
                except:
                    pass
                log(f"[OutputType] Fixed Parallel (swapped): {new_name}")
                return new_path
            else:
                temp_path = sbs_path.parent / f"_{base_name}_temp{suffix}.jpg"
                final.save(temp_path, "JPEG", quality=95, subsampling=0)
                try:
                    sbs_path.unlink()
                    temp_path.rename(sbs_path)
                    log(f"[OutputType] Fixed Parallel in place: {sbs_path.name}")
                except Exception as e:
                    log(f"[OutputType] Failed to overwrite: {e}")
                    return temp_path
                return sbs_path

        new_path = sbs_path.parent / new_name
        final.save(new_path, "JPEG", quality=95, subsampling=0)
        log(f"[OutputType] Saved {output_type}: {new_name}")
        try:
            if new_path != sbs_path and sbs_path.exists():
                sbs_path.unlink()
                log(f"[OutputType] Removed original: {sbs_path.name}")
        except:
            pass
        return new_path

    except Exception as e:
        if log_func:
            log_func(f"[OutputType] Failed: {e}")
            import traceback
            log_func(traceback.format_exc())
        else:
            print(f"[OutputType] Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
        return None


def create_sbs_image(left_img, right_img, output_type="parallel"):
    """
    Simple version for inference_bat_final.py
    - left_img, right_img: PIL Images
    - output_type: parallel, xd, etc.
    Returns: PIL SBS image
    """
    needs_align = "align" in output_type
    is_xd = "xd" in output_type

    # Note: Auto-align should be done via StereoAutoAlign.exe externally
    # Here we only handle swap logic
    if is_xd:
        return make_sbs(right_img, left_img)
    else:
        return make_sbs(left_img, right_img)


# For backward compatibility with GUI
def get_output_type_display_name(value):
    """Get display name from value"""
    for k, v in OUTPUT_TYPE_OPTIONS.items():
        if v == value:
            return k
    return "1. Parallel (Default)"
