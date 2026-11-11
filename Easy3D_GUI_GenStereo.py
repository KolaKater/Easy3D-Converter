# FIX Done color: Done 那行深蓝 -> 浅黄 #FFE87A, 深灰底上才看得清
# FIX About layout icon: About 弹窗整齐布局 + Easy3D 图标 (不用系统默认), Close 不被切
# FIX Preset apply gen_mode face_restore: apply_params 补 gen_mode / face_restore, Preset 加载时界面才变
# FIX v7.0 Icon BG images: 图标+背景改到 images\Easy3D.ico / images\Easy3D.png, 保留老路径回退
# FIX v7.0 Dark force v3: 修 Labelframe 样式名 (f 小写), io/pf2 创建时指 Dark
# FIX v7.0 Dark force v2: 浅色强制改深 (补 TRadiobutton+TScale, __init__ 最后再走一遍)
# FIX v7.0 Clear right of Cancel: Clear 在 Cancel 右边 + 浅色全改深
# FIX v7.0 OBS dark: OBS 深色风格 (底深灰, 按钮灰蓝, 选中蓝, 字浅灰)
# FIX v7.0 Done open folder: Done - 地址可点开文件夹 + 悬浮提示, 点击打开文件夹
# FIX v7.0 Running anim: Running... 打字机动画 R->Running...->R 循环, 让用户眼睛知道还在跑
# FIX v7.0 Warped Gamma: _WARPED 补最后那次 Gamma 1.40, 跟最终 SBS 一样亮 (只补明暗, 不动尺寸)
# FIX v7.0 Depth/Warped ratio: _DEPTH/_WARPED 改正图片比例 (vis 是推理图)
# FIX v75: 3D Strength, Follow Original (只显示标签, 文件名/预设/代码不动)
# FIX v73: Mode+Limit 上移到 Row 0 取代 Depth/Fusion, Fusion 取消显示 1.0, Depth 合并进 Baseline, 删 Mode 注释
# FIX v39: 删重复的 V37 else (一个 if 只能跟一个 else)
# FIX v38: 空列表显示 Easy3D.* 背景 (任意后缀)
# FIX v31: MDE 标签改短 DA2-L/B/S FD-L/S
"""
Easy3D - Diffusion & Depthmap Combi Edition, RTX 3070+ friendly.

Kept from the old SD3D GUI: Input/Output, Preset Save/Load/Delete,
Log, Preview, File List, Start / Suspend / Cancel, progress bar.
Removed: Clarity, Clarity Blend, Enhance, Text Fix, Mode, 
Resize Mode, Batch, Anti-Alias, Gamma A, Gamma B, Gamma panel.

New Easy3D options:
  Finetuned Model, Pretrained Models (VAE / Image Encoder),
  MDE Model, 3D Strength,
  Resolution, Steps, Seed, Output Type, Low VRAM, Save Depth/Warped.

"""
import json, os, random, subprocess, sys, threading, time
from pathlib import Path
import pathlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

__VERSION__ = "v7.0- 2026-11-11"
APP_TITLE = "Easy3D: I love Depth. I also love Light."
# About 窗口链接: GitHub 是占位, 把你的 GitHub 用户名告诉我, 我帮你换成真的
ABOUT_GITHUB_URL = "https://github.com/KolaKater/Easy3D-Converter"
ABOUT_FACEBOOK_URL = "https://www.facebook.com/100000242772138"
CONFIG_FILE = "preset/easy3d_gui_config.json"
PRESET_FILE = "preset/easy3d_presets.json"
LOG_FILE = "easy3d_debug.log"
SCRIPT_NAME = "test.py"
SCRIPT_ARBITRARY = "test_arbitrary_size.py"  # v3.1 UNUSED - patch path removed, Kept for reference, do NOT use.
# Final SBS size = ORIGINAL image size (W x H), NOT Res. Res controls inference detail. Res 1024+ on 3070 8GB is slow / may OOM - lower Res if OOM.
# v1.9 fixed dependencies (not options): VAE + Image Encoder live in models/
FIXED_VAE = "sd-vae-ft-mse"
FIXED_VAE_PATH = "models/sd-vae-ft-mse"
FIXED_ENCODER = "image_encoder"
FIXED_ENCODER_PATH = "models/image_encoder"
SUPPORTED_INPUT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"}
OUTPUT_JPG_QUALITY = 95
# ---- Gamma (v6.8) ----
# 公式: out = 255 * (in/255) ** gamma, LUT 实现
# GAMMA_PRE_SQUARE: 原图 -> 推理之前, 对原图做一次 0.7
# GAMMA_PRE_OUTPUT: SBS 合成后 -> Output Type 之前, 对整张 SBS 做一次 1.40
# 1.0 = 不做, GAMMA_ENABLED=False = 两个都不做
GAMMA_PRE_SQUARE = 0.70
GAMMA_PRE_OUTPUT = 1.40  # SBS 合成后 -> Output Type 之前, 整张 SBS
GAMMA_ENABLED = True

# ---- Auto Batch Chunk (v6.8, 代码里改, 不上界面) ----
# BATCH_CHUNK_SIZE: 每批几张. 2=最快最稳, 5=推荐, 10=又会变慢, 0=不分批(回到旧版一次跑完)
# BATCH_COOLDOWN_SEC: 每批之间歇几秒, 让显卡降温、显存释放. 0=不歇, 5=推荐, 10=等太久
BATCH_CHUNK_SIZE = 2
BATCH_COOLDOWN_SEC = 5
# ---- Defer FaceRestore (v6.8, correct) ----
# True = all diffusion+SBS first, Phase 2 does FaceRestore -> Output Type once at end.
# GFPGAN not loaded during diffusion, diffusion gets full 8GB.
DEFER_FACE_RESTORE = True

# test.py writes vis/<stem>/{left,generated_right,warped,disp}.png - GUI never scans Output for old files
try:
    from output_type import OUTPUT_TYPE_OPTIONS as _OT_OPTS, process_sbs_file as _process_sbs_file
except Exception as _e:
    print(f"[output_type] not found, using fallback: {_e}")
    _OT_OPTS = None
    _process_sbs_file = None

FINETUNED_OPTIONS = {
    "Finetune-v1.5": "genstereo-v1.5",
    "Finetune-v2.1": "genstereo-v2.1",
}
MDE_OPTIONS = {
    "FD-L": "flashdepth-l",
    "FD-S": "flashdepth-s",
    "DA2-L": "depth_anything_v2_vitl",
    "DA2-B": "depth_anything_v2_vitb",
    "DA2-S": "depth_anything_v2_vits",
}
# MDE model id -> relative checkpoint path (relative to Easy3D folder, no absolute paths).
# FlashDepth paths match flashdepth_model.py MODEL_FILES / find_config_dir():
#   models/FlashDepth/configs/flashdepth-l/iter_10001.pth
#   models/FlashDepth/configs/flashdepth-s/iter_14001.pth
MDE_PATHS = {
    "depth_anything_v2_vitl": "models/depth_anything_v2_vitl.pth",
    "depth_anything_v2_vitb": "models/depth_anything_v2_vitb.pth",
    "depth_anything_v2_vits": "models/depth_anything_v2_vits.pth",
    "flashdepth-l": "models/FlashDepth/configs/flashdepth-l/iter_10001.pth",
    "flashdepth-s": "models/FlashDepth/configs/flashdepth-s/iter_14001.pth",
    "flashdepth": "models/FlashDepth/configs/flashdepth/iter_43002.pth",
}

# MDE model id -> filename suffix (added after Finetuned model in output_stem)
MDE_SUFFIXES = {
    "flashdepth-l": "_FDL",
    "flashdepth-s": "_FDS",
    "flashdepth": "_FD",
    "depth_anything_v2_vitl": "_DA2L",
    "depth_anything_v2_vitb": "_DA2B",
    "depth_anything_v2_vits": "_DA2S",
    "custom": "_MDEC",
}
VAE_OPTIONS = {
    "sd-vae-ft-mse (default)": "sd-vae-ft-mse",
    "Custom - Use VAE Path below": "custom",
}
ENCODER_OPTIONS = {
    "sd-image-variations-diffusers / image_encoder (default)": "image_encoder",
    "Custom - Use Image Encoder Path below": "custom",
}
RESOLUTION_OPTIONS = {
    "512 - Common, Fast": 512,
    "768 - Finetune-v2.1": 768,
    "896 - Common": 896,
    "1280 - Finetune-v1.5 Recommended": 1280,
    "1344 - Finetune-v2.1 Recommended": 1344,
    "1792 - Common": 1792,
    "1920 - Finetune-v2.1": 1920,
    "2048 - Finetune-v1.5": 2048,
    "2240 - Finetune-v2.1 Recommended": 2240,
    "2560 - Finetune-v1.5": 2560,
    "2688 - Finetune-v2.1 Recommended": 2688,
    "2816 - Finetune-v1.5": 2816,
    "3072 - Common": 3072,
    "3136 - Finetune-v2.1 Recommended": 3136,
    "3328 - Finetune-v1.5": 3328,
    "3456 - Finetune-v2.1": 3456,
    "4096 - Finetune-v1.5": 4096,
    "4224 - Finetune-v2.1": 4224,
    "4352 - Finetune-v1.5": 4352,
    "4480 - Common, Recommended": 4480,
    "4928 - Common, Recommended": 4928,
    "4864 - Finetune-v1.5": 4864,
    "4992 - Finetune-v2.1": 4992,
    "5120 - Finetune-v1.5, Slowest": 5120,
}
FACE_RESTORE_OPTIONS = {
    "None": "none",
    "Both Eyes": "gfpgan_both",
}
GEN_MODE_OPTIONS = {
    "Orig_Right": "orig_right",
    "Orig_Left": "orig_left",
    "Both / Symmetric": "both",
}
MODE_OPTIONS = {
    "Quality - Step": "quality",
    "Fast - Decoupled A": "decoupled_a",
    "Fast - Decoupled B": "decoupled_b",
    "Fast - Decoupled C": "decoupled_c",
}
# FIX v26: Mode -> _FaDe suffix
_MODE_FADE_SUFFIX = {"decoupled": "_FaDe", "decoupled_a": "_FaDeA", "decoupled_b": "_FaDeB", "decoupled_c": "_FaDeC"}
def _mode_fade_suffix(mode):
    _m = str(mode or "quality").strip().lower()
    if _m == "decoupled": return "_FaDe"
    return _MODE_FADE_SUFFIX.get(_m, "")
# FIX v22: Limit to Source 列表, 和 test.py _RESOLUTION_LIST 一致 (四舍五入取最近)
_LIMIT_RES_LIST = [512,768,896,1024,1088,1216,1280,1344,1472,1600,1664,1728,1792,1920,1984,2048,2240,2304,2560,2688,2816,3072,3136,3328,3456,3584,3840,4096,4224,4352,4480,4608,4864,4928,4992,5120]
def _limit_to_source_gui(gui_res, orig_long, limit_on=True):
    """FIX v22: GUI 版 Limit to Source. orig>=768 才生效, orig<768 不往下压."""
    try:
        _g = int(gui_res); _o = int(orig_long) if orig_long else 0
    except Exception:
        return int(gui_res)
    if not limit_on or _o < 768:
        return _g
    _nearest = min(_LIMIT_RES_LIST, key=lambda x: abs(x - _o))
    return min(_g, int(_nearest))
STEPS_OPTIONS = {
    "5  - Blink preview": 5,          # 眨眼就出，只看构图
    "10 - Ultra fast preview": 10,    # 超快预览
    "15 - Fast": 15,                  # 快
    "20 - Fast preview": 20,          # 快预览
    "25 - Balanced preview": 25,      # 均衡预览，25 就是 20 和 30 中间
    "30 - Balanced": 30,              # 均衡
    "35 - Refined": 35,               # 精修，开始有细节了
    "40 - Detailed": 40,               # 细节版
    "45 - Near final": 45,             # 接近成品
    "50 - Default quality": 50,        # 官方默认，正式出图
    "75 - High quality, slow": 75,     # 最高画质，慢
}
OUTPUT_TYPE_OPTIONS = _OT_OPTS if _OT_OPTS else {
    "1. Parallel (Default)": "parallel",
    "2. Cross-Eyed (XD Swap)": "xd",
    "3. Auto-Align Parallel": "align_parallel",
    "4. Auto-Align Cross-Eyed (XD)": "align_xd",
    "5. Auto-Align Parallel+XD (Both - 2 Files)": "align_both",
}
SEED_OPTIONS = {
    "Random (Default)": "",
    "Seed A": "510902704",    
    "Seed B": "457315609",
    "Seed C": "402260655",
    "Seed D": "1999",
    "Seed E": "2026",
}

# Old MDE labels (before res patch) -> new labels. Old presets/configs keep working.
_MDE_LABEL_ALIASES = {
    "Depth Anything V2 ViT-L": "DA2-L",
    "Depth Anything V2 ViT-B": "DA2-B",
    "Depth Anything V2 ViT-S": "DA2-S",
    "FlashDepth-L - Official, 518px": "FD-L",
    "FlashDepth-S - Official, 392px, Fast": "FD-S",
}
_FINETUNED_LABEL_ALIASES = {
    "Genstereo-v1.5": "Finetune-v1.5",
    "Genstereo-v2.1": "Finetune-v2.1",
    "GenStereo-v1.5": "Finetune-v1.5",
    "GenStereo-v2.1": "Finetune-v2.1",
}
_MODE_LABEL_ALIASES = {
    "Fast - Decoupled A (768)": "Fast - Decoupled A",
    "Fast - Decoupled B (1024)": "Fast - Decoupled B",
    "Fast - Decoupled C (1344)": "Fast - Decoupled C",
    "Fast - Decoupled": "Fast - Decoupled B",
}
def _migrate(saved, options):
    if saved in _FINETUNED_LABEL_ALIASES and _FINETUNED_LABEL_ALIASES[saved] in options: return _FINETUNED_LABEL_ALIASES[saved]
    if saved in _MODE_LABEL_ALIASES and _MODE_LABEL_ALIASES[saved] in options: return _MODE_LABEL_ALIASES[saved]
    if saved in options:
        return saved
    if saved in _MDE_LABEL_ALIASES and _MDE_LABEL_ALIASES[saved] in options:
        return _MDE_LABEL_ALIASES[saved]
    return None

def _seed_label_for_value(val):
    """Seed value ("510902704" / "") -> Label ("Seed A" / "Random (Default)").

    修复持久化 bug：config/preset 应存 Label，但旧代码存了数字，
    _migrate() 重启时认不出来就回落到 "Random (Default)"。
    也能修复旧 config 里已经存坏的纯数字。
    """
    _sv = str(val)
    for _label, _v in SEED_OPTIONS.items():
        if str(_v) == _sv:
            return _label
    return None

def _apply_gamma(img, gamma):
    """out = 255*(in/255)**gamma, gamma=1.0 直接返回原图."""
    try:
        if img is None or not gamma or float(gamma) == 1.0:
            return img
        g = float(gamma)
        lut = [max(0, min(255, int(round(255.0 * ((i/255.0) ** g))))) for i in range(256)]
        return img.point(lut * 3)
    except Exception as e:
        print(f'[Gamma] failed {gamma}: {e}')
        return img

class GenStereoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE + " - " + __VERSION__)
        self.root.geometry("1280x820")
        try:
            # 图标: images\Easy3D.ico 优先, 再回退根目录 Easy3D.ico
            _icon = None
            for _cand in (Path(__file__).parent / "images" / "Easy3D.ico", Path(__file__).parent / "Easy3D.ico"):
                if _cand.exists():
                    _icon = _cand; break
            if _icon is not None:
                self.root.iconbitmap(str(_icon))
            else:
                # .ico 都没有: 用 images\Easy3D.png 做 iconphoto 兜底
                _png = None
                for _cand in (Path(__file__).parent / "images" / "Easy3D.png", Path(__file__).parent / "Easy3D.png"):
                    if _cand.exists():
                        _png = _cand; break
                if _png is not None:
                    from PIL import Image as _PILImg, ImageTk as _PILImgTk
                    _im = _PILImg.open(_png).convert("RGBA")
                    _im.thumbnail((64, 64), _PILImg.LANCZOS)
                    self._icon_tk = _PILImgTk.PhotoImage(_im)
                    self.root.iconphoto(True, self._icon_tk)
        except Exception as _e:
            print(f"[Icon] { _e}")
        self.root.minsize(1120, 750)
        self.process = None
        self._orig_backup = {}  # v6.8fix: original deleted during run -> use cached copy
        self.pause_flag = False
        self.stop_flag = False
        self.config = self.load_config()
        # --- v7.0 深色全套 v2: 含 TRadiobutton + TScale + Dark 面板版 ---
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        _BG = "#23272E"
        _PANEL = "#2B2F36"
        _ENTRY = "#1A1D21"
        _BTN = "#3A3F47"
        _BTN_HOVER = "#4A5058"
        _ACCENT = "#4A90D9"
        _TEXT = "#E8EAED"
        _MUTED = "#9AA0A8"
        self.root.configure(bg=_BG)
        self.root.option_add("*Listbox.background", _ENTRY)
        self.root.option_add("*Listbox.foreground", _TEXT)
        self.root.option_add("*Listbox.selectBackground", _ACCENT)
        self.root.option_add("*Listbox.selectForeground", "white")
        self.root.option_add("*Entry.background", _ENTRY)
        self.root.option_add("*Entry.foreground", _TEXT)
        style.configure("TFrame", background=_BG)
        style.configure("Dark.TFrame", background=_PANEL)
        style.configure("TLabel", font=("Segoe UI", 9), background=_BG, foreground=_TEXT)
        style.configure("Dark.TLabel", font=("Segoe UI", 9), background=_PANEL, foreground=_TEXT)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), background=_BG, foreground=_TEXT)
        style.configure("Small.TLabel", font=("Segoe UI", 8), background=_BG, foreground=_MUTED)
        style.configure("Dark.Small.TLabel", font=("Segoe UI", 8), background=_PANEL, foreground=_MUTED)
        style.configure("TLabelframe", background=_PANEL, foreground=_TEXT, borderwidth=1, relief="flat")
        style.configure("Dark.TLabelframe", background=_PANEL, foreground=_TEXT, borderwidth=1, relief="flat")
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"), background=_PANEL, foreground=_TEXT)
        style.configure("Dark.TLabelframe.Label", font=("Segoe UI", 9, "bold"), background=_PANEL, foreground=_TEXT)
        style.configure("TButton", font=("Segoe UI", 9), background=_BTN, foreground=_TEXT, borderwidth=1, relief="flat", padding=(10, 4))
        style.map("TButton", background=[("active", _BTN_HOVER), ("disabled", "#2B2F36")], foreground=[("disabled", _MUTED)])
        style.configure("TCheckbutton", font=("Segoe UI", 9), background=_BG, foreground=_TEXT)
        style.configure("Dark.TCheckbutton", font=("Segoe UI", 9), background=_PANEL, foreground=_TEXT)
        style.map("TCheckbutton", background=[("active", _BG)], foreground=[("active", _TEXT)])
        style.map("Dark.TCheckbutton", background=[("active", _PANEL)], foreground=[("active", _TEXT)])
        style.configure("TRadiobutton", font=("Segoe UI", 9), background=_BG, foreground=_TEXT)
        style.configure("Dark.TRadiobutton", font=("Segoe UI", 9), background=_PANEL, foreground=_TEXT)
        style.map("TRadiobutton", background=[("active", _BG)], foreground=[("active", _TEXT)])
        style.map("Dark.TRadiobutton", background=[("active", _PANEL)], foreground=[("active", _TEXT)])
        style.configure("TEntry", fieldbackground=_ENTRY, foreground=_TEXT, borderwidth=1, insertcolor=_TEXT)
        style.configure("Dark.TEntry", fieldbackground=_ENTRY, foreground=_TEXT, borderwidth=1, insertcolor=_TEXT)
        style.configure("TCombobox", fieldbackground=_ENTRY, background=_BTN, foreground=_TEXT, arrowcolor=_TEXT, borderwidth=1, selectbackground=_ENTRY, selectforeground=_TEXT)
        style.configure("Dark.TCombobox", fieldbackground=_ENTRY, background=_BTN, foreground=_TEXT, arrowcolor=_TEXT, borderwidth=1, selectbackground=_ENTRY, selectforeground=_TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", _ENTRY)], foreground=[("readonly", _TEXT)], background=[("readonly", _BTN)])
        style.map("Dark.TCombobox", fieldbackground=[("readonly", _ENTRY)], foreground=[("readonly", _TEXT)], background=[("readonly", _BTN)])
        style.configure("Horizontal.TScale", background=_PANEL, troughcolor=_ENTRY, borderwidth=0)
        style.configure("Dark.Horizontal.TScale", background=_PANEL, troughcolor=_ENTRY, borderwidth=0)
        style.configure("Vertical.TScale", background=_PANEL, troughcolor=_ENTRY, borderwidth=0)
        style.configure("Horizontal.TProgressbar", background=_ACCENT, troughcolor=_ENTRY, borderwidth=0)
        style.configure("Vertical.TScrollbar", background=_BTN, troughcolor=_PANEL, borderwidth=0, arrowcolor=_TEXT)
        style.configure("Horizontal.TScrollbar", background=_BTN, troughcolor=_PANEL, borderwidth=0, arrowcolor=_TEXT)
        self._obs_colors = (_BG, _TEXT, _MUTED, _ACCENT)

        main = ttk.Frame(root, padding=(10, 6)); main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main, text=APP_TITLE, style="Header.TLabel").pack()

        # ---- Input / Output (v6.8mockup: Browse left, Entry middle, checkboxes right) ----
        io = ttk.LabelFrame(main, text="Input / Output", padding=(10, 6), style="Dark.TLabelframe"); io.pack(fill=tk.X, pady=(0, 6))
        # Row 0: Input: [Browse File][Browse Folder][Entry ........] [Save Depth][Save Warped][Low VRAM]
        io_r0 = ttk.Frame(io, style="Dark.TFrame"); io_r0.pack(fill=tk.X, pady=2)
        ttk.Label(io_r0, text="Input:").pack(side=tk.LEFT, padx=5)
        self.input_var = tk.StringVar(value=self.config.get("input_path", ""))
        ttk.Button(io_r0, text="Browse File", width=13, command=self.browse_input_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_r0, text="Browse Folder", width=13, command=self.browse_input_folder).pack(side=tk.LEFT, padx=2)
        self.input_entry = ttk.Entry(io_r0, textvariable=self.input_var); self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        # Drag-drop: drop file/folder onto Input entry or whole window
        self._setup_dnd()
        self.save_depth_var = tk.BooleanVar(value=bool(self.config.get("save_depth", False)))
        ttk.Checkbutton(io_r0, text="Save Depth Map", variable=self.save_depth_var).pack(side=tk.LEFT, padx=8)
        self.save_warped_var = tk.BooleanVar(value=bool(self.config.get("save_warped", False)))
        ttk.Checkbutton(io_r0, text="Save Warped Image", variable=self.save_warped_var).pack(side=tk.LEFT, padx=8)
        self.lowvram_var = tk.BooleanVar(value=bool(self.config.get("low_vram", True)))
        ttk.Checkbutton(io_r0, text="Low VRAM (3070 8GB)", variable=self.lowvram_var).pack(side=tk.LEFT, padx=8)
        # Row 1: Output: [Default/Custom][Choose Folder][Entry ....] [Preset Combo][Preset: Load Save Delete]
        io_r1 = ttk.Frame(io, style="Dark.TFrame"); io_r1.pack(fill=tk.X, pady=2)
        ttk.Label(io_r1, text="Output:").pack(side=tk.LEFT, padx=5)
        self.output_mode_var = tk.StringVar(value=self.config.get("output_mode", "default_new"))
        self.output_path_var = tk.StringVar(value=self.config.get("output_path", ""))
        ttk.Radiobutton(io_r1, text="Default (InputName_3D)", variable=self.output_mode_var, value="default_new", command=self.on_output_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(io_r1, text="Custom", variable=self.output_mode_var, value="custom", command=self.on_output_mode_change).pack(side=tk.LEFT, padx=8)
        ttk.Button(io_r1, text="Choose Folder", width=13, command=self.browse_output_folder).pack(side=tk.LEFT, padx=2)
        self.output_entry = ttk.Entry(io_r1, textvariable=self.output_path_var); self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.preset_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(io_r1, textvariable=self.preset_var, state="normal", width=28); self.preset_combo.pack(side=tk.LEFT, padx=2)
        ttk.Label(io_r1, text="Preset:").pack(side=tk.LEFT, padx=5)
        ttk.Button(io_r1, text="Load", width=10, command=self.on_preset_load).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_r1, text="Save", width=10, command=self.on_preset_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_r1, text="Delete", width=10, command=self.on_preset_delete).pack(side=tk.LEFT, padx=2)
        self.refresh_preset_combo()
        # v6.8: GenStereo Model frame removed - Finetuned / MDE moved into Parameters (mockup).
        self.finetuned_path_var = tk.StringVar(value="models")
        self.mde_path_var = tk.StringVar(value="models/FlashDepth/configs/flashdepth-l/iter_10001.pth")  # FIX v18b: 默认 MDE 路径 -> FlashDepth-L
        self.vae_var = tk.StringVar(value=list(VAE_OPTIONS)[0])
        self.vae_path_var = tk.StringVar(value=FIXED_VAE_PATH)
        self.encoder_var = tk.StringVar(value=list(ENCODER_OPTIONS)[0])
        self.encoder_path_var = tk.StringVar(value=FIXED_ENCODER_PATH)
        # ---- Stereo Parameters (v6.8mockup: 12-col, 3 rows, no separate Model frame) ----
        pf2 = ttk.LabelFrame(main, text="Stereo Parameters", padding=(10, 8), style="Dark.TLabelframe"); pf2.pack(fill=tk.X, pady=(0, 6))
        for _cc in range(12): pf2.columnconfigure(_cc, weight=1)
        # Row 0: Mode / Limit to Source / Baseline / Guidance (V73: Mode+Limit 取代 Depth/Fusion)
        # V73: Depth 合并进 Baseline, Fusion 固定 1.0
        _old_depth = float(self.config.get("depth_strength", 1.0) or 1.0)
        _old_base = float(self.config.get("baseline", 0.4) or 0.4)
        _merged_base = _old_base if float(self.config.get("baseline_merged", 0) or 0) else _old_base * _old_depth
        _merged_base = max(0.2, min(2.0, float(_merged_base)))
        self.depth_var = tk.DoubleVar(value=1.0)  # V73: 隐藏, 固定 1.0, Depth 已折叠进 Baseline
        self.fusion_var = tk.DoubleVar(value=1.0)  # V73: 隐藏, 固定 1.0 (Rowan mask-only 只有 1.0 才好看)
        # V73: Mode + Limit 上移到 Row 0, 取代原来 Depth/Fusion 的位置 (先选处理方式, 再选模型/分辨率)
        self.mode_var = tk.StringVar(value=_migrate(self.config.get("mode_display", ""), MODE_OPTIONS) or list(MODE_OPTIONS)[0])
        ttk.Label(pf2, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.mode_var, values=list(MODE_OPTIONS), state="readonly").grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.limit_var = tk.BooleanVar(value=bool(self.config.get("limit_to_source", True)))
        ttk.Checkbutton(pf2, text="Limit to Source", variable=self.limit_var, command=self.save_config_event).grid(row=0, column=3, columnspan=3, sticky=tk.W, padx=5, pady=4)
        self.baseline_var = tk.DoubleVar(value=_merged_base)
        ttk.Label(pf2, text="3D Strength:").grid(row=0, column=6, sticky=tk.W, padx=5, pady=4)
        ttk.Scale(pf2, from_=0.2, to=2.0, orient=tk.HORIZONTAL, variable=self.baseline_var, command=self._on_baseline).grid(row=0, column=7, sticky=tk.EW, padx=5, pady=4)
        self.baseline_label = ttk.Label(pf2, text=f"{self.baseline_var.get():.2f}"); self.baseline_label.grid(row=0, column=8, sticky=tk.W, padx=5)
        self.guidance_var = tk.DoubleVar(value=float(self.config.get("guidance", 1.5)))
        ttk.Label(pf2, text="Follow Original:").grid(row=0, column=9, sticky=tk.W, padx=5, pady=4)
        ttk.Scale(pf2, from_=0.0, to=5.0, orient=tk.HORIZONTAL, variable=self.guidance_var, command=self._on_guidance).grid(row=0, column=10, sticky=tk.EW, padx=5, pady=4)
        self.guidance_label = ttk.Label(pf2, text=f"{self.guidance_var.get():.2f}"); self.guidance_label.grid(row=0, column=11, sticky=tk.W, padx=5)
        # Row 1: Finetuned / MDE / Resolution / Steps (each: Label + Combo span 2 = 3 cols)
        self.finetuned_var = tk.StringVar(value=_migrate(self.config.get("finetuned_display", ""), FINETUNED_OPTIONS) or list(FINETUNED_OPTIONS)[0])
        ttk.Label(pf2, text="Finetuned:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.finetuned_var, values=list(FINETUNED_OPTIONS), state="readonly").grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.mde_var = tk.StringVar(value=_migrate(self.config.get("mde_display", ""), MDE_OPTIONS) or list(MDE_OPTIONS)[0])
        ttk.Label(pf2, text="MDE:").grid(row=1, column=3, sticky=tk.W, padx=5, pady=4)
        self.mde_combo = ttk.Combobox(pf2, textvariable=self.mde_var, values=list(MDE_OPTIONS), state="readonly")
        self.mde_combo.grid(row=1, column=4, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.mde_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mde_change())
        self.resolution_var = tk.StringVar(value=_migrate(self.config.get("resolution_display", ""), RESOLUTION_OPTIONS) or list(RESOLUTION_OPTIONS)[0])
        ttk.Label(pf2, text="Resolution:").grid(row=1, column=6, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.resolution_var, values=list(RESOLUTION_OPTIONS), state="readonly").grid(row=1, column=7, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.steps_var = tk.StringVar(value=_migrate(self.config.get("steps_display", ""), STEPS_OPTIONS) or "50 - Default quality")
        ttk.Label(pf2, text="Steps:").grid(row=1, column=9, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.steps_var, values=list(STEPS_OPTIONS), state="readonly").grid(row=1, column=10, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        # Row 2: Gen Mode / Seed / Face Restore / Output Type
        self.gen_mode_var = tk.StringVar(value=_migrate(self.config.get("gen_mode_display", ""), GEN_MODE_OPTIONS) or list(GEN_MODE_OPTIONS)[0])
        ttk.Label(pf2, text="Gen Mode:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.gen_mode_var, values=list(GEN_MODE_OPTIONS), state="readonly").grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        # Seed：同时接受存的 Label ("Seed A") 和旧版本存坏的数字 ("510902704")
        _saved_seed = self.config.get("seed_display", "")
        _seed_label = _migrate(_saved_seed, SEED_OPTIONS) or _seed_label_for_value(_saved_seed)
        self.seed_var = tk.StringVar(value=_seed_label or list(SEED_OPTIONS)[0])
        ttk.Label(pf2, text="Seed:").grid(row=2, column=3, sticky=tk.W, padx=5, pady=4)
        self.seed_combo = ttk.Combobox(pf2, textvariable=self.seed_var, values=list(SEED_OPTIONS), state="normal")
        self.seed_combo.grid(row=2, column=4, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.seed_combo.bind("<<ComboboxSelected>>", lambda e: self._on_seed_custom())
        self.seed_combo.bind("<Return>", lambda e: self._on_seed_custom())
        self.seed_combo.bind("<FocusOut>", lambda e: self._on_seed_custom())
        self.face_restore_var = tk.StringVar(value=_migrate(self.config.get("face_restore_display", ""), FACE_RESTORE_OPTIONS) or list(FACE_RESTORE_OPTIONS)[0])
        ttk.Label(pf2, text="Face Restore:").grid(row=2, column=6, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.face_restore_var, values=list(FACE_RESTORE_OPTIONS), state="readonly").grid(row=2, column=7, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        self.output_type_var = tk.StringVar(value=_migrate(self.config.get("output_type_display", ""), OUTPUT_TYPE_OPTIONS) or list(OUTPUT_TYPE_OPTIONS)[0])
        ttk.Label(pf2, text="Output Type:").grid(row=2, column=9, sticky=tk.W, padx=5, pady=4)
        ttk.Combobox(pf2, textvariable=self.output_type_var, values=list(OUTPUT_TYPE_OPTIONS), state="readonly").grid(row=2, column=10, columnspan=2, sticky=tk.EW, padx=5, pady=4)
        # ---- Buttons / progress ----
        bf = ttk.Frame(main); bf.pack(fill=tk.X, pady=(8, 4))
        # v7.0 Clear: 在 Cancel 右边 (先 pack 的最靠右, 所以 Clear 要在 Cancel 之前 pack)
        self.clear_btn = ttk.Button(bf, text="Clear", width=10, command=self.clear_input_and_filelist); self.clear_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.cancel_btn = ttk.Button(bf, text="Cancel", width=10, command=self.cancel_conversion, state=tk.DISABLED); self.cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.suspend_btn = ttk.Button(bf, text="Suspend", width=10, command=self.toggle_suspend, state=tk.DISABLED); self.suspend_btn.pack(side=tk.RIGHT, padx=5)
        self.about_btn = ttk.Button(bf, text="About", width=10, command=self._show_about); self.about_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.start_btn = ttk.Button(bf, text="Start", width=10, command=self.start_conversion); self.start_btn.pack(side=tk.RIGHT, padx=5)
        pg = ttk.Frame(main); pg.pack(fill=tk.X, pady=2); pg.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        # v7.0: Done 那行可点开文件夹, 所以 Label 要拿到引用, 绑定点击 + 悬浮提示
        self.status_label = ttk.Label(pg, textvariable=self.status_var, style="Small.TLabel")
        self.status_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self._done_dir = None
        self._status_tip = None
        self.status_label.bind("<Button-1>", lambda e: self._open_done_folder())
        self.status_label.bind("<Enter>", lambda e: self._show_status_tip())
        self.status_label.bind("<Leave>", lambda e: self._hide_status_tip())
        self.progress_var = tk.DoubleVar(value=0); ttk.Progressbar(pg, variable=self.progress_var, maximum=100).grid(row=0, column=1, sticky=tk.EW, padx=(10, 0))

        tf = ttk.Frame(main); tf.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(tf, text="View:", style="Small.TLabel").pack(side=tk.LEFT, padx=(5, 10))
        self.show_log_var = tk.BooleanVar(value=self.config.get("show_log", False))
        self.show_preview_var = tk.BooleanVar(value=self.config.get("show_preview", True))
        self.show_filelist_var = tk.BooleanVar(value=self.config.get("show_filelist", True))
        ttk.Checkbutton(tf, text="Log", variable=self.show_log_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(tf, text="Preview", variable=self.show_preview_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(tf, text="File List", variable=self.show_filelist_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)

        bottom = ttk.Frame(main); bottom.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        for c in (0, 1, 2): bottom.columnconfigure(c, weight=1)
        bottom.rowconfigure(0, weight=1)
        # v6.8: Log / File List fill the same height as Preview (no fixed height, rowconfigure weight=1)
        self.log_frame = ttk.LabelFrame(bottom, text="Log", padding=5, style="Dark.TLabelframe"); self.log_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5)); self.log_frame.rowconfigure(0, weight=1); self.log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, font=("Consolas", 8), bg="#1A1D21", fg="#E8EAED", insertbackground="white")
        sb = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview); self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW); sb.grid(row=0, column=1, sticky=tk.NS)
        self.preview_frame = ttk.LabelFrame(bottom, text="Preview (Input)", padding=5, style="Dark.TLabelframe"); self.preview_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5); self.preview_frame.rowconfigure(0, weight=1); self.preview_frame.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(self.preview_frame, text="No Image", anchor=tk.CENTER); self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label.bind("<Configure>", lambda e: self._refresh_preview_scaled())
        self.preview_pil = None; self.preview_tk = None
        self.file_frame = ttk.LabelFrame(bottom, text="File List", padding=5, style="Dark.TLabelframe"); self.file_frame.grid(row=0, column=2, sticky=tk.NSEW, padx=(5, 0)); self.file_frame.rowconfigure(0, weight=1); self.file_frame.columnconfigure(0, weight=1)
        self.file_listbox = tk.Listbox(self.file_frame, font=("Consolas", 8), bg="#1A1D21", fg="#E8EAED", selectbackground="#4A90D9")
        fs = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview); self.file_listbox.configure(yscrollcommand=fs.set)
        self.file_listbox.grid(row=0, column=0, sticky=tk.NSEW); fs.grid(row=0, column=1, sticky=tk.NS)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_list_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.refresh_file_list_and_preview)
        self.root.after(400, self.toggle_panels)
        self._apply_dark_force()
        self.log("GUI started - " + APP_TITLE + " " + __VERSION__)

    def _setup_dnd(self):
        if not HAS_DND:
            print('[DnD] tkinterdnd2 not installed, drag-drop disabled')
            return
        try:
            for _w in (self.root, self.input_entry):
                _w.drop_target_register(DND_FILES)
                _w.dnd_bind('<<Drop>>', self._on_drop)
            print('[DnD] Drag-drop enabled: drop file/folder onto Input')
        except Exception as e:
            print(f'[DnD] setup failed: {e}')

    def _on_drop(self, event):
        try:
            files = self.root.tk.splitlist(event.data)
            if not files: return
            # Clean {braces} - splitlist already handles, strip quotes
            files = [str(f).strip('{}').strip('"') for f in files]
            p = pathlib.Path(files[0]) if False else Path(files[0])
            # Multiple files in same folder -> use folder; single file/folder -> use it
            if len(files) > 1:
                parents = {str(Path(f).parent) for f in files}
                if len(parents) == 1 and all(Path(f).is_file() for f in files):
                    p = Path(files[0]).parent
                else:
                    p = Path(files[0])
            self.input_var.set(str(p))
            self.save_config_event()
            self.log(f'[DnD] Dropped: {p}')
            self.root.after(100, self.refresh_file_list_and_preview)
        except Exception as e:
            self.log(f'[DnD] Drop failed: {e}')

    def _on_mde_change(self, *a):
        # MDE changed -> auto-switch mde_path to the matching relative checkpoint.
        # Custom keeps the current mde_path (user-typed). FlashDepth / DepthAnything auto-fill.
        try:
            mid = MDE_OPTIONS.get(self.mde_var.get(), '')
            if mid and mid != 'custom' and mid in MDE_PATHS:
                self.mde_path_var.set(MDE_PATHS[mid])
            self.save_config_event()
        except Exception as e: print(f'[MDE] {e}')

    def _on_seed_custom(self, *a):
        """把 Seed 规范化为 Label 并立刻保存。

        原来这个方法不存在（Return/FocusOut 绑定会报 AttributeError），
        且没有 <<ComboboxSelected>> 绑定，改 Seed 不一定能进 save_config()。
        """
        try:
            raw = (self.seed_var.get() or "").strip()
            if not raw or raw in SEED_OPTIONS:
                self.save_config_event()
                return
            import re
            m = re.search(r"-?\d+", raw)
            if m:
                _sv = m.group(0)
                _lbl = f"{_sv} - Custom"
                if _lbl not in SEED_OPTIONS:
                    SEED_OPTIONS[_lbl] = _sv
                    try:
                        self.seed_combo["values"] = list(SEED_OPTIONS)
                    except Exception:
                        pass
                self.seed_var.set(_lbl)
            self.save_config_event()
        except Exception as e:
            print(f'[Seed] {e}')

    def _on_depth(self, v):
        try: self.depth_label.configure(text=f"{float(v):.2f}")
        except Exception: pass
        self.save_config_event()
    def _on_baseline(self, v):
        try: self.baseline_label.configure(text=f"{float(v):.2f}")
        except Exception: pass
        self.save_config_event()
    def _on_fusion(self, v):
        try: self.fusion_label.configure(text=f"{float(v):.2f}")
        except Exception: pass
        self.save_config_event()
    def _on_guidance(self, v):
        try: self.guidance_label.configure(text=f"{float(v):.2f}")
        except Exception: pass
        self.save_config_event()

    # ---- config / presets ----
    def load_config(self):
        try:
            p = Path(CONFIG_FILE)
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e: print(f"[Config] {e}")
        return {}
    def save_config(self):
        try:
            p = Path(CONFIG_FILE); p.parent.mkdir(parents=True, exist_ok=True)
            cfg = {
                "input_path": self.input_var.get(), "output_mode": self.output_mode_var.get(), "output_path": self.output_path_var.get(),
                "finetuned_display": self.finetuned_var.get(), "gen_mode_display": self.gen_mode_var.get(), "finetuned_path": self.finetuned_path_var.get(),
                "mde_display": self.mde_var.get(), "mde_path": self.mde_path_var.get(),
                "vae_display": self.vae_var.get(), "vae_path": self.vae_path_var.get(),
                "encoder_display": self.encoder_var.get(), "encoder_path": self.encoder_path_var.get(),
                "depth_strength": 1.0, "baseline": self.baseline_var.get(), "baseline_merged": 1,
                "fusion_strength": 1.0, "guidance": self.guidance_var.get(),
                "resolution_display": self.resolution_var.get(), "steps_display": self.steps_var.get(), "mode_display": self.mode_var.get(), "limit_to_source": bool(self.limit_var.get()),
                "seed_display": self.seed_var.get(), "output_type_display": self.output_type_var.get(), "face_restore_display": self.face_restore_var.get(),
                "low_vram": self.lowvram_var.get(), "save_depth": self.save_depth_var.get(), "save_warped": self.save_warped_var.get(),
                "show_log": self.show_log_var.get(), "show_preview": self.show_preview_var.get(), "show_filelist": self.show_filelist_var.get(),
            }
            p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e: print(f"[Config Save] {e}")
    def save_config_event(self, *a): self.save_config()
    def load_presets(self):
        try:
            p = Path(PRESET_FILE)
            if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
        return {}
    def save_presets(self, d):
        p = Path(PRESET_FILE); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"); return True
    def refresh_preset_combo(self):
        self.preset_combo["values"] = list(self.load_presets().keys())
    def collect_params(self):
        seed_raw = (self.seed_var.get() or "").strip()
        seed = SEED_OPTIONS.get(seed_raw, "")
        if seed_raw not in SEED_OPTIONS and seed_raw:
            import re
            m = re.search(r"-?\d+", seed_raw)
            if m and "random" not in seed_raw.lower(): seed = m.group(0)
        if not seed: seed = str(random.randint(0, 2147483647))
        return {
            "gen_mode": GEN_MODE_OPTIONS.get(self.gen_mode_var.get(), "orig_right"),
            "face_restore": FACE_RESTORE_OPTIONS.get(self.face_restore_var.get(), "none"),
            "finetuned_model": FINETUNED_OPTIONS.get(self.finetuned_var.get(), "genstereo-v1.5"),
            "finetuned_path": self.finetuned_path_var.get().strip(),
            "mde_model": MDE_OPTIONS.get(self.mde_var.get(), "flashdepth-l"),
            "mde_path": (MDE_PATHS.get(MDE_OPTIONS.get(self.mde_var.get(), ""), "") or self.mde_path_var.get().strip()),
            "vae": VAE_OPTIONS.get(self.vae_var.get(), "sd-vae-ft-mse"),
            "vae_path": self.vae_path_var.get().strip(),
            "image_encoder": ENCODER_OPTIONS.get(self.encoder_var.get(), "image_encoder"),
            "image_encoder_path": self.encoder_path_var.get().strip(),
            "depth_strength": 1.0,
            "baseline": round(float(self.baseline_var.get()), 3),
            "fusion_strength": 1.0,
            "guidance_scale": round(float(self.guidance_var.get()), 3),
            "resolution": RESOLUTION_OPTIONS.get(self.resolution_var.get(), 512),
            "mode": MODE_OPTIONS.get(self.mode_var.get(), "quality"),
            "limit_to_source": bool(self.limit_var.get()),
            "steps": STEPS_OPTIONS.get(self.steps_var.get(), 50),
            "seed": seed,
            "output_type": OUTPUT_TYPE_OPTIONS.get(self.output_type_var.get(), "parallel"),
            "low_vram": bool(self.lowvram_var.get()),
            "save_depth": bool(self.save_depth_var.get()),
            "save_warped": bool(self.save_warped_var.get()),
        }
    def apply_params(self, d):
        rev = lambda opts, val: next((k for k, v in opts.items() if v == val), None)
        if "finetuned_model" in d and rev(FINETUNED_OPTIONS, d["finetuned_model"]): self.finetuned_var.set(rev(FINETUNED_OPTIONS, d["finetuned_model"]))
        if "mde_model" in d and rev(MDE_OPTIONS, d["mde_model"]): self.mde_var.set(rev(MDE_OPTIONS, d["mde_model"]))
        if "mde_model" in d and d["mde_model"] in MDE_PATHS: self.mde_path_var.set(MDE_PATHS[d["mde_model"]])
        if "vae" in d and rev(VAE_OPTIONS, d["vae"]): self.vae_var.set(rev(VAE_OPTIONS, d["vae"]))
        if "image_encoder" in d and rev(ENCODER_OPTIONS, d["image_encoder"]): self.encoder_var.set(rev(ENCODER_OPTIONS, d["image_encoder"]))
        if "gen_mode" in d and rev(GEN_MODE_OPTIONS, d["gen_mode"]): self.gen_mode_var.set(rev(GEN_MODE_OPTIONS, d["gen_mode"]))
        if "face_restore" in d and rev(FACE_RESTORE_OPTIONS, d["face_restore"]): self.face_restore_var.set(rev(FACE_RESTORE_OPTIONS, d["face_restore"]))
        # V73: 老 preset 的 Depth 折叠进 Baseline, Fusion 固定 1.0
        _d_depth = float(d.get("depth_strength", 1.0) or 1.0) if "depth_strength" in d else 1.0
        if "baseline" in d: self.baseline_var.set(max(0.2, min(2.0, float(d["baseline"]) * _d_depth))); self._on_baseline(self.baseline_var.get())
        self.depth_var.set(1.0); self.fusion_var.set(1.0)
        if "guidance_scale" in d: self.guidance_var.set(float(d["guidance_scale"])); self._on_guidance(d["guidance_scale"])
        if "resolution" in d and rev(RESOLUTION_OPTIONS, d["resolution"]): self.resolution_var.set(rev(RESOLUTION_OPTIONS, d["resolution"]))
        if "mode" in d and rev(MODE_OPTIONS, d["mode"]): self.mode_var.set(rev(MODE_OPTIONS, d["mode"]))
        if "limit_to_source" in d:
            try: self.limit_var.set(bool(d["limit_to_source"]))
            except Exception: pass
        if "steps" in d and rev(STEPS_OPTIONS, d["steps"]): self.steps_var.set(rev(STEPS_OPTIONS, d["steps"]))
        if "seed" in d:
            # 修复：绝不要把数字 ("510902704") 存进 seed_var。
            # seed_var 必须始终是 Label ("Seed A")，否则下次保存写进
            # config 的是数字，_migrate() 重启时认不出来，Seed 就重置回 Random。
            _sv = str(d["seed"])
            _label = _seed_label_for_value(_sv)
            if _label:
                self.seed_var.set(_label)
            elif _sv:
                _lbl = f"{_sv} - Custom"
                if _lbl not in SEED_OPTIONS:
                    SEED_OPTIONS[_lbl] = _sv
                try:
                    self.seed_combo["values"] = list(SEED_OPTIONS)
                except Exception:
                    pass
                self.seed_var.set(_lbl)
            else:
                self.seed_var.set("Random (Default)")
        if "output_type" in d and rev(OUTPUT_TYPE_OPTIONS, d["output_type"]): self.output_type_var.set(rev(OUTPUT_TYPE_OPTIONS, d["output_type"]))
        for k, var in (("finetuned_path", self.finetuned_path_var), ("mde_path", self.mde_path_var), ("vae_path", self.vae_path_var), ("image_encoder_path", self.encoder_path_var)):
            if k in d: var.set(d[k])
        for k, var in (("low_vram", self.lowvram_var), ("save_depth", self.save_depth_var), ("save_warped", self.save_warped_var)):
            if k in d: var.set(bool(d[k]))
    def on_preset_save(self):
        name = self.preset_var.get().strip()
        if not name: messagebox.showwarning("Save Preset", "Type a preset name first"); return
        p = self.load_presets(); p[name] = self.collect_params(); self.save_presets(p); self.refresh_preset_combo(); self.log(f"[Preset] Saved '{name}'")
    def on_preset_load(self):
        name = self.preset_var.get().strip(); p = self.load_presets()
        if name not in p: messagebox.showerror("Load Preset", f"Preset '{name}' not found"); return
        self.apply_params(p[name]); self.save_config(); self.log(f"[Preset] Loaded '{name}'"); messagebox.showinfo("Load Preset", f"Preset '{name}' loaded!")
    def on_preset_delete(self):
        name = self.preset_var.get().strip(); p = self.load_presets()
        if name not in p: messagebox.showwarning("Delete Preset", "Select a preset"); return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"): return
        del p[name]; self.save_presets(p); self.preset_var.set(""); self.refresh_preset_combo(); self.log(f"[Preset] Deleted '{name}'")

    # ---- IO / preview / file list ----
    def on_output_mode_change(self):
        try:
            if self.output_mode_var.get() == "custom": self.output_entry.grid()
            else: self.output_entry.grid_remove()
        except Exception: pass
        self.save_config_event()
    def browse_input_file(self):
        p = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif *.avif *.heic *.heif"), ("All", "*.*")])
        if p: self.input_var.set(p); self.save_config_event(); self.root.after(100, self.refresh_file_list_and_preview)
    def browse_input_folder(self):
        p = filedialog.askdirectory(title="Select Input Folder")
        if p: self.input_var.set(p); self.save_config_event(); self.root.after(100, self.refresh_file_list_and_preview)
    def browse_output_folder(self):
        p = filedialog.askdirectory(title="Select Output Folder")
        if p: self.output_path_var.set(p); self.output_mode_var.set("custom"); self.on_output_mode_change(); self.save_config_event()
    def get_output_dir(self, input_path):
        inp = Path(input_path)
        if self.output_mode_var.get() == "custom" and self.output_path_var.get().strip():
            out = Path(self.output_path_var.get().strip())
            if inp.is_dir() and out.resolve() == inp.resolve():
                messagebox.showerror("Output Error", "If input is a folder, output must be a different folder."); return None
            return out
        return inp.parent / f"{inp.stem}_3D" if inp.is_file() else inp.parent / f"{inp.name}_3D"
    def refresh_file_list_and_preview(self):
        try:
            inp = self.input_var.get().strip().strip('"')
            self.file_listbox.delete(0, tk.END)
            if not inp:
                # FIX v38: Input 为空也显示背景 (Easy3D.* 任意后缀)
                _bg0 = None
                for _base0 in (Path(__file__).parent / "images", Path(__file__).parent):
                    for _ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"):
                        _c0 = Path(_base0) / ("Easy3D" + _ext)
                        if _c0.exists(): _bg0 = _c0; break
                    if _bg0 is not None: break
                if _bg0 is not None: self.show_preview(_bg0)
                return
            p = Path(inp)
            files = [p] if p.is_file() else [x for x in sorted(p.iterdir()) if x.suffix.lower() in SUPPORTED_INPUT_EXTS] if p.is_dir() else []
            for f in files: self.file_listbox.insert(tk.END, f.name)
            if files: self.show_preview(files[0])
            else:
                # FIX v38: 空列表显示背景, Easy3D.* 任意图片后缀 (不只 .png, 同名不同后缀也能找到)
                _bg = None
                _bases = []
                try: _bases.append(Path(__file__).parent / "images")
                except Exception: pass
                try: _bases.append(p)
                except Exception: pass
                try: _bases.append(Path(__file__).parent)
                except Exception: pass
                try: _bases.append(Path(__file__).parent / "assets")
                except Exception: pass
                _exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic")
                for _base in _bases:
                    for _ext in _exts:
                        _c = Path(_base) / ("Easy3D" + _ext)
                        try:
                            if _c.exists(): _bg = _c; break
                        except Exception: pass
                    if _bg is not None: break
                print(f"[Preview] empty file list, background={_bg}", flush=True)
                if _bg is not None: self.show_preview(_bg)
                else: self.preview_pil = None; self.preview_label.configure(image="", text="No Image")
        except Exception as e: print(f"[FileList] {e}")
    def show_preview(self, path):
        try:
            from PIL import Image
            img = Image.open(path)
            # FIX v38: 透明 PNG 合成到白底, 不然 convert("RGB") 会把透明变黑, 看起来像没图
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in getattr(img, "info", {})):
                _rgba = img.convert("RGBA")
                _bg = Image.new("RGB", _rgba.size, (255, 255, 255))
                _bg.paste(_rgba, mask=_rgba.split()[3])
                img = _bg
            else:
                img = img.convert("RGB")
            self.preview_pil = img
            self._refresh_preview_scaled()
        except Exception as e: self.preview_label.configure(text=f"No preview\n{e}")
    def _refresh_preview_scaled(self, event=None):
        # Preview follows window: scale stored full-res PIL to preview_label's current size, keep aspect.
        try:
            if getattr(self, "preview_pil", None) is None: return
            from PIL import Image, ImageTk
            w = self.preview_label.winfo_width(); h = self.preview_label.winfo_height()
            if w <= 10 or h <= 10: return
            # Leave a small padding so the image never touches the frame border
            w = max(10, w - 8); h = max(10, h - 8)
            img = self.preview_pil
            scale = min(w / img.width, h / img.height)
            if scale <= 0: return
            nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            img2 = img.resize((nw, nh), Image.LANCZOS)
            self.preview_tk = ImageTk.PhotoImage(img2); self.preview_label.configure(image=self.preview_tk, text="")
        except Exception: pass
    def on_file_list_select(self, event=None):
        try:
            sel = self.file_listbox.curselection()
            if not sel: return
            name = self.file_listbox.get(sel[0]); inp = Path(self.input_var.get().strip().strip('"'))
            p = inp / name if inp.is_dir() else inp
            if p.exists(): self.show_preview(p)
        except Exception: pass
    def toggle_panels(self):
        try:
            self.log_frame.grid() if self.show_log_var.get() else self.log_frame.grid_remove()
            self.preview_frame.grid() if self.show_preview_var.get() else self.preview_frame.grid_remove()
            self.file_frame.grid() if self.show_filelist_var.get() else self.file_frame.grid_remove()
            self.save_config()
        except Exception as e: print(f"[Panels] {e}")
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        try: self.log_text.insert(tk.END, line); self.log_text.see(tk.END)
        except Exception: pass
        try:
            with open(LOG_FILE, "a", encoding="utf-8", errors="ignore") as f: f.write(line)
        except Exception: pass
        print(line, end="")


    # ---- Run ----
    # ---- v1.9 Output naming + vis -> SBS ----
    def _param_code(self, value, prefix):
        try: v = float(value)
        except Exception: v = 0.0
        return f"{prefix}{int(round(v*100)):03d}"
    def output_stem(self, input_file, params, extra=""):
        stem = Path(input_file).stem
        if "_Easy3D_" in stem:
            stem = stem.split("_Easy3D_")[0]
        # v3.1: Finetuned choice is always respected (no patch forcing v2.1).
        model = "V15" if "v1.5" in str(params.get("finetuned_model", "")) else ("V21" if "v2.1" in str(params.get("finetuned_model", "")) else "V15")
        _mde = str(params.get("mde_model", "")).strip().lower()
        _mde_suf = MDE_SUFFIXES.get(_mde, "_MDEC" if _mde else "")
        ot_map = {"parallel": "PAR", "xd": "XD", "align_parallel": "APAR", "align_xd": "AXD", "align_both": "ABOTH"}
        ot = ot_map.get(str(params.get("output_type", "parallel")), "PAR")
        res = int(params.get("resolution", 512) or 512)
        _fr = "_FRBoth" if str(params.get("face_restore", "none")) == "gfpgan_both" else ""
        # FIX v24: Decoupled 出图加 _FaDe 后缀, Quality 不加, 保持老文件名不变
        # FIX v26: _FaDeA/B/C 查表
        _mode_suf = _mode_fade_suffix(params.get("mode", "quality"))
        return f"{stem}_Easy3D_{model}{_mde_suf}{_mode_suf}_R{res}_{self._param_code(params.get('baseline', 1.0), 'B')}_S{int(params.get('steps', 50))}_{self._param_code(params.get('guidance_scale', 1.5), 'G')}{_fr}_{ot}_Seed{params.get('seed', '')}{extra}"
    def _prepare_input(self, input_path, res):
        """
        Returns (temp_prep_path, (orig_W, orig_H)). Temp file is _cache_<stem>.jpg in ./_cache/."""
        from PIL import Image
        orig = Image.open(input_path).convert("RGB")
        # Gamma 1: 原图 -> 推理之前
        if globals().get("GAMMA_ENABLED", True):
            orig = _apply_gamma(orig, globals().get("GAMMA_PRE_SQUARE", 1.0))
        ow, oh = orig.size
        try:
            import shutil as _shutil
            _cache_dir = Path(__file__).parent / "_cache"
            _cache_dir.mkdir(exist_ok=True)
            _backup = _cache_dir / f"{Path(input_path).stem}_orig.jpg"
            _shutil.copy2(str(input_path), str(_backup))
            if not hasattr(self, "_orig_backup"): self._orig_backup = {}
            self._orig_backup[str(Path(input_path))] = _backup
        except Exception: pass
        tmp_dir = Path(__file__).parent / "_cache"
        tmp_dir.mkdir(exist_ok=True)
        if (ow, oh) == (res, res):
            # 尺寸已是 Res, 也不能直接返回原图, 否则 Gamma 1.6 会被跳过
            # 把 Gamma 后的图存到 _cache, 返回 _cache 路径
            if globals().get("GAMMA_ENABLED", True) and float(globals().get("GAMMA_PRE_SQUARE", 1.0)) != 1.0:
                tmp2 = tmp_dir / f"{Path(input_path).stem}_prep{res}.jpg"
                orig.save(tmp2, "JPEG", quality=95, subsampling=0, optimize=True)
                return tmp2, (ow, oh)
            return Path(input_path), (ow, oh)
        tmp = tmp_dir / f"{Path(input_path).stem}_prep{res}.jpg"
        # CANCELLED: no ml-sharp pre-shrink. GenStereo != ml-sharp. Plain resize only.
        _sq = orig.resize((int(res), int(res)), Image.LANCZOS)
        _sq.save(tmp, "JPEG", quality=95, subsampling=0, optimize=True)
        return tmp, (ow, oh)
    def _cleanup_vis_and_tmp(self, target, params=None):
        """v3.4: after SBS for ONE image is done, delete its vis/ folder + its _cache file. No leftovers."""
        import shutil
        _sq = getattr(self, "_prep_map", {}).get(str(Path(target)), None)
        _prep_stem = _sq[0] if _sq else Path(target).stem
        _vis = Path(__file__).parent / "vis" / _prep_stem
        _tmp = Path(__file__).parent / "_cache" / f"{Path(target).stem}_prep{int((params or {}).get('_eff_res', (params or {}).get('resolution', 512)) or 512)}.jpg"  # FIX v22: _prep 是有效 Res, 不是 GUI Res
        # FIX v17: 还要删 _orig.jpg 备份. _prepare_input 存了 _prep + _orig 两个, 清理只删了 _prep,
        # _orig 会越攒越多. SBS 合成完备份已没用, Phase 2 也不用它.
        _orig = Path(__file__).parent / "_cache" / f"{Path(target).stem}_orig.jpg"
        # Also try the exact tmp path from _prep_map if it was stored as a full path
        for _p, _kind in ((_vis, "vis"), (_tmp, "tmp_prep"), (_orig, "orig_backup")):
            try:
                if _p.is_dir(): shutil.rmtree(_p, ignore_errors=True)
                elif _p.is_file(): _p.unlink()
            except Exception as _e:
                self.log(f"[Cleanup] Could not delete {_kind} {_p}: {_e}")
        # FIX v17: 把 _orig_backup 里这张的记录也弹出, 否则字典一直长
        try:
            getattr(self, "_orig_backup", {}).pop(str(Path(target)), None)
        except Exception: pass
        # If _cache is now empty, remove the folder itself so it does not linger
        try:
            _td = Path(__file__).parent / "_cache"
            if _td.is_dir() and not any(_td.iterdir()): _td.rmdir()
        except Exception: pass

    def build_outputs_from_vis(self, targets, output_dir, params):
        """test.py -> vis/<stem>/{left,generated_right,warped,disp}.png
        1) create_sbs_image(left, right) -> temp SBS (no Easy3D name yet)
        2) process_sbs_file(temp, output_type) - output_type.py is authoritative, it deletes temp
        3) rename result(s) to _Easy3D_... .jpg Q95 in output_dir
        Depth/Warped: vis disp/warped (方形推理图) -> resize 到原图比例 -> _DEPTH/_WARPED .jpg Q95"""
        from PIL import Image
        try:
            from output_type import create_sbs_image as _create_sbs
        except Exception:
            _create_sbs = None
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        ot = str(params.get("output_type", "parallel"))
        done = []
        for t in targets:
            # v3.0: t may be the ORIGINAL path. vis dir uses the PREP temp stem, not the original stem.
            _sq = getattr(self, "_prep_map", {}).get(str(Path(t)), None)
            _prep_stem = _sq[0] if _sq else Path(t).stem
            _orig_size = _sq[1] if _sq else None
            stem = Path(t).stem
            vis = Path(__file__).parent / "vis" / _prep_stem
            left_p, right_p = vis / "left.png", vis / "generated_right.png"
            if not right_p.exists():
                # Fallback: older runs wrote vis/<orig_stem>/
                vis = Path(__file__).parent / "vis" / stem
                right_p = vis / "generated_right.png"
                if not right_p.exists():
                    self.log(f"[SBS] Missing vis files for {stem}: {vis}"); continue
            try:
                # v3.3 Gen Mode: which eye is AI?
                # orig_right: Left=Orig, Right=vis generated_right.png
                # orig_left:  Left=vis generated_left.png, Right=Orig
                # both:       Left=vis generated_left.png, Right=vis generated_right.png (orig is center)
                _gm = str(params.get("gen_mode", "orig_right"))
                try:
                    _orig = Image.open(t).convert("RGB")
                    # Gamma 1.6 全链路: 左眼/右眼的 Orig 也必须是 Gamma 后的图
                    # 否则 Left=原图(暗), Right=Gamma后生成的图(亮), 两眼亮度不一致
                    if globals().get("GAMMA_ENABLED", True):
                        _orig = _apply_gamma(_orig, globals().get("GAMMA_PRE_SQUARE", 1.0))
                except FileNotFoundError:
                    _backup = getattr(self, "_orig_backup", {}).get(str(Path(t)))
                    if _backup and Path(_backup).exists():
                        self.log(f"[SBS] Original deleted during run, using cached copy for {Path(t).name}")
                        _orig = Image.open(_backup).convert("RGB")
                        if globals().get("GAMMA_ENABLED", True):
                            _orig = _apply_gamma(_orig, globals().get("GAMMA_PRE_SQUARE", 1.0))
                    else:
                        self.log(f"[SBS] Original missing and no backup, skip {Path(t).name}")
                        continue
                if _orig_size is not None:
                    _orig = _orig.resize(_orig_size, Image.LANCZOS)
                left = _orig; right = _orig
                if _gm == "orig_left":
                    _lp = vis / "generated_left.png"
                    if not _lp.exists():
                        self.log(f"[SBS] Missing generated_left.png for {stem}: {vis} (gen_mode=orig_left)"); continue
                    left = Image.open(_lp).convert("RGB")
                    if left.size != _orig.size: left = left.resize(_orig.size, Image.LANCZOS)
                    right = _orig
                elif _gm == "both":
                    _lp, _rp = vis / "generated_left.png", vis / "generated_right.png"
                    if not _lp.exists() or not _rp.exists():
                        self.log(f"[SBS] Missing generated_left/right for {stem}: {vis} (gen_mode=both)"); continue
                    left = Image.open(_lp).convert("RGB"); right = Image.open(_rp).convert("RGB")
                    if left.size != _orig.size: left = left.resize(_orig.size, Image.LANCZOS)
                    if right.size != _orig.size: right = right.resize(_orig.size, Image.LANCZOS)
                else:
                    left = _orig
                    right = None
                    for _try in range(3):
                        try:
                            right = Image.open(right_p).convert("RGB"); break
                        except OSError as _oe:
                            if "truncated" in str(_oe).lower() and _try < 2:
                                import time as _time2; _time2.sleep(1.0); continue
                            raise
                    if right.size != left.size: right = right.resize(left.size, Image.LANCZOS)
                # CANCELLED: no ml-sharp post-expand / crop. GenStereo != ml-sharp. Left/right as-is.
                # Edge crop is small, Auto-Align will crop anyway.
                pass
                # Temp SBS: plain Left|Right, let output_type.py decide swap/align
                if _create_sbs is not None:
                    sbs_img = _create_sbs(left, right, output_type="parallel")
                else:
                    sbs_img = Image.new("RGB", (left.width + right.width, left.height)); sbs_img.paste(left, (0, 0)); sbs_img.paste(right, (left.width, 0))
                # v6.8Face Restore: AFTER SBS merge, BEFORE Output Type (output_type.py).
                # One call on the whole SBS image - GFPGAN sees both eyes' faces at once (Both Eyes).
                _defer = bool(globals().get("DEFER_FACE_RESTORE", False)) and str(params.get("face_restore", "none")) == "gfpgan_both"
                if str(params.get("face_restore", "none")) == "gfpgan_both" and not _defer:
                    try:
                        from GFPGAN_face import init_face_restorer, restore_face
                        if not getattr(self, "_face_restorer_ready", False):
                            self.log("[FaceRestore] Loading GFPGAN (first use, ~10s)...")
                            _ok = init_face_restorer(restore_type="gfpgan")
                            self._face_restorer_ready = bool(_ok)
                            self.log(f"[FaceRestore] GFPGAN loaded: {_ok}")
                        if getattr(self, "_face_restorer_ready", False):
                            self.log(f"[FaceRestore] Restoring faces on SBS (Both Eyes) for {stem} ...")
                            sbs_img = restore_face(sbs_img, blend=1.0, feather=15)
                        else:
                            self.log("[FaceRestore] GFPGAN not available, skipping (SBS unchanged)")
                    except Exception as _fre:
                        self.log(f"[FaceRestore] Failed, skipping (SBS unchanged): {_fre}")
                # Gamma 1.4: SBS 合成后 -> Output Type 之前, 对整张 SBS, 左右眼一起调, 亮度才一致
                # 位置: FaceRestore 之后, Output Type 之前 (FaceRestore 看的是 Gamma 0.7 版, 最后统一压回 1.4)
                if not _defer and globals().get("GAMMA_ENABLED", True):
                    sbs_img = _apply_gamma(sbs_img, globals().get("GAMMA_PRE_OUTPUT", 1.0))
                tmp = out / f"_tmp_sbs_{stem}.jpg"
                sbs_img.save(tmp, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True)
                # v6.8 DEFER: do NOT enter Output Type yet. Move SBS to _deferred/, Phase 2 handles it.
                # tmp is MOVED, not copied, so no _tmp_sbs duplicate is left behind.
                if _defer:
                    _def_dir = out / "_deferred"; _def_dir.mkdir(parents=True, exist_ok=True)
                    _def_dst = _def_dir / f"{stem}_SBS.jpg"
                    import shutil as _sh2; _sh2.move(str(tmp), str(_def_dst))
                    self.log(f"[Defer] {stem} SBS saved to _deferred/, Phase 2 later: {_def_dst.name}")
                    done.append(_def_dst)
                    results = []  # rename loop below runs empty; Depth/Warped below still runs
                else:
                    # output_type.py handles parallel/xd/align_*/both and deletes tmp
                    results = None
                    if _process_sbs_file is not None:
                        results = _process_sbs_file(tmp, ot, log_func=lambda m: self.log(m))
                    else:
                        results = tmp
                # Normalize to list of paths
                if results is None:
                    self.log(f"[SBS] output_type returned None for {stem}, using temp SBS"); results = [tmp] if tmp.exists() else []
                elif isinstance(results, (str, Path)):
                    results = [Path(results)]
                else:
                    results = [Path(r) for r in results]
                # Rename each result to Easy3D naming
                for idx, r in enumerate(results):
                    if not Path(r).exists(): continue
                    # output_type adds _XD / _Align / _Align_XD suffix - detect it to pick OT label
                    rn = Path(r).name
                    if "_Align_XD" in rn: ot_label = "AXD"
                    elif "_Align" in rn: ot_label = "APAR" if "both" not in ot else ("APAR" if idx == 0 else "AXD")
                    elif rn.endswith("_XD.jpg") or "_XD" in rn: ot_label = "XD"
                    else: ot_label = {"parallel": "PAR", "xd": "XD", "align_parallel": "APAR", "align_xd": "AXD", "align_both": "APAR" if idx == 0 else "AXD"}.get(ot, "PAR")
                    base = self.output_stem(str(t), params, extra="")
                    import re as _re
                    base2 = _re.sub(r"_(PAR|XD|APAR|AXD|ABOTH)_Seed", f"_{ot_label}_Seed", base)
                    dst = out / (base2 + ".jpg")
                    # Re-save Q95 (output_type already saved Q95, but ensure format)
                    try:
                        im = Image.open(r).convert("RGB"); im.save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True)
                        if Path(r).resolve() != dst.resolve():
                            try: Path(r).unlink()
                            except Exception: pass
                    except Exception:
                        try:
                            import shutil; shutil.move(str(r), str(dst))
                        except Exception as _e2: self.log(f"[SBS] rename failed {r}: {_e2}"); continue
                    done.append(dst); self.log(f"[Output] {dst.name}")
                # Depth / Warped (only if checked)
                if params.get("save_depth"):
                    disp = vis / "disp.png"
                    if disp.exists():
                        d = Image.open(disp).convert("RGB")
                        # v7.0: vis/disp.png 是方形推理图, 改回原图比例再给用户看
                        if d.size != _orig.size: d = d.resize(_orig.size, Image.LANCZOS)
                        dst = out / (self.output_stem(str(t), params, extra="_DEPTH") + ".jpg")
                        d.save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True); self.log(f"[Output] {dst.name} ({d.size[0]}x{d.size[1]})")
                if params.get("save_warped"):
                    wp = vis / "warped.png"
                    if wp.exists():
                        w = Image.open(wp).convert("RGB")
                        # v7.0: vis/warped.png 是方形推理图, 改回原图比例再给用户看
                        if w.size != _orig.size: w = w.resize(_orig.size, Image.LANCZOS)
                                                # v7.0: WARPED 补最后那次 Gamma 1.40, 跟最终 SBS 一样亮 (vis 只有 0.70, 偏亮)
                        if globals().get("GAMMA_ENABLED", True):
                            w = _apply_gamma(w, globals().get("GAMMA_PRE_OUTPUT", 1.0))
                        dst = out / (self.output_stem(str(t), params, extra="_WARPED") + ".jpg")
                        w.save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True); self.log(f"[Output] {dst.name} ({w.size[0]}x{w.size[1]})")
            except Exception as e:
                self.log(f"[SBS] Failed {stem}: {e}")
                import traceback; self.log(traceback.format_exc())
        return done

    def run_phase2_deferred(self, output_dir, params):
        """v6.8 Phase 2: _deferred/*_SBS.jpg -> Gamma 1.40 -> FaceRestore -> Output Type -> Easy3D name."""
        from PIL import Image
        out = Path(output_dir); def_dir = out / "_deferred"
        if not def_dir.exists(): return []
        deferred = sorted(def_dir.glob("*_SBS.jpg"))
        if not deferred: return []
        self.log(f"[Defer] Phase 2 started: {len(deferred)} SBS -> Gamma -> FaceRestore -> Output Type")
        ot = str(params.get("output_type", "parallel")); done = []
        face_ready = False; restore_fn = None
        if str(params.get("face_restore", "none")) == "gfpgan_both":
            try:
                from GFPGAN_face import init_face_restorer, restore_face
                self.log("[FaceRestore] Loading GFPGAN (Phase 2, once)...")
                face_ready = bool(init_face_restorer(restore_type="gfpgan")); restore_fn = restore_face
                self.log(f"[FaceRestore] GFPGAN loaded: {face_ready}")
            except Exception as e:
                self.log(f"[FaceRestore] Failed, Phase 2 continues without it: {e}")
        for src_path in deferred:
            stem = src_path.name[:-len("_SBS.jpg")]
            try:
                sbs_img = Image.open(src_path).convert("RGB")
                if globals().get("GAMMA_ENABLED", True):
                    sbs_img = _apply_gamma(sbs_img, globals().get("GAMMA_PRE_OUTPUT", 1.0))
                if face_ready and restore_fn is not None:
                    try: sbs_img = restore_fn(sbs_img, blend=1.0, feather=15)
                    except Exception as e: self.log(f"[FaceRestore] Failed for {stem}: {e}")
                tmp = out / f"_tmp_sbs_{stem}.jpg"
                sbs_img.save(tmp, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True)
                results = _process_sbs_file(tmp, ot, log_func=lambda m: self.log(m)) if _process_sbs_file is not None else tmp
                if isinstance(results, (str, Path)): results = [Path(results)]
                elif results is None: results = [tmp] if tmp.exists() else []
                else: results = [Path(r) for r in results]
                made = []
                for idx, r in enumerate(results):
                    if not Path(r).exists(): continue
                    rn = Path(r).name
                    if "_Align_XD" in rn: ot_label = "AXD"
                    elif "_Align" in rn: ot_label = "APAR" if "both" not in ot else ("APAR" if idx == 0 else "AXD")
                    elif rn.endswith("_XD.jpg") or "_XD" in rn: ot_label = "XD"
                    else: ot_label = {"parallel": "PAR", "xd": "XD", "align_parallel": "APAR", "align_xd": "AXD", "align_both": "APAR" if idx == 0 else "AXD"}.get(ot, "PAR")
                    import re as _re
                    base = self.output_stem(stem + ".jpg", params, extra="")  # v6.8fix: stem already bare, +".jpg" stops Path().stem truncating "my.photo" -> "my"
                    base2 = _re.sub(r"_(PAR|XD|APAR|AXD|ABOTH)_Seed", f"_{ot_label}_Seed", base)
                    dst = out / (base2 + ".jpg")
                    Image.open(r).convert("RGB").save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True)
                    if Path(r).resolve() != dst.resolve():
                        try: Path(r).unlink()
                        except Exception: pass
                    made.append(dst); done.append(dst); self.log(f"[Output] {dst.name}")
                if made and all(d.exists() and d.stat().st_size > 0 for d in made):
                    try: src_path.unlink()
                    except Exception: pass
                    _dup = out / f"_tmp_sbs_{stem}.jpg"
                    try:
                        if _dup.exists(): _dup.unlink()
                    except Exception: pass
            except Exception as e:
                self.log(f"[Defer] Phase 2 failed for {stem}: {e}")
                import traceback; self.log(traceback.format_exc())
        try:
            if def_dir.exists() and not any(def_dir.iterdir()):
                def_dir.rmdir(); self.log("[Defer] _deferred/ cleaned (empty)")
        except Exception: pass
        self.log(f"[Defer] Phase 2 finished: {len(done)} final file(s)")
        return done

    def build_command(self, params, input_path, output_dir):
        """EDIT HERE if your test.py accepts extra flags.
        Default: official CLI (input path only) + JSON + env vars."""
        exe = sys.executable
        for cand in (Path("venv_Easy3D/Scripts/python.exe"), Path("venv/Scripts/python.exe")):
            if cand.exists(): exe = str(cand); break
        # Always test.py single-pass, NO patch. test.py runs UNet at Res (cfg.height/width = Res).
        scale = 0.15 * float(params.get("baseline", 1.0)) * float(params.get("depth_strength", 1.0))
        # v3.2: input_path can be a single prepared OR a list of prepareds (batch, load model once).
        _inputs = [str(x) for x in input_path] if isinstance(input_path, (list, tuple)) else [str(input_path)]
        # FIX v22: 有效 Res 已在 run_thread 里算好, 存在 params["_eff_res"] / params["_orig_long"] (批量按最小原图)
        _eff_res = int(params.get("_eff_res", params.get("resolution", 512)) or 512)
        _orig_long = int(params.get("_orig_long", 0) or 0)
        cmd = [exe, "-u", SCRIPT_NAME, *_inputs,
               "--output", "./vis",
               "--scale_factor", str(round(scale, 5)),
               "--steps", str(int(params.get("steps", 50))),
               "--guidance", str(float(params.get("guidance_scale", 1.5))),
               "--fusion_strength", str(float(params.get("fusion_strength", 1.0))),
               "--seed", str(params.get("seed", "-1")),
               "--gen_mode", str(params.get("gen_mode", "orig_right")),
               "--image_size", str(_eff_res),
               "--mode", str(params.get("mode", "quality")),
               "--orig_long", str(_orig_long),
               "--limit_to_source", "1" if params.get("limit_to_source", True) else "0"]
        return cmd, exe
    def start_conversion(self):
        input_path = self.input_var.get().strip().strip('"')
        if not input_path: messagebox.showwarning("Input", "Select an input image or folder"); return
        if not Path(input_path).exists(): messagebox.showerror("Input", f"Not found:\n{input_path}"); return
        output_dir = self.get_output_dir(input_path)
        if output_dir is None: return
        params = self.collect_params(); self.save_config()
        self.log(f"Params: {params}")
        _inp0 = Path(input_path)
        if _inp0.is_file(): _targets0 = [_inp0]
        elif _inp0.is_dir(): _targets0 = [x for x in sorted(_inp0.iterdir()) if x.suffix.lower() in SUPPORTED_INPUT_EXTS and x.name.lower() not in {"easy3d.png", "easy3d.ico"}]
        else: _targets0 = []
        if not _targets0:
            messagebox.showwarning("Input", "No convertible images found."); return
        if len(_targets0) > 1: self.log(f"[Batch] {len(_targets0)} images")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "easy3d_gui_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        self.stop_flag = False; self.pause_flag = False
        self.start_btn.configure(state=tk.DISABLED); self.suspend_btn.configure(state=tk.NORMAL, text="Suspend"); self.cancel_btn.configure(state=tk.NORMAL)
        self._clear_done(); self._start_running_anim("Running"); self.progress_var.set(0)
        def run_thread():
            try:
                env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
                for k, v in params.items(): env[f"GENSTEREO_{str(k).upper()}"] = str(v)
                env["GENSTEREO_FUSION_STRENGTH"] = str(float(params.get("fusion_strength", 1.0)))  # FIX v28
                env["GENSTEREO_MODE"] = str(params.get("mode", "quality"))
                env["GENSTEREO_LIMIT_TO_SOURCE"] = "1" if params.get("limit_to_source", True) else "0"  # FIX v22: Mode/Limit env
                _fm = str(params.get("finetuned_model", "genstereo-v1.5"))
                env["GENSTEREO_VERSION"] = _fm.replace("genstereo-", "").replace("genstereo", "v1.5") if "genstereo" in _fm else _fm
                _res_rt = int(params.get("resolution", 512) or 512)
                self.log(f"[Work] Resolution {_res_rt} -> test.py, Mode={params.get('mode', 'quality')}, LimitToSource={'ON' if params.get('limit_to_source', True) else 'OFF'}, Finetuned={_fm}")  # FIX v22: Mode + Limit
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                _rc = 0
                _chunk = int(globals().get("BATCH_CHUNK_SIZE", 1) or 0)
                _chunks = [_targets0] if _chunk <= 0 else [_targets0[i:i+_chunk] for i in range(0, len(_targets0), _chunk)]
                _cooldown = float(globals().get("BATCH_COOLDOWN_SEC", 5) or 0)
                if len(_chunks) > 1:
                    self.log(f"[Batch] Auto-Chunk: {len(_targets0)} 张 -> {len(_chunks)} 批, 每批 {_chunk} 张, 间隔 {_cooldown:.0f}s")
                _total_done = 0
                for _ci, _chunk_targets in enumerate(_chunks):
                    if self.stop_flag: break
                    if _ci > 0 and _cooldown > 0:
                        self.log(f"[Batch] 歇 {_cooldown:.0f}s 降温...")
                        _slept = 0.0
                        while _slept < _cooldown and not self.stop_flag:
                            _step = min(0.2, _cooldown - _slept); time.sleep(_step); _slept += _step
                            while self.pause_flag and not self.stop_flag: time.sleep(0.2)
                        if self.stop_flag: break
                    self.log(f"[Batch] 批 {_ci+1}/{len(_chunks)}: {len(_chunk_targets)} 张")
                    self._prep_map = {}
                    _sq_paths = []
                    # FIX v22: Limit to Source (本批按最小原图算). 先读每张原图长边, 再算有效 Res.
                    _orig_longs = []
                    for _t in _chunk_targets:
                        try:
                            from PIL import Image as _PILImage
                            _o = _PILImage.open(_t); _orig_longs.append(max(_o.size))
                        except Exception:
                            _orig_longs.append(0)
                    _min_orig = min([x for x in _orig_longs if x] or [0])
                    _limit_on = bool(params.get("limit_to_source", True))
                    _eff_res = _limit_to_source_gui(_res_rt, _min_orig, _limit_on) if _min_orig else _res_rt
                    if _min_orig >= 768 and _eff_res < _res_rt:
                        self.log(f"[Limit] Orig min {_min_orig}px, Res {_res_rt} -> {_eff_res} (Limit to Source, orig>=768)")
                    elif _min_orig and _min_orig < 768:
                        self.log(f"[Limit] Orig min {_min_orig}px <768, no cap, Res {_res_rt} (small orig, do not shrink)")
                    _params_chunk = dict(params); _params_chunk["_eff_res"] = int(_eff_res); _params_chunk["_orig_long"] = int(_min_orig)
                    for _t in _chunk_targets:
                        _sq_path, _orig_sz = self._prepare_input(_t, int(_eff_res))
                        self._prep_map[str(Path(_t))] = (_sq_path.stem, _orig_sz)
                        _sq_paths.append(str(_sq_path))
                    _cmd, _ = self.build_command(_params_chunk, _sq_paths, output_dir)
                    self.log(f"Command: {' '.join(_cmd)}")
                    self.process = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore", bufsize=1, creationflags=creationflags, cwd=str(Path(__file__).parent), env=env)
                    _pending = list(_chunk_targets); _done_set = set(); _watcher_stop = {"v": False}; _vis_stat = {}
                    def _vis_ready(_t):
                        _sq = self._prep_map.get(str(Path(_t)), None)
                        if not _sq: return False
                        _vis = Path(__file__).parent / "vis" / _sq[0]
                        _names = ["left.png", "generated_right.png", "disp.png", "warped.png"]
                        if str(params.get("gen_mode", "orig_right")) in ("both", "orig_left"): _names.append("generated_left.png")
                        _stat = []
                        for _n in _names:
                            _p = _vis / _n
                            if not _p.exists(): return False
                            try: _st = _p.stat()
                            except Exception: return False
                            if _st.st_size == 0: return False
                            _stat.append((_n, _st.st_size, _st.st_mtime))
                        _key = str(_t); _prev = _vis_stat.get(_key); _vis_stat[_key] = tuple(_stat)
                        return _prev == tuple(_stat)
                    def _watcher():
                        import time as _time
                        while not _watcher_stop["v"]:
                            for _t in list(_pending):
                                if self.stop_flag: return
                                try:
                                    if _vis_ready(_t):
                                        _built = []
                                        try: _built = self.build_outputs_from_vis([_t], output_dir, params) or []
                                        except Exception as _fe: self.log(f"[SBS] {_fe}")
                                        if not _built: continue
                                        self._cleanup_vis_and_tmp(_t, params)
                                        _pending.remove(_t); _done_set.add(str(_t))
                                        _n = _total_done + len(_done_set)
                                        self.root.after(0, lambda _n=_n: (self.progress_var.set(_n * 100.0 / max(1, len(_targets0))), self.refresh_file_list_and_preview()))
                                        self.log(f"[Stream] {_t.name} done ({_n}/{len(_targets0)})")
                                except Exception as _we: self.log(f"[Stream] {_we}")
                            _time.sleep(1.5)
                    _wt = threading.Thread(target=_watcher, daemon=True); _wt.start()
                    for line in self.process.stdout:
                        if self.stop_flag: break
                        while self.pause_flag and not self.stop_flag: time.sleep(0.2)
                        self.log(line.rstrip())
                    self.process.wait(); _rc = self.process.returncode or 0
                    _watcher_stop["v"] = True
                    try: _wt.join(timeout=3.0)
                    except Exception: pass
                    for _t in list(_pending):
                        # v6.8fix: test.py 崩了就不进 Final sweep，直接报错，不要假装 done
                        if _rc != 0:
                            self.log(f"[Error] test.py crashed (code {_rc}), {_t.name} 没出图，请看上面 Traceback (多半是缺模型文件)")
                            continue
                        try:
                            _ready = False
                            for _retry in range(3):
                                if _vis_ready(_t):
                                    _ready = True
                                    break
                                time.sleep(1.5)
                            if not _ready:
                                self.log(f"[Error] {_t.name} 没出图: vis 里没找到完整文件，跳过，不记为 done")
                                continue
                            _built = []
                            try: _built = self.build_outputs_from_vis([_t], output_dir, params) or []
                            except Exception as _fe: self.log(f"[SBS] {_fe}")
                            if not _built:
                                self.log(f"[Error] {_t.name} SBS 合成失败，不记为 done")
                                continue
                        except Exception: pass
                        else:
                            self._cleanup_vis_and_tmp(_t, params); _pending.remove(_t); _done_set.add(str(_t))
                            _n2 = _total_done + len(_done_set)
                            self.root.after(0, lambda _n=_n2: (self.progress_var.set(_n * 100.0 / max(1, len(_targets0))), self.refresh_file_list_and_preview()))
                            self.log(f"[Stream] {_t.name} done ({_n2}/{len(_targets0)}) via Final sweep")
                    _total_done += len(_done_set)
                    self.log(f"[Batch] 批 {_ci+1}/{len(_chunks)} 完成, 累计 {_total_done}/{len(_targets0)}")
                    self.root.after(0, lambda _n=_total_done: (self.progress_var.set(_n * 100.0 / max(1, len(_targets0))), self._set_running_base(f"Running {_total_done}/{len(_targets0)}"), self.refresh_file_list_and_preview()))
                    if self.stop_flag: break
                # v6.8 Phase 2: all diffusion done -> FaceRestore + Output Type
                # v6.8fix: skip Phase 2 if user cancelled (else cancelled batch still gets face-restored)
                if not self.stop_flag:
                    try: self.run_phase2_deferred(output_dir, params)
                    except Exception as _p2e: self.log(f"[Defer] Phase 2 error: {_p2e}")
                else:
                    self.log("[Defer] Phase 2 skipped (cancelled)")
                class _P: returncode = _rc
                self.process = _P()
                if self.stop_flag: self.root.after(0, lambda: (self._clear_done(), self._set_status_final("Cancelled"), self.progress_var.set(0)))
                elif self.process.returncode == 0:
                    self.root.after(0, lambda: (self._set_done(output_dir), self.progress_var.set(100)))
                    self.root.after(0, self.refresh_file_list_and_preview)
                else: self.root.after(0, lambda: self._set_status_final(f"Finished code {self.process.returncode}"))
            except Exception as e:
                self.log(f"[Exception] {e}")
            finally:
                self.process = None
                self.root.after(0, lambda: self.start_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.suspend_btn.configure(state=tk.DISABLED, text="Suspend"))
                self.root.after(0, lambda: self.cancel_btn.configure(state=tk.DISABLED))
        threading.Thread(target=run_thread, daemon=True).start()
    # --- v7.0 Running 打字机动画: R -> Ru -> ... -> Running... -> R, 循环 ---
    # status_var 不能直接 set("Running..."), 否则定死不动, 用户以为卡了。
    # 所有“还在跑”的地方都走 _start_running_anim() / _set_running_base(), 动画自己每 250ms 走一格。
    _running_anim_after = None
    _running_anim_base = "Running"  # 不带点的本体, 如 "Running" 或 "Running 1/5"
    _running_anim_step = 0
    def _start_running_anim(self, base="Running"):
        """开始跑: 重置动画, 从 R 开始走。base 如 "Running" 或 "Running 1/5"。"""
        self._stop_running_anim()
        self._running_anim_base = base
        self._running_anim_step = 0
        self._tick_running_anim()
    def _set_running_base(self, base):
        """还在跑, 只换计数 (如 1/5 -> 2/5), 动画不停, 不从头开始。"""
        self._running_anim_base = base
        # 下一格 tick 就会用新 base, 不用立刻 set, 免得跟 tick 打架
    def _tick_running_anim(self):
        # "Running" 的前缀 + 点: R, Ru, Run, ..., Running, Running., Running.., Running..., 回到 R
        _word = "Running"
        _frames = ["R", "Ru", "Run", "Runn", "Runni", "Runnin", "Running", "Running.", "Running..", "Running..."]
        _frame = _frames[self._running_anim_step % len(_frames)]
        # base 里 "Running" 后面的部分就是计数, 如 " 1/5", 跟着帧一起走: "R 1/5" -> ... -> "Running... 1/5"
        _suffix = self._running_anim_base[len(_word):] if self._running_anim_base.startswith(_word) else ""
        try:
            self.status_var.set(_frame + _suffix)
        except Exception:
            return
        self._running_anim_step += 1
        try:
            self._running_anim_after = self.root.after(250, self._tick_running_anim)
        except Exception:
            self._running_anim_after = None
    def _stop_running_anim(self):
        """停动画: Done / Cancelled / Suspended / Finished 时调, 之后 status_var 由调用方定死。"""
        try:
            if getattr(self, "_running_anim_after", None) is not None:
                self.root.after_cancel(self._running_anim_after)
        except Exception:
            pass
        self._running_anim_after = None
    def _set_status_final(self, text):
        """最终状态: 先停动画, 再定死显示 (Done / Cancelled / Suspended / Finished 都走这里)。"""
        self._stop_running_anim()
        self.status_var.set(text)
    # --- v7.0 Done 可点开文件夹: 点 Done - 地址 -> 打开文件夹, 悬浮提示 ---
    def _set_done(self, output_dir):
        """转换完成: 记下文件夹, 状态变成可点的 Done - 地址。"""
        # 如果有 Running 动画 (已跑过 Running patch 的版), 先停动画, 否则动画会把 Done 顶掉
        try:
            if hasattr(self, "_set_status_final"):
                self._set_status_final(f"Done - {output_dir}")
            else:
                self.status_var.set(f"Done - {output_dir}")
        except Exception:
            self.status_var.set(f"Done - {output_dir}")
        self._done_dir = str(output_dir)
        try:
            self.status_label.configure(cursor="hand2", foreground="#FFE87A")
        except Exception:
            try: self.status_label.configure(cursor="hand2")
            except Exception: pass
    def _clear_done(self):
        """不是 Done 时: 清掉可点状态, 鼠标变回箭头, 点了没反应。"""
        self._done_dir = None
        self._hide_status_tip()
        try:
            self.status_label.configure(cursor="", foreground="")
        except Exception:
            pass
    def _open_done_folder(self, event=None):
        """点 Done - 地址: 真的打开文件夹。不是 Done 时点了没反应。"""
        if not getattr(self, "_done_dir", None):
            return
        _d = Path(self._done_dir)
        if not _d.exists():
            try:
                from tkinter import messagebox as _mb
                _mb.showwarning("Folder", f"Folder not found:\n{_d}")
            except Exception:
                pass
            return
        try:
            import os as _os, subprocess as _sp, sys as _sys
            if _sys.platform == "win32":
                _os.startfile(str(_d))
            elif _sys.platform == "darwin":
                _sp.Popen(["open", str(_d)])
            else:
                _sp.Popen(["xdg-open", str(_d)])
            self.log(f"[Open] { _d}")
        except Exception as _e:
            self.log(f"[Open] Failed: {_e}")
    def _show_status_tip(self, event=None):
        """悬浮在 Done - 地址上: 提示“点一下打开文件夹 + 地址”。不是 Done 时不显示。"""
        if not getattr(self, "_done_dir", None):
            return
        self._hide_status_tip()
        try:
            _tip = tk.Toplevel(self.root)
            _tip.wm_overrideredirect(True)
            _tip.wm_attributes("-topmost", True)
            _label = tk.Label(_tip, text=f"Click to open folder:\n{self._done_dir}", justify="left",
                              background="#ffffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
            _label.pack(ipadx=6, ipady=4)
            _x = self.root.winfo_pointerx() + 12
            _y = self.root.winfo_pointery() + 12
            _tip.wm_geometry(f"+{_x}+{_y}")
            self._status_tip = _tip
        except Exception:
            self._status_tip = None
    def _hide_status_tip(self, event=None):
        try:
            if getattr(self, "_status_tip", None) is not None:
                self._status_tip.destroy()
        except Exception:
            pass
        self._status_tip = None
    def _show_about(self):
        """About 弹窗 (v7.0 fix v3: 右侧背景图 = 专门的 Easy3D_s.png)."""
        try:
            _win = tk.Toplevel(self.root)
            _win.title("About Easy3D")
            _win.geometry("560x320")
            _win.resizable(False, False)
            try:
                _win.transient(self.root)
                _win.grab_set()
            except Exception:
                pass
            try:
                _icon = None
                for _cand in (Path(__file__).parent / "images" / "Easy3D.ico", Path(__file__).parent / "Easy3D.ico"):
                    if _cand.exists():
                        _icon = _cand; break
                if _icon is not None:
                    _win.iconbitmap(str(_icon))
                else:
                    _png = None
                    for _cand in (Path(__file__).parent / "images" / "Easy3D.png", Path(__file__).parent / "Easy3D.png"):
                        if _cand.exists():
                            _png = _cand; break
                    if _png is not None:
                        from PIL import Image as _PILImg, ImageTk as _PILImgTk
                        _im = _PILImg.open(_png).convert("RGBA")
                        _im.thumbnail((64, 64), _PILImg.LANCZOS)
                        self._about_icon_tk = _PILImgTk.PhotoImage(_im)
                        _win.iconphoto(True, self._about_icon_tk)
            except Exception as _ie:
                print(f"[About Icon] {_ie}")
            _BG = "#23272E"
            _TEXT = "#E8EAED"
            _MUTED = "#9AA0A8"
            _LINK = "#6FB3FF"
            _win.configure(bg=_BG)
            _body = tk.Frame(_win, bg=_BG)
            _body.pack(fill=tk.BOTH, expand=True, padx=28, pady=(22, 10))
            _left = tk.Frame(_body, bg=_BG)
            _left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(_left, text="Easy3D", font=("Segoe UI", 20, "bold"), bg=_BG, fg=_TEXT).pack(anchor=tk.W, pady=(0, 10))
            tk.Label(_left, text="It is not only a 3D converter,", font=("Segoe UI", 11), bg=_BG, fg=_TEXT).pack(anchor=tk.W)
            tk.Label(_left, text="but also a 3D creator.", font=("Segoe UI", 11), bg=_BG, fg=_TEXT).pack(anchor=tk.W, pady=(0, 14))
            tk.Label(_left, text="Written by Tony Lin.", font=("Segoe UI", 10), bg=_BG, fg=_MUTED).pack(anchor=tk.W, pady=(0, 16))
            tk.Frame(_left, bg="#3A3F47", height=1).pack(fill=tk.X, pady=(0, 16))
            _links = tk.Frame(_left, bg=_BG)
            _links.pack(anchor=tk.W, pady=(0, 16))
            def _open(url):
                try:
                    import webbrowser
                    webbrowser.open(url)
                except Exception as _e:
                    self.log(f"[About] Open failed: {_e}")
            _gh = tk.Label(_links, text="GitHub", fg=_LINK, bg=_BG, cursor="hand2", font=("Segoe UI", 10, "underline"))
            _gh.pack(side=tk.LEFT, padx=(0, 24))
            _gh.bind("<Button-1>", lambda e: _open(globals().get("ABOUT_GITHUB_URL", "https://github.com/KolaKater/Easy3D-Converter")))
            _fb = tk.Label(_links, text="Facebook", fg=_LINK, bg=_BG, cursor="hand2", font=("Segoe UI", 10, "underline"))
            _fb.pack(side=tk.LEFT)
            _fb.bind("<Button-1>", lambda e: _open(globals().get("ABOUT_FACEBOOK_URL", "https://www.facebook.com/100000242772138")))
            tk.Label(_left, text=__VERSION__, font=("Segoe UI", 8), bg=_BG, fg=_MUTED).pack(anchor=tk.W)
            try:
                _bg_path = None
                for _s in (Path(__file__).parent / "images" / "Easy3D_s.png", Path(__file__).parent / "Easy3D_s.png"):
                    if _s.exists():
                        _bg_path = _s; break
                if _bg_path is None:
                    for _base in (Path(__file__).parent / "images", Path(__file__).parent):
                        for _ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                            _c = Path(_base) / ("Easy3D" + _ext)
                            if _c.exists():
                                _bg_path = _c; break
                        if _bg_path is not None:
                            break
                if _bg_path is not None:
                    from PIL import Image as _PILImg2, ImageTk as _PILImgTk2
                    _img = _PILImg2.open(_bg_path)
                    if _img.mode in ("RGBA", "LA") or (_img.mode == "P" and "transparency" in getattr(_img, "info", {})):
                        _rgba = _img.convert("RGBA")
                        _bgw = _PILImg2.new("RGB", _rgba.size, (35, 39, 46))
                        _bgw.paste(_rgba, mask=_rgba.split()[3])
                        _img = _bgw
                    else:
                        _img = _img.convert("RGB")
                    _img.thumbnail((170, 210), _PILImg2.LANCZOS)
                    self._about_bg_tk = _PILImgTk2.PhotoImage(_img)
                    tk.Label(_body, image=self._about_bg_tk, bg=_BG).pack(side=tk.RIGHT, padx=(20, 0), anchor=tk.N)
            except Exception as _be:
                print(f"[About BG] {_be}")
            _foot = tk.Frame(_win, bg=_BG)
            _foot.pack(fill=tk.X, padx=28, pady=(0, 10))
            ttk.Button(_foot, text="Close", width=12, command=_win.destroy).pack(anchor=tk.CENTER)
        except Exception as _e:
            try:
                messagebox.showinfo("About Easy3D", "Easy3D\nIt is not only a 3D converter, but also a 3D creator.\nWritten by Tony Lin.")
            except Exception:
                pass


    def clear_input_and_filelist(self):
        """Clear: 清空输入地址 + File List + Preview, 不删文件, 不清参数。"""
        try:
            self.input_var.set("")
        except Exception:
            pass
        try:
            self.file_listbox.delete(0, tk.END)
        except Exception:
            pass
        try:
            self.preview_pil = None
            self.preview_tk = None
            self.preview_label.configure(image="", text="No Image")
        except Exception:
            pass
        try:
            self.refresh_file_list_and_preview()
        except Exception:
            pass
        try:
            self.save_config()
        except Exception:
            pass
        try:
            self.log("[Clear] Input + File List cleared")
        except Exception:
            pass
    def _apply_dark_force(self):
        """__init__ 最后走一遍所有控件, 按父控件强制指深色 style (前面被 clam 顶掉也能拉回来)。"""
        try:
            def _is_in_labelframe(w):
                try:
                    p = w.nametowidget(w.winfo_parent()) if w.winfo_parent() else None
                    # 往上找两层: Label -> Frame(io_r0) -> LabelFrame(io), Label 在 Frame 里, Frame 在 LabelFrame 里
                    for _ in range(3):
                        if p is None: return False
                        if p.winfo_class() == "TLabelframe": return True
                        p = p.nametowidget(p.winfo_parent()) if p.winfo_parent() else None
                    return False
                except Exception:
                    return False
            def _walk(w):
                try:
                    _cls = w.winfo_class()
                except Exception:
                    return
                try:
                    _in_panel = _is_in_labelframe(w)
                    if _cls == "TLabelframe":
                        w.configure(style="Dark.TLabelframe")
                    elif _cls == "TFrame":
                        w.configure(style="Dark.TFrame" if _in_panel else "TFrame")
                    elif _cls == "TLabel":
                        # status 那行是 ttk.Label 但 style 是 Small.TLabel, 别动它 (Done patch 管它)
                        try:
                            _cur = str(w.cget("style") or "")
                        except Exception:
                            _cur = ""
                        if _cur not in ("Small.TLabel", "Header.TLabel"):
                            w.configure(style="Dark.TLabel" if _in_panel else "TLabel")
                    elif _cls == "TEntry":
                        w.configure(style="Dark.TEntry")
                    elif _cls == "TCombobox":
                        w.configure(style="Dark.TCombobox")
                    elif _cls == "TCheckbutton":
                        w.configure(style="Dark.TCheckbutton" if _in_panel else "TCheckbutton")
                    elif _cls == "TRadiobutton":
                        w.configure(style="Dark.TRadiobutton" if _in_panel else "TRadiobutton")
                    elif _cls == "TScale":
                        w.configure(style="Dark.Horizontal.TScale")
                    elif _cls == "TButton":
                        pass  # 按钮已是深色, 不动
                except Exception:
                    pass
                try:
                    for _c in w.winfo_children():
                        _walk(_c)
                except Exception:
                    pass
            _walk(self.root)
            try:
                self.log_text.configure(bg="#1A1D21", fg="#E8EAED", insertbackground="white")
            except Exception:
                pass
            try:
                self.file_listbox.configure(bg="#1A1D21", fg="#E8EAED", selectbackground="#4A90D9", selectforeground="white")
            except Exception:
                pass
            try:
                self.preview_label.configure(style="Dark.TLabel")
            except Exception:
                pass
        except Exception as _e:
            print(f"[Dark] fixup failed: {_e}")
    def toggle_suspend(self):
        if not self.process: return
        self.pause_flag = not self.pause_flag
        self.suspend_btn.configure(text="Resume" if self.pause_flag else "Suspend")
        if self.pause_flag:
            self._set_status_final("Suspended")
        else:
            self._start_running_anim("Running")
        self.log("Suspended" if self.pause_flag else "Resumed")
    def cancel_conversion(self):
        if not self.process: return
        self.stop_flag = True; self.pause_flag = False; self.log("Cancelling...")
        try:
            self.process.terminate(); time.sleep(0.5)
            if self.process.poll() is None: self.process.kill()
        except Exception as e: self.log(f"[Cancel Failed] {e}")
        self._set_status_final("Cancelling...")
    def on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Quit", "Conversion running, quit and cancel?"): return
            self.cancel_conversion(); time.sleep(0.5)
        self.save_config(); self.root.destroy()

def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    app = GenStereoGUI(root); root.mainloop()
if __name__ == "__main__":
    main()
