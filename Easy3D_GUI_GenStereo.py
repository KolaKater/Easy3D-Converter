"""
Easy3D GUI - GenStereo edition
Tkinter GUI for GenStereo (ICCV 2025), RTX 3070 8GB friendly.

Kept from the old GUI: Input/Output, Preset Save/Load/Delete,
Log, Preview, File List, Start / Suspend / Cancel, progress bar.
Removed: Clarity, Clarity Blend, Enhance, Text Fix, Face Restore,
Mode, Resize Mode, Batch, Anti-Alias, Gamma A, Gamma B, Gamma panel.

New GenStereo options:
  Finetuned Model, Pretrained Models (VAE / Image Encoder),
  MDE Model, Depth Strength, Baseline / Disparity Scale,
  Resolution, Steps, Seed, Output Type, Low VRAM, Save Depth/Warped.

Backend contract:
  The GUI calls:  <venv python> test.py <input>  (official GenStereo CLI)
  and writes all extra options to:
      <output_dir>/easy3d_gui_params.json
  plus exports them as GENSTEREO_* env vars for a wrapper.
  If your local test.py accepts extra flags, edit build_command()
  in this file to append them - the GUI already collects them.
"""
import json, os, random, subprocess, sys, threading, time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

__VERSION__ = "v4.4 - 2026-11-07"
APP_TITLE = "Easy3D - GenStereo, FlashDepth, DA2, Gfpgan Converter"
CONFIG_FILE = "preset/easy3d_gui_config.json"
PRESET_FILE = "preset/easy3d_presets.json"
LOG_FILE = "easy3d_debug.log"
SCRIPT_NAME = "test.py"
SCRIPT_ARBITRARY = "test_arbitrary_size.py"  # v3.1 UNUSED - patch path removed, stretch-only. Kept for reference, do NOT use.
# v3.0 SD3D-style: NO Mode option. Resolution = inference SQUARE size (stretch target).
# GUI stretches original -> Res x Res square, inference runs on square, GUI stretches right view back to ORIGINAL W x H.
# No crop, no letterbox, output aspect == input aspect, always.
# v3.1 STRETCH ONLY (SD3D-style, user confirmed stretch is best): NO patch, NO crop, NO letterbox.
# Always test.py single-pass. Resolution = stretch square side. GUI stretches original -> Res x Res, test.py runs UNet at Res, GUI stretches right view back to original W x H.
# Final SBS size = ORIGINAL image size (W x H), NOT Res. Res controls inference detail. Res 1024+ on 3070 8GB is slow / may OOM - lower Res if OOM.
# v1.9 fixed dependencies (not options): VAE + Image Encoder live in models/
FIXED_VAE = "sd-vae-ft-mse"
FIXED_VAE_PATH = "models/sd-vae-ft-mse"
FIXED_ENCODER = "image_encoder"
FIXED_ENCODER_PATH = "models/image_encoder"
SUPPORTED_INPUT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".avif", ".heic", ".heif"}
OUTPUT_JPG_QUALITY = 95
# test.py writes vis/<stem>/{left,generated_right,warped,disp}.png - GUI never scans Output for old files
try:
    from output_type import OUTPUT_TYPE_OPTIONS as _OT_OPTS, process_sbs_file as _process_sbs_file
except Exception as _e:
    print(f"[output_type] not found, using fallback: {_e}")
    _OT_OPTS = None
    _process_sbs_file = None

FINETUNED_OPTIONS = {
    "genstereo-v1.5": "genstereo-v1.5",
    "genstereo-v2.1": "genstereo-v2.1",
}
MDE_OPTIONS = {
    "Depth Anything V2 ViT-L": "depth_anything_v2_vitl",
    "Depth Anything V2 ViT-B": "depth_anything_v2_vitb",
    "Depth Anything V2 ViT-S": "depth_anything_v2_vits",
    "FlashDepth-L": "flashdepth-l",
    "FlashDepth-S, Fast": "flashdepth-s",
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
    "512 - Inference, Fast": 512,
    "640 - Inference": 640,
    "768 - Inference": 768,
    "896 - Inference": 896,
    "1024 - Inference": 1024,
    "1152 - Inference": 1152,
    "1280 - Inference": 1280,
    "1408 - Inference": 1408,
    "1536 - Inference": 1536,
    "1664 - Inference": 1664,
    "1792 - Inference": 1792,
    "1920 - Inference": 1920,
    "2048 - Inference, Slowest": 2048,
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
STEPS_OPTIONS = {
    "20 - Fast preview": 20,
    "30 - Balanced": 30,
    "50 - Default quality": 50,
    "75 - High quality, slow": 75,
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
    "457315609 - Reproducible": "457315609",
    "123 - Reproducible": "123",
    "999 - Reproducible": "999",
    "2025 - Reproducible": "2025",
}

# Old MDE labels (before res patch) -> new labels. Old presets/configs keep working.
_MDE_LABEL_ALIASES = {
    "FlashDepth-L - Official, 518px": "FlashDepth-L",
    "FlashDepth-S - Official, 392px, Fast": "FlashDepth-S, Fast",
}
def _migrate(saved, options):
    if saved in options:
        return saved
    if saved in _MDE_LABEL_ALIASES and _MDE_LABEL_ALIASES[saved] in options:
        return _MDE_LABEL_ALIASES[saved]
    return None

class GenStereoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE + " - " + __VERSION__)
        self.root.geometry("1280x820")
        try:
            _icon = Path(__file__).parent / "Easy3D.ico"
            if _icon.exists():
                self.root.iconbitmap(str(_icon))
        except Exception as _e:
            print(f"[Icon] { _e}")
        self.root.minsize(1120, 750)
        self.process = None
        self.pause_flag = False
        self.stop_flag = False
        self.config = self.load_config()
        style = ttk.Style()
        try: style.theme_use("vista")
        except Exception:
            try: style.theme_use("winnative")
            except Exception: style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 8), foreground="#555555")

        main = ttk.Frame(root, padding=(10, 6)); main.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main, text=APP_TITLE, style="Header.TLabel").pack()

        # ---- Input / Output (v4.4 mockup: Browse left, Entry middle, checkboxes right) ----
        io = ttk.LabelFrame(main, text="Input / Output", padding=(10, 6)); io.pack(fill=tk.X, pady=(0, 6))
        # Row 0: Input: [Browse File][Browse Folder][Entry ........] [Save Depth][Save Warped][Low VRAM]
        io_r0 = ttk.Frame(io); io_r0.pack(fill=tk.X, pady=2)
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
        io_r1 = ttk.Frame(io); io_r1.pack(fill=tk.X, pady=2)
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
        # v4.4: GenStereo Model frame removed - Finetuned / MDE moved into Parameters (mockup).
        self.finetuned_path_var = tk.StringVar(value="models")
        self.mde_path_var = tk.StringVar(value="models/depth_anything_v2_vitl.pth")
        self.vae_var = tk.StringVar(value=list(VAE_OPTIONS)[0])
        self.vae_path_var = tk.StringVar(value=FIXED_VAE_PATH)
        self.encoder_var = tk.StringVar(value=list(ENCODER_OPTIONS)[0])
        self.encoder_path_var = tk.StringVar(value=FIXED_ENCODER_PATH)
        # ---- Stereo Parameters (v4.4 mockup: 12-col, 3 rows, no separate Model frame) ----
        pf2 = ttk.LabelFrame(main, text="Stereo Parameters", padding=(10, 8)); pf2.pack(fill=tk.X, pady=(0, 6))
        for _cc in range(12): pf2.columnconfigure(_cc, weight=1)
        # Row 0: 4 sliders - Depth / Fusion / Baseline / Guidance (each: Label + Scale + Value = 3 cols)
        self.depth_var = tk.DoubleVar(value=float(self.config.get("depth_strength", 1.0)))
        ttk.Label(pf2, text="Depth Strength:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        ttk.Scale(pf2, from_=0.2, to=2.0, orient=tk.HORIZONTAL, variable=self.depth_var, command=self._on_depth).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)
        self.depth_label = ttk.Label(pf2, text=f"{self.depth_var.get():.2f}"); self.depth_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        self.fusion_var = tk.DoubleVar(value=float(self.config.get("fusion_strength", 0.3)))
        ttk.Label(pf2, text="Fusion Strength:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=4)
        ttk.Scale(pf2, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.fusion_var, command=self._on_fusion).grid(row=0, column=4, sticky=tk.EW, padx=5, pady=4)
        self.fusion_label = ttk.Label(pf2, text=f"{self.fusion_var.get():.2f}"); self.fusion_label.grid(row=0, column=5, sticky=tk.W, padx=5)
        self.baseline_var = tk.DoubleVar(value=float(self.config.get("baseline", 0.4)))
        ttk.Label(pf2, text="Baseline / Disparity Scale:").grid(row=0, column=6, sticky=tk.W, padx=5, pady=4)
        ttk.Scale(pf2, from_=0.2, to=2.0, orient=tk.HORIZONTAL, variable=self.baseline_var, command=self._on_baseline).grid(row=0, column=7, sticky=tk.EW, padx=5, pady=4)
        self.baseline_label = ttk.Label(pf2, text=f"{self.baseline_var.get():.2f}"); self.baseline_label.grid(row=0, column=8, sticky=tk.W, padx=5)
        self.guidance_var = tk.DoubleVar(value=float(self.config.get("guidance", 1.8)))
        ttk.Label(pf2, text="Guidance Scale:").grid(row=0, column=9, sticky=tk.W, padx=5, pady=4)
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
        self.seed_var = tk.StringVar(value=_migrate(self.config.get("seed_display", ""), SEED_OPTIONS) or list(SEED_OPTIONS)[0])
        ttk.Label(pf2, text="Seed:").grid(row=2, column=3, sticky=tk.W, padx=5, pady=4)
        self.seed_combo = ttk.Combobox(pf2, textvariable=self.seed_var, values=list(SEED_OPTIONS), state="normal")
        self.seed_combo.grid(row=2, column=4, columnspan=2, sticky=tk.EW, padx=5, pady=4)
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
        self.cancel_btn = ttk.Button(bf, text="Cancel", width=10, command=self.cancel_conversion, state=tk.DISABLED); self.cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self.suspend_btn = ttk.Button(bf, text="Suspend", width=10, command=self.toggle_suspend, state=tk.DISABLED); self.suspend_btn.pack(side=tk.RIGHT, padx=5)
        self.start_btn = ttk.Button(bf, text="Start", width=10, command=self.start_conversion); self.start_btn.pack(side=tk.RIGHT, padx=5)
        pg = ttk.Frame(main); pg.pack(fill=tk.X, pady=2); pg.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Ready"); ttk.Label(pg, textvariable=self.status_var, style="Small.TLabel").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.progress_var = tk.DoubleVar(value=0); ttk.Progressbar(pg, variable=self.progress_var, maximum=100).grid(row=0, column=1, sticky=tk.EW, padx=(10, 0))

        tf = ttk.Frame(main); tf.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(tf, text="View:", style="Small.TLabel").pack(side=tk.LEFT, padx=(5, 10))
        self.show_log_var = tk.BooleanVar(value=self.config.get("show_log", True))
        self.show_preview_var = tk.BooleanVar(value=self.config.get("show_preview", True))
        self.show_filelist_var = tk.BooleanVar(value=self.config.get("show_filelist", True))
        ttk.Checkbutton(tf, text="Log", variable=self.show_log_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(tf, text="Preview", variable=self.show_preview_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(tf, text="File List", variable=self.show_filelist_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)

        bottom = ttk.Frame(main); bottom.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        for c in (0, 1, 2): bottom.columnconfigure(c, weight=1)
        bottom.rowconfigure(0, weight=1)
        # v4.4: Log / File List fill the same height as Preview (no fixed height, rowconfigure weight=1)
        self.log_frame = ttk.LabelFrame(bottom, text="Log", padding=5); self.log_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5)); self.log_frame.rowconfigure(0, weight=1); self.log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        sb = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview); self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW); sb.grid(row=0, column=1, sticky=tk.NS)
        self.preview_frame = ttk.LabelFrame(bottom, text="Preview (Input)", padding=5); self.preview_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5); self.preview_frame.rowconfigure(0, weight=1); self.preview_frame.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(self.preview_frame, text="No Image", anchor=tk.CENTER); self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label.bind("<Configure>", lambda e: self._refresh_preview_scaled())
        self.preview_pil = None; self.preview_tk = None
        self.file_frame = ttk.LabelFrame(bottom, text="File List", padding=5); self.file_frame.grid(row=0, column=2, sticky=tk.NSEW, padx=(5, 0)); self.file_frame.rowconfigure(0, weight=1); self.file_frame.columnconfigure(0, weight=1)
        self.file_listbox = tk.Listbox(self.file_frame, font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4", selectbackground="#007acc")
        fs = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview); self.file_listbox.configure(yscrollcommand=fs.set)
        self.file_listbox.grid(row=0, column=0, sticky=tk.NSEW); fs.grid(row=0, column=1, sticky=tk.NS)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_list_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.refresh_file_list_and_preview)
        self.root.after(400, self.toggle_panels)
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
                "depth_strength": self.depth_var.get(), "baseline": self.baseline_var.get(),
                "fusion_strength": self.fusion_var.get(), "guidance": self.guidance_var.get(),
                "resolution_display": self.resolution_var.get(), "steps_display": self.steps_var.get(),
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
            "mde_model": MDE_OPTIONS.get(self.mde_var.get(), "depth_anything_v2_vitl"),
            "mde_path": (MDE_PATHS.get(MDE_OPTIONS.get(self.mde_var.get(), ""), "") or self.mde_path_var.get().strip()),
            "vae": VAE_OPTIONS.get(self.vae_var.get(), "sd-vae-ft-mse"),
            "vae_path": self.vae_path_var.get().strip(),
            "image_encoder": ENCODER_OPTIONS.get(self.encoder_var.get(), "image_encoder"),
            "image_encoder_path": self.encoder_path_var.get().strip(),
            "depth_strength": round(float(self.depth_var.get()), 3),
            "baseline": round(float(self.baseline_var.get()), 3),
            "fusion_strength": round(float(self.fusion_var.get()), 3),
            "guidance_scale": round(float(self.guidance_var.get()), 3),
            "resolution": RESOLUTION_OPTIONS.get(self.resolution_var.get(), 512),
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
        if "depth_strength" in d: self.depth_var.set(float(d["depth_strength"])); self._on_depth(d["depth_strength"])
        if "baseline" in d: self.baseline_var.set(float(d["baseline"])); self._on_baseline(d["baseline"])
        if "fusion_strength" in d: self.fusion_var.set(float(d["fusion_strength"])); self._on_fusion(d["fusion_strength"])
        if "guidance_scale" in d: self.guidance_var.set(float(d["guidance_scale"])); self._on_guidance(d["guidance_scale"])
        if "resolution" in d and rev(RESOLUTION_OPTIONS, d["resolution"]): self.resolution_var.set(rev(RESOLUTION_OPTIONS, d["resolution"]))
        if "steps" in d and rev(STEPS_OPTIONS, d["steps"]): self.steps_var.set(rev(STEPS_OPTIONS, d["steps"]))
        if "seed" in d:
            _sv = str(d["seed"])
            if _sv and _sv not in SEED_OPTIONS.values():
                _lbl = f"{_sv} - Custom"
                if _lbl not in SEED_OPTIONS: SEED_OPTIONS[_lbl] = _sv
                try: self.seed_combo["values"] = list(SEED_OPTIONS)
                except Exception: pass
                self.seed_var.set(_lbl)
            else: self.seed_var.set(_sv)
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
            if not inp: return
            p = Path(inp)
            files = [p] if p.is_file() else [x for x in sorted(p.iterdir()) if x.suffix.lower() in SUPPORTED_INPUT_EXTS] if p.is_dir() else []
            for f in files: self.file_listbox.insert(tk.END, f.name)
            if files: self.show_preview(files[0])
        except Exception as e: print(f"[FileList] {e}")
    def show_preview(self, path):
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB"); self.preview_pil = img
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
        # v3.1: Finetuned choice is always respected (stretch-only, no patch forcing v2.1).
        model = "V15" if "v1.5" in str(params.get("finetuned_model", "")) else ("V21" if "v2.1" in str(params.get("finetuned_model", "")) else "V15")
        _mde = str(params.get("mde_model", "")).strip().lower()
        _mde_suf = MDE_SUFFIXES.get(_mde, "_MDEC" if _mde else "")
        ot_map = {"parallel": "PAR", "xd": "XD", "align_parallel": "APAR", "align_xd": "AXD", "align_both": "ABOTH"}
        ot = ot_map.get(str(params.get("output_type", "parallel")), "PAR")
        res = int(params.get("resolution", 512) or 512)
        _fr = "_FRBoth" if str(params.get("face_restore", "none")) == "gfpgan_both" else ""
        return f"{stem}_Easy3D_{model}{_mde_suf}_R{res}_{self._param_code(params.get('depth_strength', 1.0), 'D')}_{self._param_code(params.get('baseline', 1.0), 'B')}_{self._param_code(params.get('fusion_strength', 0.5), 'F')}_S{int(params.get('steps', 50))}_{self._param_code(params.get('guidance_scale', 1.5), 'G')}{_fr}_{ot}_Seed{params.get('seed', '')}{extra}"
    def _stretch_to_square(self, input_path, res):
        """SD3D-style stretch: original (W x H, any aspect) -> Res x Res square, NO crop, NO black bars.
        Returns (temp_square_path, (orig_W, orig_H)). Temp file is _tmp_square_<stem>.jpg in ./_tmp_square/."""
        from PIL import Image
        orig = Image.open(input_path).convert("RGB")
        ow, oh = orig.size
        if (ow, oh) == (res, res):
            return Path(input_path), (ow, oh)
        tmp_dir = Path(__file__).parent / "_tmp_square"
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / f"{Path(input_path).stem}_square{res}.jpg"
        orig.resize((int(res), int(res)), Image.LANCZOS).save(tmp, "JPEG", quality=95, subsampling=0, optimize=True)
        return tmp, (ow, oh)
    def _cleanup_vis_and_tmp(self, target, params=None):
        """v3.4: after SBS for ONE image is done, delete its vis/ folder + its _tmp_square file. No leftovers."""
        import shutil
        _sq = getattr(self, "_square_map", {}).get(str(Path(target)), None)
        _square_stem = _sq[0] if _sq else Path(target).stem
        _vis = Path(__file__).parent / "vis" / _square_stem
        _tmp = Path(__file__).parent / "_tmp_square" / f"{Path(target).stem}_square{int((params or {}).get('resolution', 512) or 512)}.jpg"
        # Also try the exact tmp path from _square_map if it was stored as a full path
        for _p, _kind in ((_vis, "vis"), (_tmp, "tmp_square")):
            try:
                if _p.is_dir(): shutil.rmtree(_p, ignore_errors=True)
                elif _p.is_file(): _p.unlink()
            except Exception as _e:
                self.log(f"[Cleanup] Could not delete {_kind} {_p}: {_e}")
        # If _tmp_square is now empty, remove the folder itself so it does not linger
        try:
            _td = Path(__file__).parent / "_tmp_square"
            if _td.is_dir() and not any(_td.iterdir()): _td.rmdir()
        except Exception: pass
    def build_outputs_from_vis(self, targets, output_dir, params):
        """test.py -> vis/<stem>/{left,generated_right,warped,disp}.png
        1) create_sbs_image(left, right) -> temp SBS (no Easy3D name yet)
        2) process_sbs_file(temp, output_type) - output_type.py is authoritative, it deletes temp
        3) rename result(s) to _Easy3D_... .jpg Q95 in output_dir
        Depth/Warped: vis disp/warped -> _DEPTH/_WARPED .jpg Q95"""
        from PIL import Image
        try:
            from output_type import create_sbs_image as _create_sbs
        except Exception:
            _create_sbs = None
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        ot = str(params.get("output_type", "parallel"))
        done = []
        for t in targets:
            # v3.0: t may be the ORIGINAL path. vis dir uses the SQUARE temp stem, not the original stem.
            _sq = getattr(self, "_square_map", {}).get(str(Path(t)), None)
            _square_stem = _sq[0] if _sq else Path(t).stem
            _orig_size = _sq[1] if _sq else None
            stem = Path(t).stem
            vis = Path(__file__).parent / "vis" / _square_stem
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
                _orig = Image.open(t).convert("RGB")
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
                # Temp SBS: plain Left|Right, let output_type.py decide swap/align
                if _create_sbs is not None:
                    sbs_img = _create_sbs(left, right, output_type="parallel")
                else:
                    sbs_img = Image.new("RGB", (left.width + right.width, left.height)); sbs_img.paste(left, (0, 0)); sbs_img.paste(right, (left.width, 0))
                # v4.4 Face Restore: AFTER SBS merge, BEFORE Output Type (output_type.py).
                # One call on the whole SBS image - GFPGAN sees both eyes' faces at once (Both Eyes).
                if str(params.get("face_restore", "none")) == "gfpgan_both":
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
                tmp = out / f"_tmp_sbs_{stem}.jpg"
                sbs_img.save(tmp, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True)
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
                        d = Image.open(disp).convert("RGB"); dst = out / (self.output_stem(str(t), params, extra="_DEPTH") + ".jpg")
                        d.save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True); self.log(f"[Output] {dst.name}")
                if params.get("save_warped"):
                    wp = vis / "warped.png"
                    if wp.exists():
                        w = Image.open(wp).convert("RGB"); dst = out / (self.output_stem(str(t), params, extra="_WARPED") + ".jpg")
                        w.save(dst, "JPEG", quality=OUTPUT_JPG_QUALITY, subsampling=0, optimize=True); self.log(f"[Output] {dst.name}")
            except Exception as e:
                self.log(f"[SBS] Failed {stem}: {e}")
                import traceback; self.log(traceback.format_exc())
        return done
    def build_command(self, params, input_path, output_dir):
        """EDIT HERE if your test.py accepts extra flags.
        Default: official CLI (input path only) + JSON + env vars."""
        exe = sys.executable
        for cand in (Path("venv_Easy3D/Scripts/python.exe"), Path("venv/Scripts/python.exe")):
            if cand.exists(): exe = str(cand); break
        # v3.1 STRETCH ONLY: input_path is ALREADY the stretched square (Res x Res) from _stretch_to_square().
        # Always test.py single-pass, NO patch. test.py runs UNet at Res (cfg.height/width = Res).
        scale = 0.15 * float(params.get("baseline", 1.0)) * float(params.get("depth_strength", 1.0))
        # v3.2: input_path can be a single square OR a list of squares (batch, load model once).
        _inputs = [str(x) for x in input_path] if isinstance(input_path, (list, tuple)) else [str(input_path)]
        cmd = [exe, "-u", SCRIPT_NAME, *_inputs,
               "--output", "./vis",
               "--scale_factor", str(round(scale, 5)),
               "--steps", str(int(params.get("steps", 50))),
               "--guidance", str(float(params.get("guidance_scale", 1.5))),
               "--seed", str(params.get("seed", "-1")),
               "--gen_mode", str(params.get("gen_mode", "orig_right")),
               "--image_size", str(int(params.get("resolution", 512)))]
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
            messagebox.showwarning("Input", "No convertible images found (Easy3D.png excluded)."); return
        if len(_targets0) > 1: self.log(f"[Batch] {len(_targets0)} images (Easy3D.png excluded)")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "easy3d_gui_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        self.stop_flag = False; self.pause_flag = False
        self.start_btn.configure(state=tk.DISABLED); self.suspend_btn.configure(state=tk.NORMAL, text="Suspend"); self.cancel_btn.configure(state=tk.NORMAL)
        self.status_var.set("Running..."); self.progress_var.set(0)
        def run_thread():
            try:
                env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
                for k, v in params.items(): env[f"GENSTEREO_{str(k).upper()}"] = str(v)
                _fm = str(params.get("finetuned_model", "genstereo-v1.5"))
                env["GENSTEREO_VERSION"] = _fm.replace("genstereo-", "").replace("genstereo", "v1.5") if "genstereo" in _fm else _fm
                _res_rt = int(params.get("resolution", 512) or 512)
                self.log(f"[Stretch] Resolution {_res_rt} -> test.py single-pass, UNet at {_res_rt}x{_res_rt}, Finetuned={_fm}, GenMode={params.get('gen_mode','orig_right')} (no patch, no crop)")
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                _rc = 0
                # v3.4 STREAMING (StereoSpace-style): ONE test.py call (load once, v3.2 speedup kept),
                # but SBS is built PER-IMAGE as soon as its vis/ files appear, then vis/ + _tmp_square for
                # THAT image are deleted immediately. No waiting for all 6, no leftover temp files.
                self._square_map = {}
                _sq_paths = []
                for _t in _targets0:
                    _sq_path, _orig_sz = self._stretch_to_square(_t, int(params.get("resolution", 512) or 512))
                    self._square_map[str(Path(_t))] = (_sq_path.stem, _orig_sz)
                    _sq_paths.append(str(_sq_path))
                    self.log(f"[Stretch] {Path(_t).name} {_orig_sz} -> {params.get('resolution')}x{params.get('resolution')} square (no crop), back to {_orig_sz} after inference")
                self.log(f"[Batch] {len(_sq_paths)} images, ONE test.py call, model loaded ONCE (saves ~25s x {len(_sq_paths)-1} reloads)")
                _cmd, _ = self.build_command(params, _sq_paths, output_dir)
                self.log(f"Command: {' '.join(_cmd)}")
                self.process = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore", bufsize=1, creationflags=creationflags, cwd=str(Path(__file__).parent), env=env)
                # --- Watcher: poll vis/ every 1.5s, output + cleanup per image as soon as it is ready ---
                _pending = list(_targets0); _done_set = set(); _watcher_stop = {"v": False}
                _vis_stat = {}
                def _vis_ready(_t):
                    # Stable = all expected PNGs exist, size>0, AND size+mtime unchanged since last poll.
                    # test.py writes 4 files one by one; a file that just appeared may still be half-written.
                    _sq = self._square_map.get(str(Path(_t)), None)
                    if not _sq: return False
                    _vis = Path(__file__).parent / "vis" / _sq[0]
                    _names = ["left.png", "generated_right.png", "disp.png", "warped.png"]
                    if str(params.get("gen_mode", "orig_right")) in ("both", "orig_left"):
                        _names.append("generated_left.png")
                    _stat = []
                    for _n in _names:
                        _p = _vis / _n
                        if not _p.exists(): return False
                        try: _st = _p.stat()
                        except Exception: return False
                        if _st.st_size == 0: return False
                        _stat.append((_n, _st.st_size, _st.st_mtime))
                    _key = str(_t)
                    _prev = _vis_stat.get(_key)
                    _vis_stat[_key] = tuple(_stat)
                    if _prev != tuple(_stat):
                        return False  # first sighting, or still growing -> wait for next poll
                    return True
                def _watcher():
                    import time as _time
                    while not _watcher_stop["v"]:
                        for _t in list(_pending):
                            if self.stop_flag: return
                            try:
                                if _vis_ready(_t):
                                    self.log(f"[Stream] {_t.name} vis ready -> building SBS now (not waiting for batch)")
                                    _built = []
                                    try: _built = self.build_outputs_from_vis([_t], output_dir, params) or []
                                    except Exception as _fe: self.log(f"[SBS] {_fe}")
                                    if not _built:
                                        self.log(f"[Stream] {_t.name} SBS not ready yet (file still writing?), will retry next poll, vis kept")
                                        continue
                                    self._cleanup_vis_and_tmp(_t, params)
                                    _pending.remove(_t); _done_set.add(str(_t))
                                    _n_done = len(_done_set)
                                    self.root.after(0, lambda _n=_n_done: (self.progress_var.set(_n * 100.0 / max(1, len(_targets0))), self.refresh_file_list_and_preview()))
                                    self.log(f"[Stream] {_t.name} done ({_n_done}/{len(_targets0)}), vis + _tmp_square deleted")
                            except Exception as _we: self.log(f"[Stream] {_we}")
                        _time.sleep(1.5)
                _watcher_thread = threading.Thread(target=_watcher, daemon=True); _watcher_thread.start()
                for line in self.process.stdout:
                    if self.stop_flag: break
                    while self.pause_flag and not self.stop_flag: time.sleep(0.2)
                    self.log(line.rstrip())
                self.process.wait(); _rc = self.process.returncode or 0
                # Final sweep: process any vis that became ready after the last poll, then stop watcher
                for _t in list(_pending):
                    try:
                        if _vis_ready(_t):
                            self.log(f"[Stream] Final sweep: {_t.name}")
                            _built = []
                            try: _built = self.build_outputs_from_vis([_t], output_dir, params) or []
                            except Exception as _fe: self.log(f"[SBS] {_fe}")
                            if not _built:
                                self.log(f"[Stream] Final sweep: {_t.name} build failed, vis kept for manual recovery")
                                continue
                    except Exception: pass
                    self._cleanup_vis_and_tmp(_t, params)
                _watcher_stop["v"] = True
                class _P: returncode = _rc
                self.process = _P()
                if self.stop_flag: self.root.after(0, lambda: (self.status_var.set("Cancelled"), self.progress_var.set(0)))
                elif self.process.returncode == 0:
                    self.root.after(0, lambda: (self.status_var.set(f"Done - {output_dir}"), self.progress_var.set(100)))
                    self.root.after(0, self.refresh_file_list_and_preview)
                else: self.root.after(0, lambda: self.status_var.set(f"Finished code {self.process.returncode}"))
            except Exception as e:
                self.log(f"[Exception] {e}")
            finally:
                self.process = None
                self.root.after(0, lambda: self.start_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.suspend_btn.configure(state=tk.DISABLED, text="Suspend"))
                self.root.after(0, lambda: self.cancel_btn.configure(state=tk.DISABLED))
        threading.Thread(target=run_thread, daemon=True).start()
    def toggle_suspend(self):
        if not self.process: return
        self.pause_flag = not self.pause_flag
        self.suspend_btn.configure(text="Resume" if self.pause_flag else "Suspend")
        self.status_var.set("Suspended" if self.pause_flag else "Running..."); self.log("Suspended" if self.pause_flag else "Resumed")
    def cancel_conversion(self):
        if not self.process: return
        self.stop_flag = True; self.pause_flag = False; self.log("Cancelling...")
        try:
            self.process.terminate(); time.sleep(0.5)
            if self.process.poll() is None: self.process.kill()
        except Exception as e: self.log(f"[Cancel Failed] {e}")
        self.status_var.set("Cancelling...")
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
