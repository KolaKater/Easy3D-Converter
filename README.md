<p align="center">

### 💙 Join Our Facebook Group · 加入 Facebook 群组

# 👉 [3D Stereo Lover](https://www.facebook.com/share/g/1KokTEPmCs/)

**Share your SBS results · Share presets · Ask questions · 分享 SBS 成果 · 问参数 · 抄 Preset**

</p>

---

# Easy3D Converter: I love Depth. I also love Light.

**Turn one image into a side-by-side (SBS) stereo 3D image. / 把一张图变成 SBS 立体 3D 图。**

**Easy3D is not only a 3D converter, but also a 3D creator. / 它不只是一个 3D 转换器，也是一个 3D 创作者。**

Easy3D Converter is a free, open-source tool. It runs on a consumer GPU such as an NVIDIA RTX 3070 or higher.
Easy3D Converter 是一个免费开源工具，在 RTX 3070 及以上的普通显卡上就能跑。

> **Current version: v7.0 / 当前版本：v7.0**

[[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[[Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[[GPU: RTX 3070+](https://img.shields.io/badge/GPU-RTX%203070%2B-brightgreen.svg)](https://www.nvidia.com/)

<img width="1280" height="847" alt="Easy3D Converter GUI" src="https://github.com/user-attachments/assets/b195f497-e303-4491-9896-5db86fb2c512" />

*Easy3D Converter GUI*

<img width="3836" height="1077" alt="Original image to SBS stereo 3D result" src="https://github.com/user-attachments/assets/1a982c23-568b-4f75-aa75-d0ddfd092b50" />

*Original image → SBS stereo 3D result / 原图 → SBS 立体 3D 成果*

---

## Table of Contents / 目录

- [Why Easy3D Is Different](#why-easy3d-is-different--为什么-easy3d-不一样)
- [What's New in v7.0](#whats-new-in-v70--v70-更新)
- [Quick Start](#quick-start--快速上手)
- [How to Pick: Mode + Resolution](#how-to-pick-mode--resolution--怎么选模式--分辨率)
- [Parameters](#parameters--参数)
- [Requirements](#requirements--配置要求)
- [Installation](#installation--安装)
- [Update](#update--更新)
- [Troubleshooting](#troubleshooting--故障排查)
- [FAQ](#faq--常见问题)
- [Community](#community--facebook-群组)
- [Credits & License](#credits--license--致谢与许可)

---

## Why Easy3D Is Different · 为什么 Easy3D 不一样

> **Easy3D doesn't just capture Depth — it also captures Light, Fire, Water, Fog...**
> **Easy3D 抓的不只是 Depth（深度），还有 Light（光）、Fire（火）、Water（水）、Fog（雾）……**

Most 2D-to-3D tools only do one thing: take a depth map and shift pixels left / right. That works for solid things like a chair or a building, but not for things with no solid surface.

一般的 2D 转 3D 工具只做一件事：拿深度图，把像素左右挪一下。实心东西（椅子、楼）这样没问题，但没有实心表面的东西就破功了：

| It breaks on / 一碰到这些就破功 | Why / 为什么 |
|---|---|
| Light, glow, neon / 光、辉光、霓虹 | Light has no depth. / 光没有深度。 |
| Fire, flames / 火、火焰 | Fire is transparent and flickering. / 火是透明又会闪的。 |
| Water, reflections / 水、反光 | Water is transparent and reflective. / 水透明又会反光。 |
| Fog, smoke, mist / 雾、烟、霾 | Fog is air, not a surface. / 雾是空气，不是面。 |
| Glass / 玻璃 | One pixel has two depths: the glass and what is behind it. / 一个像素里有两个深度：玻璃和玻璃后面的东西。 |

The result is the classic failure: holes, black edges, tearing, ghosting, or the fire / fog looks flat while everything else pops.

结果就是经典翻车：黑边、撕裂、重影，或者火 / 雾看着是平的，只有别的东西立体。

**Easy3D is different.** It does not only shift pixels. It re-generates the other eye's view, so the glow still glows, the fire still burns, the water still reflects, and the fog still floats — all in 3D.

**Easy3D 不一样。** 它不只挪像素，还会重新生成另一只眼看到的画面。所以同一张 SBS 里，主体有深度，光还在发光、火还在烧、水还在反光、雾还在飘，都是立体的，不是平贴上去的。

---

## What's New in v7.0 · v7.0 更新

- **Simpler GUI: fewer knobs, clearer intent. / 界面更简单：旋钮更少，意图更清楚。**
- **Up to 4096 px on RTX 3070 or above, without OOM. / RTX 3070 及以上，分辨率上限到 4096，不再 OOM。**
- **Better quality than before — fewer ghosting / double images and hard seams. / 质量比以前好：重影 / 双影、硬缝更少了。**
- **Recommended: 1344 px for the best quality / speed balance. 1792 / 2688 px for the cleanest results. 5120 px is the slowest. / 推荐：1344 最划算，1792 / 2688 最干净，5120 最慢。**

---

## Quick Start · 快速上手

1. Double-click `3_Launch_Easy3D_GUI.bat`.
   双击 `3_Launch_Easy3D_GUI.bat`。
2. **Input:** Click **Browse File / Browse Folder**, or drag & drop an image / folder onto the Input box.
   **输入：** 点 **Browse File / Browse Folder**，或者把图片 / 文件夹拖到输入框。
3. **Output:** Leave **Default (InputName_3D)**. Output goes to `<InputName>_3D\` next to the input.
   **输出：** 默认 **Default (InputName_3D)** 就行，输出在输入旁边 `<InputName>_3D\` 文件夹里。
4. Set the parameters below, then click **Start**. Watch **Log / Preview / File List**. **Suspend** pauses, **Cancel** stops.
   按下面设好参数，点 **Start**。看 **Log / Preview / File List**。**Suspend** 是暂停，**Cancel** 是停止。

**Recommended first run (RTX 3070+ / 第一次推荐参数)：**

```
Finetuned:    genstereo-v2.1
MDE:          FlashDepth-L
Resolution:   1344 - Finetune-v2.1 Recommended
Steps:        20 - Fast
3D Strength:  0.40
Follow Original: 1.80
Gen Mode:     Orig_Right
Face Restore: None
Output Type:  1. Parallel (Default)
```

The filename already contains all your settings, for example:
文件名里已经有全部参数，例如：

```
White Rose_Easy3D_V21_FDL_R1024_B040_G180_PAR_Seed402260655.jpg
```

> **Tip: start with landscape photos.** Easy3D works best with landscape (wide) images. Portrait images work, but usually look flatter. See [FAQ](#faq--常见问题).
> **建议：先用横屏照片。** Easy3D 最适合横屏（宽）图片。竖屏也能用，但通常看着比较平。见 [FAQ](#faq--常见问题)。

---

## How to Pick: Mode + Resolution · 怎么选：模式 + 分辨率

You only need to remember two choices: **Mode** decides speed vs. quality, **Resolution** decides how fine the result is. Pick them together, and the result will look better.

你只需要记住两个选择：**Mode（模式）决定快慢和质量，Resolution（分辨率）决定精细度。** 两个一起配，效果才好。

### Mode / 模式

| Mode / 模式 | What it feels like / 用起来像什么 | Use it when / 什么时候用 |
|---|---|---|
| **Quality** | Slowest, cleanest. / 最慢，最干净。 | You want the best result and are not in a hurry. / 想要最好效果，不赶时间。 |
| **Fast - Decoupled A** | Fast. / 快。 | Daily use, quick preview. / 日常出图、快速预览。 |
| **Fast - Decoupled B** | Faster than Quality, finer than A. / 比 Quality 快，比 A 细。 | You want a balance of speed and quality. / 想要速度和质量都兼顾。 |
| **Fast - Decoupled C** | Finest of the three Fast modes, still faster than Quality. / 三个 Fast 里最细的，仍比 Quality 快。 | You want the finest Fast result. / 想要 Fast 里最细的效果。 |

### The pairing that works best / 这样配，效果最好

**Just remember this table. You do not need to know why it works inside — this pairing simply gives the best result.**
**记住下面这张表就行，不用管里面是怎么做的，这样配效果最好。**

| If you pick / 如果你选 | Resolution should be at least / Resolution 至少选 |
|---|---|
| **Fast - Decoupled A** | **768 or above / 768 或以上** |
| **Fast - Decoupled B** | **896 or above / 896 或以上** |
| **Fast - Decoupled C** | **1280 or above / 1280 或以上** |

> **Why? / 为什么？**
> This pairing gives the best result. If Resolution is too low for the Mode you picked, the result will look worse.
> 这样配，效果最好。如果 Resolution 配得太低，效果会变差。

**Example / 举例：**

- `Fast - Decoupled A` + `Resolution 768` — good. / 可以。
- `Fast - Decoupled A` + `Resolution 512` — too low, avoid it. / 太低了，别这样配。
- `Fast - Decoupled C` + `Resolution 1344` — good. / 可以。
- `Fast - Decoupled C` + `Resolution 896` — too low, avoid it. / 太低了，别这样配。

> **One more rule: do not upscale a small source.** If your source image is only 2048 px, running at 4096 px will not add detail — it only upscales. Use 2048 px or below for a 2048 px source. Use 2560 / 3072 / 4096 px only when the source itself is 3K / 4K.
> **还有一条：原图小，别硬拉大。** 原图只有 2048 px，跑 4096 px 也不会多出细节，只是放大。原图 2048 px 就用 2048 px 或以下；原图本身是 3K / 4K 再用 2560 / 3072 / 4096 px。

### Resolution labels / Resolution 标签怎么看

Each item in the Resolution dropdown has 3 parts: `number - which Finetuned it fits + speed / recommendation`.
Resolution 下拉框里每一项是三段式：`数字 - 最配哪个 Finetuned + 快慢 / 推荐`。

| Word in label / 标签里的词 | Meaning / 意思 |
|---|---|
| `Finetune-v1.5` | Best for `genstereo-v1.5`. / 最配 `genstereo-v1.5`。 |
| `Finetune-v2.1` | Best for `genstereo-v2.1`. / 最配 `genstereo-v2.1`。 |
| `Common` | Works for both. / 两个都能用。 |
| `Fast` | Fastest, least detail, good for preview. / 最快，细节最少，适合预览。 |
| `Recommended` | Best quality / speed balance for daily use. / 日常推荐：质量 / 速度最划算。 |
| `Slowest` | Slowest, most detail, use sparingly. / 最慢，细节最多，慎用。 |

**Example / 举例 — `1344 - Finetune-v2.1 Recommended`：** Resolution is `1344`, best for `genstereo-v2.1`, and recommended for daily use. / 分辨率是 `1344`，最配 `genstereo-v2.1`，而且是日常推荐。

So: if Finetuned is `genstereo-v2.1`, pick `1344` for daily use. If Finetuned is `genstereo-v1.5`, pick `1280 - Finetune-v1.5 Recommended`.
所以：Finetuned 选 `genstereo-v2.1`，日常优先选 `1344`；选 `genstereo-v1.5`，日常优先选 `1280 - Finetune-v1.5 Recommended`。

---

## Parameters · 参数

You do not need to tune everything. Start with the Quick Start settings above, then only change one thing at a time.
不用每个都调。先用上面快速上手的设置，然后一次只改一个。

### 3D Strength / 3D 强度

Overall stereo strength. How strong the 3D pops. Suggested 0.5. Larger = more 3D, but more deformation.
整体立体强度，3D 出屏有多强。建议 0.5。越大越立体，但变形也越多。

| Value / 值 | Effect / 效果 |
|---|---|
| 0.2 | Very weak stereo. / 立体感很弱。 |
| **0.40** | **Normal (recommended). / 正常（推荐）。** |
| 0.5 | A bit stronger, still comfortable. / 稍强一点，还舒服。 |
| 1.0+ | Too strong — dizzy, ghosting. / 太强了：头晕、重影。 |

### Follow Original / 跟原图

How strictly to follow the original image. Think of it as: who is in charge, the creator or the converter?
多严格地跟着原图走。可以这样理解：到底谁说了算，是创作者还是转换器？

- **High Follow Original:** the converter listens to the creator. The result stays close to your original photo. / **跟原图跟得紧：** 转换器听创作者的，结果跟你的原图很像。
- **Low Follow Original:** the converter freestyles more. It may invent details that were not in your photo. / **跟原图跟得松：** 转换器自己发挥得多，可能会编出原图里没有的细节。

| Value / 值 | Effect / 效果 |
|---|---|
| 0.0 | Drifts away from the original. / 跑偏。 |
| **1.80** | **Default (recommended). / 默认（推荐）。** |
| 5.0 | Too rigid, repetitive. / 太死板、重复。 |

Follow Original does **not** affect speed.
Follow Original **不影响**速度。

### Steps / 步数

More steps = cleaner, but slower. Speed is decided by **Steps, Resolution, and Finetuned Model** only. / 步数越多越干净，但越慢。速度只由 **Steps、Resolution、Finetuned** 决定。

| Steps / 步数 | Use / 用途 |
|---|---|
| 20 | Fast preview. / 快速预览。 |
| 30 | Balanced. / 均衡。 |
| 50 | Default quality. / 默认质量。 |

### Seed / 种子

`Random (Default)` = a different result each run. Or pick a preset: Seed A `510902704`, Seed B `457315609`, Seed C `402260655`, Seed D `1999`, Seed E `2026`. You can also type your own integer (`0 – 2147483647`).
`Random (Default)` = 每次都不一样。也可以选预设：Seed A `510902704`，Seed B `457315609`，Seed C `402260655`，Seed D `1999`，Seed E `2026`。也可以自己输一个整数（`0 – 2147483647`）。

### Gen Mode / 生成模式

| Mode / 模式 | Left / 左眼 | Right / 右眼 | Notes / 说明 |
|---|---|---|---|
| `Orig_Right` (default / 默认) | Original / 原图 | AI | Most common. / 最常用。 |
| `Orig_Left` | AI | Original / 原图 | Mirrored. / 镜像。 |
| `Both / Symmetric` | AI | AI | Most comfortable, two passes. / 最舒服，跑两次。 |

### Face Restore / 人脸修复

| Option / 选项 | Effect / 效果 |
|---|---|
| `None` (default / 默认) | No face restore. / 不修脸。 |
| `Both Eyes` | Fix faces in both eyes at once. Filename adds `_FRBoth`. / 两只眼一起修，文件名多 `_FRBoth`。 |

Face Restore runs **after all images are converted**, not one by one. You do not need to wait for it per image.
人脸修复是**等所有图都转完才统一做**，不是一张一张做，不用每张都等它。

Requires `gfpgan/weights/`. If it fails to load, the SBS is saved unchanged and the Log shows the reason.
需要 `gfpgan/weights/`。如果加载失败，SBS 会原样保存，Log 里会写原因。

### Depth Model (MDE) / 深度模型

MDE makes the depth map. It decides the geometry, not the final look. / MDE 是做深度图的，决定几何，不决定最终画面的风格。

| GUI option / 选项 | Quality / 质量 | Speed / 速度 | Use for / 适合 |
|---|---|---|---|
| **FlashDepth-L (default / 默认)** | Good / 好 | Fast / 快 | **Daily default. / 日常默认就是它。** |
| FlashDepth-S, Fast | Slightly weaker / 稍弱 | Faster / 更快 | 8 GB + high Resolution (>3000) to avoid OOM. / 8 GB + 高分辨率（>3000）时用它，防 OOM。 |
| Depth Anything V2 ViT-L | Good / 好 | Slow / 慢 | Try for a different look. Do not use Large on 8 GB + >3000. / 想换个风格时试。8 GB + >3000 别用 Large。 |
| Depth Anything V2 ViT-B | Medium / 中 | Medium / 中 | Alternative on 8 GB + high Resolution (>3000). / 8 GB + 高分辨率（>3000）时的备选。 |
| Depth Anything V2 ViT-S | Weaker / 弱 | Fastest / 最快 | Most stable on 8 GB + high Resolution (>3000). / 8 GB + 高分辨率（>3000）时最稳。 |

> **8 GB rule (important) / 8 GB 规则（重要）：**
> - Resolution `≤ 3000`: any MDE is fine, default `FlashDepth-L` works. / Resolution `≤ 3000`：随便选，默认 `FlashDepth-L` 就行。
> - Resolution `> 3000` (e.g. 3072 / 3584 / 4096 / 5120): **switch MDE to Base / Small first** — `FlashDepth-S`, or `Depth Anything V2 ViT-B` / `ViT-S`. / Resolution `> 3000`（如 3072 / 3584 / 4096 / 5120）：**先把 MDE 换成 Base / Small** — `FlashDepth-S`，或 `Depth Anything V2 ViT-B` / `ViT-S`。

### Low VRAM / 低显存

On by default. Recommended for RTX 3070 8GB. / 默认开。RTX 3070 8GB 推荐开。

### Save Depth / Save Warped / 保存深度图 / 保存 Warp 图

Optional. Saves `_DEPTH` and `_WARPED` JPGs alongside the SBS. / 可选。会在 SBS 旁边多存 `_DEPTH` 和 `_WARPED` 两张 JPG。

### Output Type / 输出类型

| Option / 选项 | Meaning / 意思 |
|---|---|
| **1. Parallel (Default / 默认)** | For parallel viewing / VR. / 平行眼看 / VR 看。 |
| 2. Cross-Eyed | For cross-eyed viewing. / 斗鸡眼看。 |
| 3. Auto-Align Parallel / XD | Auto-aligned. / 自动对齐。 |

### Output Naming / 文件名

Always JPG, `quality=95`. The filename already contains all settings, so you can reproduce any result from the filename alone.
永远是 JPG，`quality=95`。文件名里已经有全部参数，只看文件名就能复现。

---

## Requirements · 配置要求

| Item / 项目 | Minimum / 最低 | Recommended / 推荐 |
|---|---|---|
| OS / 系统 | Windows 10 / 11 (64-bit) | Windows 11 |
| Python | 3.11.9 | 3.11.9 |
| GPU / 显卡 | NVIDIA RTX 2060 6GB | **RTX 3070 8GB** or above / 或以上 |
| VRAM / 显存 | 6 GB (512 px) | 8 GB+ (512–4096 px) |
| Disk / 硬盘 | ~25 GB | ~30 GB |
| CUDA | Driver with CUDA 12.4+ | Latest NVIDIA driver / 最新驱动 |

> Training is **not** supported on a consumer GPU. This project is inference only. / 普通显卡**不支持**训练，这个项目只做推理。

---

## Installation · 安装

All paths below are **relative** to the Easy3D folder. Copy the folder anywhere and run. / 下面所有路径都是相对路径，相对 Easy3D 文件夹。文件夹拷到哪都能跑。

### 1. Clone / Download / 克隆 / 下载

```bash
git clone https://github.com/KolaKater/Easy3D-Converter.git
cd Easy3D-Converter
```

Or download the ZIP and extract it. Do **not** use a path with Chinese characters or parentheses if you use the `.bat` launchers — `C:\3D\Easy3D\` is a safe example.
或者下载 ZIP 解压。如果用 `.bat` 启动器，路径里**不要**有中文或括号，`C:\3D\Easy3D\` 这样就安全。

### 2. Create the Virtual Environment / 创建虚拟环境

Double-click / 双击：

```
1_install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat
```

This step auto-detects your GPU and installs the matching PyTorch CUDA build. / 这一步会自动检测显卡，装对应的 PyTorch CUDA 版本。

### 3. Download All Models / 下载所有模型

Double-click / 双击：

```
2_install_all.bat
```

### 4. Verify / 验证

You should see / 应该看到：

```
torch: 2.6.0+cu124 cuda: True
```

---

## Update · 更新

Double-click / 双击：

```
update.bat
```

Wait for `[Done] Updated to latest version!`, then double-click `3_Launch_Easy3D_GUI.bat` again.
等出现 `[Done] Updated to latest version!`，再双击 `3_Launch_Easy3D_GUI.bat`。

**What is NOT overwritten / 不会被覆盖的：**

- `models/` — your downloaded weights / 你下载的模型
- `venv_Easy3D/` / `venv/` — your Python environment / 你的 Python 环境
- `preset/` — your presets / 你的 Preset
- `vis/` / `_cache/` / `*.log` — your temp / logs / 临时 / 日志

If update fails, check network / GitHub first, then re-run `update.bat` once more.
如果更新失败，先检查网络 / GitHub，再重新双击一次 `update.bat`。

---

## Troubleshooting · 故障排查

| Problem / 问题 | Fix / 解决办法 |
|---|---|
| `torch.cuda.is_available() == False` / CPU torch | You installed CPU torch. Delete `venv_Easy3D\`, re-run `1_install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat`. / 装成 CPU 版 torch 了。删掉 `venv_Easy3D\`，重跑 `1_install_venv_Easy3D_v3_for_RTX20_to_RTX50.bat`。 |
| `pillow` / `numpy` errors | Do not upgrade them. Re-run the install `.bat` — it pins the right versions. / 别升级它们。重跑安装 `.bat`，它会锁定正确版本。 |
| `GFPGAN loaded: False` | Check `gfpgan/weights/` is complete. / 检查 `gfpgan/weights/` 是不是齐的。 |
| OOM on 3070 8GB at Resolution > 3000 | First switch **MDE to Base / Small**, then lower **Resolution** and **Steps**. / 先把 **MDE 换成 Base / Small**，再降 **Resolution** 和 **Steps**。 |
| Batch gets slower after image 1 (30s → 11 min) | You are on an old version. Update with `update.bat`. / 你是旧版本。用 `update.bat` 更新。 |
| `Output/_deferred/` left over after Cancel | Normal on Cancel. Delete it and re-run, or re-run with the same output folder. / Cancel 后留下来是正常的。删掉重跑，或者用同一个输出文件夹重跑。 |
| Portrait image looks flatter than landscape | Expected. See FAQ. / 预期的。见 FAQ。 |
| `Flux.2 Klein` cannot be loaded | Only `genstereo-v1.5` / `genstereo-v2.1` are supported. Flux is a different architecture. / 只支持 `genstereo-v1.5` / `genstereo-v2.1`。Flux 是另一种结构。 |

---

## FAQ · 常见问题

**Q: Can I use it for video? / 能做视频吗？**
No, single-image only for now. Frame-by-frame causes flicker / depth jumps. / 暂时只能做单张图。逐帧做会闪烁 / 深度跳变。

**Q: Does Follow Original affect speed? / Follow Original 影响速度吗？**
No. It is a weight. Speed is decided by **Steps, Resolution, and Finetuned Model** only. / 不影响。它是权重。速度只由 **Steps、Resolution、Finetuned** 决定。

**Q: Why does a portrait image look flatter than a landscape image? / 竖屏为什么比横屏看着平？**
This is expected. Easy3D works best with landscape (wide) images. / 这是预期的。Easy3D 最适合横屏（宽）图片。

Depth comes from left–right difference. A wide image has more room to show near / far. A tall, narrow image has less room, so the same settings look flatter.
深度靠左右眼的差别。宽图左右够宽，近 / 远分得开。竖图左右太窄，同样的参数看着就是比较平。

If you have a portrait photo: crop it to landscape if you can; raise `3D Strength` a little (e.g. `0.40 → 0.55`) and keep it comfortable; use `Resolution 1024` + `Steps 30`.
竖屏想尽量出效果：能裁成横屏就先裁；`3D Strength` 稍微加大一点（比如 `0.40 → 0.55`），别出重影；用 `Resolution 1024` + `Steps 30`。

**Q: Why are some resolutions not selectable? / 为什么有些分辨率选不了？**
Because resolutions above 960 px are tested. Some resolutions give bad results, so they are hidden on purpose. Use one of the listed resolutions. / 因为 960 px 以上的分辨率是测过的。有些分辨率效果不好，所以故意藏起来了。用列表里的分辨率就行。

**Q: Where can I share results / ask for help? / 成果发哪 / 去哪问？**
Join the Facebook group: **[3D Stereo Lover](https://www.facebook.com/share/g/1KokTEPmCs/)** — post your SBS image + filename (it contains all params) + GPU model.
加 Facebook 群组：**[3D Stereo Lover](https://www.facebook.com/share/g/1KokTEPmCs/)** — 发 SBS 图 + 文件名（里面有全部参数）+ 显卡型号。

---

## Community · Facebook 群组

> ### 💙 [3D Stereo Lover — Facebook Group](https://www.facebook.com/share/g/1KokTEPmCs/)
>
> **👉 Click to join · 点击加入：https://www.facebook.com/share/g/1KokTEPmCs/**
>
> Share your results, presets, and settings. / 分享你的成果、Preset、参数设置。
>
> **When you post, please include / 发帖时建议带上：**
> - Your SBS result image (Parallel / XD) / 你的 SBS 成果图（Parallel / XD）
> - The filename (it already contains all parameters) / 文件名（里面已经有全部参数）
> - Your GPU model (e.g. RTX 3070 8GB) / 显卡型号（例如 RTX 3070 8GB）
>
> That way others can reproduce your result at a glance. / 这样别人一眼就能复现你的效果。

---

## Credits & License · 致谢与许可

This project builds on the work of many open-source authors and teams. Thank you to all of them. / 这个项目站在很多开源作者和团队的肩膀上，谢谢他们。

| Component / 组件 | Authors / Team / 作者 | Link / License / 链接 / 许可 |
|---|---|---|
| **GenStereo** | Fan Qiao et al. — ICCV 2025 | [FQiao/GenStereo](https://github.com/FQiao/GenStereo) — see its repo for license |
| **Depth Anything V2** | Depth Anything Team | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) — Apache-2.0 |
| **FlashDepth** | FlashDepth Authors | [DepthAnything/FlashDepth](https://github.com/DepthAnything/FlashDepth) — see its repo for license |
| **Stable Diffusion** | CompVis / Stability AI / Runway | [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) — CreativeML Open RAIL-M |
| **CLIP** | OpenAI | [openai/CLIP](https://github.com/openai/CLIP) — MIT |
| **GFPGAN** | Tencent ARC — Xintao Wang et al. | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) — Apache-2.0 |
| **StereoAutoAlign Ver0.30** | Masuji Suto (Muttyan) | Auto-Align for SBS stereo. See `ffmpeg/bin/stereoautoalign030/` in this repo. |
| **BasicSR / Real-ESRGAN / FaceXLib** | Xintao Wang et al. | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) / [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) / [xinntao/facexlib](https://github.com/xinntao/facexlib) |
| **Hugging Face Diffusers / Transformers** | Hugging Face | [huggingface/diffusers](https://github.com/huggingface/diffusers) / [huggingface/transformers](https://github.com/huggingface/transformers) — Apache-2.0 |
| **PyTorch** | Meta AI | [pytorch/pytorch](https://github.com/pytorch/pytorch) — BSD-3-Clause |
| **xFormers** | Meta AI / xFormers Team | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) — BSD-3-Clause |
| **TkinterDnD2** | Michael Lange / contributors | [pmgagne/tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) — MIT |
| **Easy3D Converter** | Tony Lin (this repo) | MIT — GUI, batch, SBS merging, naming |

**Model weights** are distributed by their original authors under their own licenses. This repo does **not** re-license them. Before redistributing any downloaded weight file, check its original repo / Hugging Face page.
**模型权重**是原作者按他们自己的许可发布的，这个仓库**没有**重新许可它们。转发下载的权重文件之前，先看原仓库 / Hugging Face 页面的许可。

### License / 许可

**Easy3D Converter code in this repository is released under the MIT License:**
**这个仓库里的 Easy3D Converter 代码按 MIT 许可发布：**

```
MIT License

Copyright (c) 2026 Tony Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Third-party code, models, and weights retain their original licenses.** See the table above and each upstream repo for details.
**第三方代码、模型、权重保持原许可。** 见上表和各上游仓库。

### Disclaimer / 免责

This project is for research / personal use. Stereo 3D viewing can cause eye strain / dizziness — take breaks, and do not use while driving or operating machinery.
这个项目只供研究 / 个人使用。看立体 3D 可能引起眼睛累 / 头晕 — 多休息，开车或操作机器时不要看。

---

**Easy3D Converter — Made for RTX 3070+, shared for everyone. Enjoy your 3D!**
