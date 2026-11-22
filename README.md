<p align="center">

### 💙 Join Our Facebook Group · 加入 Facebook 群组

# 👉 [3D Stereo Lover](https://www.facebook.com/share/g/1KokTEPmCs/)

**Share your SBS results · Share presets · Ask questions · 分享 SBS 成果 · 问参数 · 抄 Preset**

</p>

---

# Easy3D: Exploring a 3D World of Light and Transparency.

**Turn one image / video into side-by-side (SBS) stereo 3D. / 把一张图 / 一段视频变成 SBS 立体 3D。**

**Easy3D is a high-quality 2D-to-3D SBS generation tool — not a generic converter. / Easy3D 是一个高质量的 2D-3D SBS 生成工具，不是一般的转换。**

**Easy3D is not only a 3D converter, but also a 3D creator. / 它不只是一个 3D 转换器，也是一个 3D 创作者。**

Easy3D Converter is a free, open-source tool. It runs on a consumer GPU such as an NVIDIA RTX 3070 or higher.
Easy3D Converter 是一个免费开源工具，在 RTX 3070 及以上的普通显卡上就能跑。

> **Current version: v10.9 / 当前版本：v10.9**

## Download Installer · 下载安装包

**Don't build from source — download the installer: / 不用自己编译，直接下安装包：**

**👉 [https://github.com/KolaKater/Easy3D-Converter/releases](https://github.com/KolaKater/Easy3D-Converter/releases)**

Download `Easy3D_Setup.exe` (~1.5 GB), double-click it, follow the installer. It installs the program + Python + PyTorch + CUDA. Model weights (~15-20 GB) are downloaded automatically on first run from Hugging Face. / 下载 `Easy3D_Setup.exe`（约 1.5 GB），双击按提示装。安装包里含程序 + Python + PyTorch + CUDA。模型权重（约 15-20 GB）首次运行时自动从 Hugging Face 下载。

> The installer needs Administrator (it writes to `C:\Program Files`). After install, run Easy3D as a **normal user** — do NOT use "Run as administrator", or drag-and-drop will be blocked by Windows (UIPI). / 安装器需要管理员（写 `C:\Program Files`）。装完后用**普通用户**跑 Easy3D——不要用“以管理员身份运行”，否则拖放会被 Windows 拦掉（UIPI）。

[[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[[Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[[GPU: RTX 3070+](https://img.shields.io/badge/GPU-RTX%203070%2B-brightgreen.svg)](https://www.nvidia.com/)

<img width="1279" height="850" alt="图片" src="https://github.com/user-attachments/assets/c4398773-5d61-49ef-911c-387db6037150" />

*Easy3D Converter GUI*

<img width="3836" height="1918" alt="1794099221872-1918_Easy3D_V21_FDL_R2048_D100_B051_F050_S30_G150_PAR_Seed510902704" src="https://github.com/user-attachments/assets/18ce0af5-352d-4005-9131-56f271bb544e" />

*Original image → SBS stereo 3D result / 原图 → SBS 立体 3D 成果*

---

## Table of Contents / 目录

- [Why Easy3D Is Different](#why-easy3d-is-different--为什么-easy3d-不一样)
- [What's New in v10.9](#whats-new-in-v109--v109-更新)
- [Quick Start](#quick-start--快速上手)
- [How to Pick: Mode + Resolution](#how-to-pick-mode--resolution--怎么选模式--分辨率)
- [Parameters](#parameters--参数)
- [Language](#language--语言)
- [Requirements](#requirements--配置要求)
  - [GPU Compatibility](#gpu-compatibility--显卡兼容)
- [Installation](#installation--安装)
- [Troubleshooting](#troubleshooting--故障排查)
- [FAQ](#faq--常见问题)
- [Community](#community--facebook-群组)
- [Credits & License](#credits--license--致谢与许可)

---

## Why Easy3D Is Different · 为什么 Easy3D 不一样

> **Easy3D is built for one thing: a high-quality SBS you actually want to look at.**
> **Easy3D 只做一件事：出一张你真的愿意看的高质量 SBS。**
> A generic converter just shifts pixels. Easy3D is tuned so the SBS keeps the original's detail, colour, and sharpness, with stable stereo and no warped / ghosted mess — even at high resolution.
> 一般的转换只是把像素挪一下。Easy3D 的目标是 SBS 保持原图的细节、颜色和清晰度，立体稳定，高分辨率也不容易走样 / 重影。

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

## What's New in v10.9 · v10.9 更新

- **Image + Video.** Image and video share one GUI. Video has its own restrictions (FlashDepth only, fixed Seed, no Face Restore, `Limit to Source` forced ON). / **图片 + 视频。** 图片和视频共用一个 GUI。视频有自己的限制（只用 FlashDepth、固定 Seed、不做 Face Restore、`Limit to Source` 强制开）。
- **Language dropdown.** Right of `Low VRAM`. List lives in `configs\Easy3D_Languages.json`, edit that file to add / remove / rename a language, no need to edit the GUI code. 24 languages, no Russian. / **语言下拉框。** 在 `Low VRAM` 右边。列表在 `configs\Easy3D_Languages.json` 里，改那个文件就能加 / 删 / 改名，不用改 GUI 代码。24 种语言，不含俄语。
- **Save Warped Image removed.** No `_WARPED` output anymore. / **`Save Warped Image` 已删。** 不再出 `_WARPED` 文件。
- **Save Depth Map moved.** From the top Input row to the button row, right of `Limit to Source`. Still outputs `_DEPTH.jpg`. / **`Save Depth Map` 移位置了。** 从顶行 Input 区移到按钮行、`Limit to Source` 右边。勾上还是出 `_DEPTH.jpg`。
- **Low VRAM renamed.** `Low VRAM (3070 8GB)` → `Low VRAM`. Behaviour unchanged, on by default. / **`Low VRAM` 改名。** `Low VRAM (3070 8GB)` → `Low VRAM`。功能没变，默认开。
- **One venv, relative paths.** The only environment is `venv\` (built by `1_MAKE_VENV.bat`). All installers use relative paths: copy the whole Easy3D folder anywhere and run. / **一个 venv，全相对路径。** 唯一环境是 `venv\`（`1_MAKE_VENV.bat` 建）。所有安装文件都用相对路径，整个 Easy3D 文件夹拷到哪都能跑。
- **Installers split by job.** `1_MAKE_VENV.bat` = venv + pip / torch. `2_install_all.bat` = 13 models. `3_Install_DA3.bat` = DA3 code + DA3-S / B / L weights. `4_Install_Distill.bat` = Distill-B deps + weight. Then `CHECK_ENV.bat` to verify, `Launch_Easy3D_GUI.bat` to start (last step, daily use). `INSTALL_FIX.bat` is repair only, not an install step. / **安装文件按活分开。** `1_MAKE_VENV.bat` = venv + pip / torch。`2_install_all.bat` = 13 个模型。`3_Install_DA3.bat` = DA3 代码 + DA3-S / B / L 权重。`4_Install_Distill.bat` = Distill-B 依赖 + 权重。然后 `CHECK_ENV.bat` 验证，`Launch_Easy3D_GUI.bat` 启动（最后一步，日常用）。`INSTALL_FIX.bat` 只修不装，不是安装步骤。
- **Depth models in v10.9:** `FD-L / FD-S / DA3-L / DA3-B / DA3-S / DA2-L / DA2-B / DA2-S / Distill-B`. / **v10.9 的深度模型：** `FD-L / FD-S / DA3-L / DA3-B / DA3-S / DA2-L / DA2-B / DA2-S / Distill-B`。

---

## Quick Start · 快速上手

1. Double-click `1_MAKE_VENV.bat`, wait until it finishes. / 双击 `1_MAKE_VENV.bat`，等它跑完。
2. Double-click `2_install_all.bat` — downloads the 13 models. / 双击 `2_install_all.bat` — 下 13 个模型。
3. Double-click `3_Install_DA3.bat` — installs DA3 code + DA3-S / B / L weights. / 双击 `3_Install_DA3.bat` — 装 DA3 代码 + DA3-S / B / L 权重。
4. (Optional, only if you use Distill-B) Double-click `4_Install_Distill.bat`. / （可选，只用 Distill-B 才需要）双击 `4_Install_Distill.bat`。
5. Double-click `CHECK_ENV.bat` — should show `torch 2.6.0+cu124 cuda: True`. / 双击 `CHECK_ENV.bat` — 应显示 `torch 2.6.0+cu124 cuda: True`。
6. Double-click `Launch_Easy3D_GUI.bat` — this is the last step, and the one you use every day. / 双击 `Launch_Easy3D_GUI.bat` — 这是最后一步，也是以后每天用的那一个。

> `INSTALL_FIX.bat` is NOT an install step. Do NOT run it after a fresh install. Only run it when the venv is broken / a package is missing. / `INSTALL_FIX.bat` 不是安装步骤，刚装完不要跑它。只有 `venv` 坏了 / 缺包的时候才跑它。

> If install / run has any problem, post it at [Issues](https://github.com/KolaKater/Easy3D-Converter/issues) or [Discussions](https://github.com/KolaKater/Easy3D-Converter/discussions) — include your `CHECK_ENV.bat` output, the exact error text, and your GPU / Windows version. / 安装 / 运行有任何问题，直接去 [Issues](https://github.com/KolaKater/Easy3D-Converter/issues) 或 [Discussions](https://github.com/KolaKater/Easy3D-Converter/discussions) 发帖，贴上 `CHECK_ENV.bat` 的输出、完整的报错文字、显卡 / Windows 版本。
7. **Input:** Click **Browse File / Browse Folder**, or drag & drop an image / video / folder onto the Input box. / **输入：** 点 **Browse File / Browse Folder**，或者把图片 / 视频 / 文件夹拖到输入框。
8. **Output:** Leave **Default (InputName_3D)**. Output goes to `<InputName>_3D\` next to the input. / **输出：** 默认 **Default (InputName_3D)** 就行，输出在输入旁边 `<InputName>_3D\` 文件夹里。
9. Set the parameters below, then click **Start**. Watch **Log / Preview / File List**. **Suspend** pauses, **Cancel** stops. / 按下面设好参数，点 **Start**。看 **Log / Preview / File List**。**Suspend** 是暂停，**Cancel** 是停止。

**Recommended first run (RTX 3070+ / 第一次推荐参数)：**

```
Mode:         Fast - Decoupled C
Finetuned:    Finetune-v2.1
Depth Model:  FD-L
Resolution:   1344 - Finetune-v2.1 Recommended
Steps:        5  - Blink preview
3D Strength:  0.40
Follow Original: 1.80
Seed:         Seed A
Gen Mode:     Orig_Right
Face Restore: None
Image Format: 1. Parallel (Default)
```

The filename already contains all your settings, for example:
文件名里已经有全部参数，例如：

```
White Rose_Easy3D_V21_FDL_R1024_B040_G180_PAR_Seed402260655.jpg
```

> **Tip: start with landscape photos.** Easy3D works best with landscape (wide) images. Portrait images work, but usually look flatter. See [FAQ](#faq--常见问题)。
> **建议：先用横屏照片。** Easy3D 最适合横屏（宽）图片。竖屏也能用，但通常看着比较平。见 [FAQ](#faq--常见问题)。

---

## How to Pick: Mode + Resolution · 怎么选：模式 + 分辨率

You only need to remember two choices: **Mode** decides speed vs. quality, **Resolution** decides how fine the result is. Pick them together, and the result will look better.

你只需要记住两个选择：**Mode（模式）决定快慢和质量，Resolution（分辨率）决定精细度。** 两个一起配，效果才好。

### Mode / 模式

| Mode / 模式 | What it feels like / 用起来像什么 | Use it when / 什么时候用 |
|---|---|---|
| **Quality** | Cleanest. Slower than Fast. / 最干净。比 Fast 慢。 | You want the best result. / 想要最好效果。 |
| **Fast - Decoupled A** | Fast. / 快。 | Daily use, quick preview. / 日常出图、快速预览。 |
| **Fast - Decoupled B** | Faster than Quality, finer than A. / 比 Quality 快，比 A 细。 | You want a balance of speed and quality. / 想要速度和质量都兼顾。 |
| **Fast - Decoupled C** | Finer than B, still faster than Quality. / 比 B 更细，仍比 Quality 快。 | You want a finer Fast result. / 想要 Fast 里更细的效果。 |
| **Fast - Decoupled D** | Finer than C, still faster than Quality. / 比 C 更细，仍比 Quality 快。 | Full-body photos, or you want the finest Fast result. / 全身像，或者想要 Fast 里更细的效果。 |
| **Fast - Decoupled E** | Finer than D. / 比 D 更细。 | Large / detailed scenes, you want even finer. / 大场景、细节多的图，想要更细。 |
| **Fast - Decoupled F** | Finest Fast mode, still faster than Quality. / Fast 里最细的，仍比 Quality 快。 | You want the finest Fast result. / 想要 Fast 里最细的效果。 |

### If there is a person in the photo / 照片里有人

**Selfie / half-body → Decoupled A or B + Face Restore. / 自拍 / 半身像 → 选 Decoupled A 或 B，并加上 Face Restore。**

**Full-body → Decoupled C or D + Face Restore. / 画面上有全身像 → 建议选 C 或 D，并加上 Face Restore。**

If you skip Face Restore on a full-body photo, the face can go blurry — the person is small in the frame, so the face is small too.
全身像不加 Face Restore，脸可能会很模糊 — 人在画面里小，脸也小。

> **One rule: if there is a person, turn Face Restore on. / 记住一条就行：有人就加 Face Restore。**

> **One more rule: do not upscale a small source.** If your source image is only 2048 px, running at 4096 px will not add detail — it only upscales. Use 2048 px or below for a 2048 px source. Use 2560 / 3072 / 4096 px only when the source itself is 3K / 4K.
> **还有一条：原图小，别硬拉大。** 原图只有 2048 px，跑 4096 px 也不会多出细节，只是放大。原图 2048 px 就用 2048 px 或以下；原图本身是 3K / 4K 再用 2560 / 3072 / 4096 px。

### Resolution labels / Resolution 标签怎么看

Each item in the Resolution dropdown has 3 parts: `number - which Finetuned it fits + speed / recommendation`.
Resolution 下拉框里每一项是三段式：`数字 - 最配哪个 Finetuned + 快慢 / 推荐`。

| Word in label / 标签里的词 | Meaning / 意思 |
|---|---|
| `Finetune-v1.5` | Best for `Finetune-v1.5`. / 最配 `Finetune-v1.5`。 |
| `Finetune-v2.1` | Best for `Finetune-v2.1`. / 最配 `Finetune-v2.1`。 |
| `Common` | Works for both. / 两个都能用。 |
| `Fast` | Fastest, least detail, good for preview. / 最快，细节最少，适合预览。 |
| `Recommended` | Best quality / speed balance for daily use. / 日常推荐：质量 / 速度最划算。 |
| `Slowest` | Slowest, most detail, use sparingly. / 最慢，细节最多，慎用。 |

**Example / 举例 — `1344 - Finetune-v2.1 Recommended`：** Resolution is `1344`, best for `Finetune-v2.1`, and recommended for daily use. / 分辨率是 `1344`，最配 `Finetune-v2.1`，而且是日常推荐。

So: if Finetuned is `Finetune-v2.1`, pick `1344` for daily use. If Finetuned is `Finetune-v1.5`, pick `1280 - Finetune-v1.5 Recommended`.
所以：Finetuned 选 `Finetune-v2.1`，日常优先选 `1344`；选 `Finetune-v1.5`，日常优先选 `1280 - Finetune-v1.5 Recommended`。

### Finetuned Model / Finetuned 模型：保守选 v1.5，要创意选 v2.1

`Finetuned` 不是新旧之分，是**保守 / 创意**之分。同一个 `Seed` 跑出来，它俩也是两张独立生成的图，不是谁的高清版。

| Finetuned / 模型 | What it does / 性格 | Use it when / 什么时候用 |
|---|---|---|
| **Finetune-v1.5** | Conservative / 保守。更听原图的话，不敢乱编。 / 更保守，更贴原图，不敢乱编。 | **要保真的时候。** 人脸、自拍、半身像、手指、花瓣、花纹、文字 — 脸不能崩、纹理不能编的地方。 / **Want it faithful.** Portraits, selfies, half-body, hands, petals, patterns, text — where the face cannot drift and the texture cannot be invented. |
| **Finetune-v2.1** | Creative / 有创意。更敢自己补，细节更多。 / 更敢创作，细节更多。 | **要创意 / 要细节的时候。** 风景、大场景、树叶、头发丝、衣服褶皱、远处建筑、遮挡严重的边缘 — `v1.5` 糊成一团的地方。 / **Want it creative / detailed.** Landscapes, large scenes, foliage, hair, fabric folds, distant buildings, heavy occlusion — where `v1.5` goes blurry. |

**一句话记住 / One line to remember：**

> **要创意，选 v2.1；要保守，选 v1.5。/ Creative → v2.1, conservative → v1.5.**

**怎么配 / How to pair it：**

- **人像 / 有脸**：`Finetune-v1.5` + `Decoupled A / B` + `Face Restore`。脸不崩是第一位的，细节靠 Face Restore 补。/ **Portrait / face:** `Finetune-v1.5` + `Decoupled A / B` + `Face Restore`.
- **风景 / 大场景**：`Finetune-v2.1` + `Decoupled C / D / E / F`。脸不是主角，细节多才是赢。/ **Landscape / large scene:** `Finetune-v2.1` + `Decoupled C / D / E / F`.
- **拿不准**：先跑 `v1.5`，脸保住了再说；觉得糊，再换 `v2.1` 跑一张对比。/ **Not sure:** run `v1.5` first to keep the face safe, then try `v2.1` and compare.

> **注意 / Note：** `v1.5` 更快、更干净，更像原图 (Faster, cleaner, looks more like your original)。`v2.1` 细节更多、更清晰，但有时候会自己改东西——比如脸变了、手指多了、花纹改了 (More detail, sharper, but sometimes changes things - face, fingers, patterns)。看到 `v1.5` 效果好，不代表 `v2.1` 坏了，只是那张图更适合“像原图”，不适合“多细节”。(If `v1.5` looks better, it doesn't mean `v2.1` is bad - that image just needs "looks like the original", not "more detail".)

---

## Parameters · 参数

You do not need to tune everything. Start with the Quick Start settings above, then only change one thing at a time.
不用每个都调。先用上面快速上手的设置，然后一次只改一个。

### 3D Strength / 3D 强度

Overall stereo strength. How strong the 3D pops. Suggested 0.5. Larger = more 3D, but more ghosting / misalignment.
整体立体强度，3D 出屏有多强。建议 0.5。越大越立体，但重影、错位也越多。

| Value / 值 | Effect / 效果 |
|---|---|
| 0.2 | Very weak stereo. / 立体感很弱。 |
| **0.40** | **Normal (recommended). / 正常（推荐）。** |
| 0.5 | A bit stronger, still comfortable. / 稍强一点，还舒服。 |
| 1.0+ | Too strong — dizzy, ghosting. / 太强了：头晕、重影。 |

> **Video: keep 3D Strength at or below 1.0. / 视频：3D Strength 建议不要超过 1.0。**
> In video the disparity jumps frame to frame, so a strength that looks fine on a still photo turns into flicker, swimming edges, and eye strain once it moves. 1.0 is the ceiling for video — above it the 3D stops looking stronger and starts looking broken. / 视频里视差是逐帧跳的，静态图上看着还行的强度，一动起来就变成闪烁、边缘游移、眼睛累。1.0 是视频的上限，超过它 3D 不是更强，是更坏。

### Follow Original / 跟原图

**一句话 / In one line：** `Follow Original` 是 `Easy3D` 在“3D 转换器”和“3D 创作者”之间切换的开关。/ `Follow Original` is the switch between “3D converter” and “3D creator”.

| 值 / Value | Easy3D 变成什么 / What Easy3D becomes | 画面什么样 / What the picture looks like |
|---|---|---|
| `0.0` | 纯创作者 / Pure creator | 完全不听原图的话，放飞。光 / 火 / 水 / 雾重画得很漂亮，但人脸、建筑也敢重画，走样、变脸、颜色跑偏全来了。/ Ignores the original completely. Light / fire / water / fog repainted beautifully, but faces and buildings get repainted too — distortion, wrong faces, colour drift. |
| `1.0` | 创作者为主 / Mostly creator | 还是敢创作，但开始听原图的话了。风景、光影效果好，人像要小心。/ Still dares to create, but starts listening to the original. Good for landscapes / light, careful with portraits. |
| **`1.8`** | **平衡点 / Balanced — 建议值 / Recommended** | **既创作，又不走样。光 / 火 / 水 / 雾还能重画，人 / 建筑 / 脸还认得出来。日常就用它。/ Creates, but does not drift. Light / fire / water / fog still get repainted, people / buildings / faces still look like themselves. Daily default.** |
| `2.2` | 转换器为主 / Mostly converter | 创作收敛了，贴原图了。人像脸崩 / 颜色跑偏时，先拉到这里救一下。/ Creation pulls back, sticks to the original. Use this first when a portrait face breaks or colour drifts. |
| `5.0` | 几乎纯转换器 / Almost pure converter | 几乎不创作了，只敢把原图像素左右挪一下，变成 SBS。跟普通 2D 转 3D 工具没区别了 — 光变平、火变平、雾变平，你用 `Easy3D` 干嘛？/ Almost no creation left, only shifts the original pixels left / right into SBS. Same as any ordinary 2D-to-3D converter — light goes flat, fire goes flat, fog goes flat. Then why use `Easy3D` at all? |

**怎么调？/ How to tune it?**

- **日常、风景、光 / 火 / 水 / 雾**：`1.8`，不用动。/ **Daily, landscapes, light / fire / water / fog**: `1.8`, do not touch it.
- **人像、自拍、脸崩了、颜色跑了**：`1.8 → 2.2`，先拉到这里，别一下拉到 `5.0`。/ **Portrait / selfie, face broken, colour drifted**: `1.8 → 2.2` first, do not jump to `5.0`.
- **想要它大胆创作**：`1.8 → 1.2` / `1.0`，光影会更漂亮，但要一张张检查，脸和建筑可能走样。/ **Want it to create boldly**: `1.8 → 1.2` / `1.0`, light will look prettier, but check every image — faces and buildings may drift.
- **想要它老老实实转换**：`1.8 → 2.5` / `3.0`，别到 `5.0`。`5.0` 不是更保真，是更僵硬，而且把 `Easy3D` 最值钱的创作能力关掉了。/ **Want it to convert honestly**: `1.8 → 2.5` / `3.0`, not `5.0`. `5.0` is not more faithful, it is more stiff, and it switches off the creation ability that makes `Easy3D` worth using.

> **新手只记一句 / If you only remember one line：**
>
> `Follow Original` 小 = 创作强，`Follow Original` 大 = 转换强。想要 `Easy3D` 帮你创作，就别把缰绳拉太紧。/ Small `Follow Original` = strong creation, large `Follow Original` = strong conversion. If you want `Easy3D` to create for you, do not pull the leash too tight.

`Follow Original` **不影响速度** — 它是权重，不是步数。/ It does **not** affect speed — it is a weight, not a step count.

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

Face Restore is **images only**. Video input forces Face Restore to `None`. / Face Restore **只对图片生效**。视频输入会强制把 Face Restore 关掉。

### Edge Protection / 边缘保护

Edge Protection keeps object edges clean, so edges do not bleed, tear, or smear into the background.
Edge Protection 是让物体边缘更干净的，不让边缘糊进背景、撕裂、拖影。

| Edge Protection Level | Effect / 效果 |
|---|---|
| 0 | Off / 关 — edges smear most easily. / 边缘最容易糊。 |
| **2–4** | **Enough in most cases / 多数情况下就够了** — daily default, use this range. / 日常就用这个范围。 |
| 5+ | Stronger, but edges can go too hard. / 更强，但边缘容易修得太硬。 |

> **Daily use: just leave it at 2–4. / 日常不用调：2–4 就行。**

### Depth Model (MDE) / 深度模型

MDE makes the depth map. It decides the geometry, not the final look. / MDE 是做深度图的，决定几何，不决定最终画面的风格。

> **GUI 里的名字就是这样写的 / Names in the GUI:** `FD-L` = FlashDepth-L，`FD-S` = FlashDepth-S，`DA2-L / DA2-B / DA2-S` = Depth Anything V2 Large / Base / Small，`DA3-S / DA3-B / DA3-L` = Depth Anything 3 Small / Base / Large，`Distill-B` = Distill Any Depth Base。

| GUI option / 选项 | Quality / 质量 | Speed / 速度 | Use for / 适合 |
|---|---|---|---|
| **FD-L (default / 默认)** | Good / 好 | Fast / 快 | **Daily default. / 日常默认就是它。** |
| FD-S | Slightly weaker / 稍弱 | Faster / 更快 | 8 GB + high Resolution (>3000) to avoid OOM. / 8 GB + 高分辨率（>3000）时用它，防 OOM。 |
| DA2-L | Good / 好 | Slow / 慢 | Try for a different look. Do not use Large on 8 GB + >3000. / 想换个风格时试。8 GB + >3000 别用 Large。 |
| DA2-B | Medium / 中 | Medium / 中 | Alternative on 8 GB + high Resolution (>3000). / 8 GB + 高分辨率（>3000）时的备选。 |
| DA2-S | Weaker / 弱 | Fastest / 最快 | Most stable on 8 GB + high Resolution (>3000). / 8 GB + 高分辨率（>3000）时最稳。 |
| **DA3-B** | Very good / 很好 | Medium / 中 | Cleaner detail and edges. Try it for a different depth style. / 细节和边缘更干净，想换个深度风格时试。 |
| **DA3-S** | Good / 好 | Fast / 快 | More stable on 8 GB + high Resolution. / 8 GB + 高分辨率时更稳。 |
| **DA3-L** | Very good / 很好 | Slow / 慢 | Finest DA3, do not use on 8 GB + >3000. / DA3 里最细，8 GB + >3000 别用。 |
| **Distill-B** | Good / 好 | Medium / 中 | Lightweight alternative. / 轻量备选。 |

> **8 GB rule (important) / 8 GB 规则（重要）：**
> - Resolution `≤ 3000`: any MDE is fine, default `FD-L` works. / Resolution `≤ 3000`：随便选，默认 `FD-L` 就行。
> - Resolution `> 3000` (e.g. 3072 / 3584 / 4096 / 5120): **switch MDE to Base / Small first** — `FD-S`, or `DA2-B` / `DA2-S`. / Resolution `> 3000`（如 3072 / 3584 / 4096 / 5120）：**先把 MDE 换成 Base / Small** — `FD-S`，或 `DA2-B` / `DA2-S`。

### Low VRAM / 低显存

On by default. Recommended for RTX 3070 8GB. / 默认开。RTX 3070 8GB 推荐开。

### Save Depth Map / 保存深度图

Optional, in the button row, right of `Limit to Source`. Saves a `_DEPTH.jpg` alongside the SBS. / 可选，在按钮行、`Limit to Source` 右边。会在 SBS 旁边多存一张 `_DEPTH.jpg`。

Video input forces Save Depth Map OFF. / 视频输入会强制关掉 Save Depth Map。

### Image Format / 图片格式

| Option / 选项 | Meaning / 意思 |
|---|---|
| **1. Parallel (Default / 默认)** | For parallel viewing / VR. / 平行眼看 / VR 看。 |
| 2. Cross-Eyed | For cross-eyed viewing. / 斗鸡眼看。 |
| 3. Auto-Align Parallel / XD | Auto-aligned. / 自动对齐。 |

### Video Format / 视频格式

Video only. `SBS (Parallel) / XD (Cross-Eyed) / Red-Green / Red-Blue`. / 只对视频生效。`SBS (Parallel) / XD (Cross-Eyed) / Red-Green / Red-Blue`。

Video restrictions: MDE forced to FlashDepth-S, Seed forced to a fixed value, Face Restore forced to `None`, Gen Mode forced to `Orig_Right`, Image Format forced to Parallel, Save Depth Map forced OFF, `Limit to Source` forced ON. / 视频限制：MDE 强制 FlashDepth-S，Seed 强制固定值，Face Restore 强制 `None`，Gen Mode 强制 `Orig_Right`，Image Format 强制 Parallel，Save Depth Map 强制关，`Limit to Source` 强制开。

Video forces FlashDepth because **only FlashDepth has a video stabilizer** (temporal smoothing + per-chunk reset, so depth does not jump frame to frame). Other depth models are per-image only — on video their depth flickers, and the 3D flickers with it. / 视频强制 FlashDepth，是因为**只有 FlashDepth 带视频稳定器**（跨帧平滑 + 每组重置，深度不会逐帧乱跳）。其他深度模型都是按单张图算的，放视频里深度会闪，3D 也跟着闪。

### Output Naming / 文件名

Always JPG, `quality=95`. The filename already contains all settings, so you can reproduce any result from the filename alone.
永远是 JPG，`quality=95`。文件名里已经有全部参数，只看文件名就能复现。

---

## Language · 语言

The Language dropdown is right of `Low VRAM`. Pick a language and the UI changes at once, no restart. / 语言下拉框在 `Low VRAM` 右边。选了语言，界面立刻变，不用重启。

The list is **not** hardcoded in the GUI. It lives in `configs\Easy3D_Languages.json`. / 列表**不在** GUI 代码里写死，它在 `configs\Easy3D_Languages.json` 里。

```json
{
  "default_label": "English",
  "languages": [
    { "label": "English", "code": "en" },
    { "label": "Chinese Simplified - 简体中文", "code": "zh-Hans" },
    { "label": "Chinese Traditional - 繁體中文", "code": "zh-Hant" },
    { "label": "German - Deutsch", "code": "de" },
    { "label": "French - Français", "code": "fr" },
    { "label": "Japanese - 日本語", "code": "ja" },
    { "label": "Korean - 한국어", "code": "ko" }
  ]
}
```

- **Add a language:** copy one entry in `languages`, change `label` and `code`. / **加语言：** 在 `languages` 里复制一行，改 `label` 和 `code`。
- **Remove a language:** delete its entry. / **删语言：** 删掉那一行。
- **Rename:** only change `label`. / **改名：** 只改 `label`。
- **Change the default:** change `default_label` to a `label` that already exists in the list. / **改默认：** 把 `default_label` 改成列表里已有的任意一个 `label`。

Default language is `English`. / 默认语言是 `English`。

If `configs\Easy3D_Languages.json` is missing or broken, the GUI falls back to its built-in list so it still opens. / 如果 `configs\Easy3D_Languages.json` 丢了或写坏了，GUI 会用内置列表兜底，还能打开。

---

## Requirements · 配置要求

| Item / 项目 | Minimum / 最低 | Recommended / 推荐 |
|---|---|---|
| OS / 系统 | Windows 10 / 11 (64-bit) | Windows 11 |
| Python | 3.11.9 | 3.11.9 |
| GPU / 显卡 | NVIDIA RTX 2060 6GB | **RTX 3070 8GB** or above / 或以上 |
| Supported RTX / 支持的型号 | RTX 2060 / 2070 / 2080 (Ti) — minimum, slow, low resolution only / 最低，能跑但慢，只能低分辨率 | **RTX 3070 / 3070 Ti / 3080 / 3080 Ti / 3090 / 3090 Ti / 4060 / 4060 Ti / 4070 / 4070 Ti / 4080 / 4080 Super / 4090 — fully supported / 完全支持** |
| NOT supported / 不支持 | **RTX 50 series (5070 / 5080 / 5090) — bundled torch has no Blackwell kernels, falls back to CPU** / **50 系 — 安装包自带的 torch 没有 Blackwell 内核，会回落到 CPU** | See [GPU Compatibility](#gpu-compatibility--显卡兼容) below / 见下面显卡兼容 |
| AMD / Intel / No GPU | No CUDA — CPU mode only, very slow / 无 CUDA，只能 CPU 模式，非常慢 | — |
| VRAM / 显存 | 6 GB (512 px) | 8 GB+ (512–4096 px) |
| Disk / 硬盘 | ~25 GB | ~30 GB |
| CUDA | Driver with CUDA 12.4+ | Latest NVIDIA driver / 最新驱动 |

> Training is **not** supported on a consumer GPU. This project is inference only. / 普通显卡**不支持**训练，这个项目只做推理。

### GPU Compatibility · 显卡兼容

**Supported / 支持：** RTX 30 series (3070 and above) + RTX 40 series (4060 and above). The installer bundles `torch 2.6.0+cu124`, which has CUDA kernels for Ampere (sm_86) and Ada Lovelace (sm_89). / 安装包自带 `torch 2.6.0+cu124`，有 Ampere (sm_86) 和 Ada Lovelace (sm_89) 的 CUDA 内核。

**RTX 50 series (Blackwell, sm_120) — NOT supported by the installer / 50 系安装包不支持：**
- The bundled torch has no Blackwell kernels. Forcing install on a 5080 / 5090 will either fail with `no kernel image is available for execution on the device`, or silently fall back to CPU. / 安装包的 torch 没有 Blackwell 内核。强行在 5080 / 5090 上装，要么报 `no kernel image is available`，要么默默回落到 CPU。
- CPU fallback is 10-50x slower than GPU. A single image can take minutes, video is impractical. / CPU 回落比 GPU 慢 10-50 倍，一张图可能要几分钟，视频没法用。
- If you have a 50 series card, build from source with `torch 2.7.1+cu128` instead of using the installer (see [Installation](#installation--安装) step 2). / 50 系请不要用安装包，按源码安装那套装 `torch 2.7.1+cu128`。

**Forcing install on other unsupported GPUs / 其他不支持的卡强行安装：**
- **VRAM < 8 GB** (e.g. RTX 2060 6GB, GTX 1660): will run out of memory (OOM) at high resolution. Lower Resolution to 512-1024 and enable Low VRAM. / **显存 < 8 GB**（如 RTX 2060 6GB、GTX 1660）：高分辨率会爆显存。把 Resolution 降到 512-1024，并开 Low VRAM。
- **GTX 10 / 16 series, RTX 20 series (Turing)**: may run, but no Tensor core acceleration for some ops, slow. VRAM limits resolution. / **GTX 10 / 16 系、RTX 20 系**：可能能跑，但部分算子没有 Tensor core 加速，慢；显存限制分辨率。
- **AMD GPU / Intel integrated / no GPU**: no CUDA, falls back to CPU. Works for testing one small image, not for batch or video. / **AMD / 核显 / 无独显**：无 CUDA，回落 CPU。测一张小图可以，批量和视频没法用。
- **Old NVIDIA driver** (CUDA < 12.4): `torch.cuda.is_available()` returns False. Update to the latest driver from nvidia.com. / **旧驱动**（CUDA < 12.4）：`torch.cuda.is_available()` 返回 False。去 nvidia.com 更新最新驱动。

---

## Installation · 安装

All paths below are **relative** to the Easy3D folder. Copy the folder anywhere and run. / 下面所有路径都是相对路径，相对 Easy3D 文件夹。文件夹拷到哪都能跑。

Do **not** use a path with Chinese characters or parentheses if you use the `.bat` launchers — `C:\3D\Easy3D\` is a safe example.
如果用 `.bat` 启动器，路径里**不要**有中文或括号，`C:\3D\Easy3D\` 这样就安全。

### 1. Clone / Download / 克隆 / 下载

```bash
git clone https://github.com/KolaKater/Easy3D-Converter.git
cd Easy3D-Converter
```

Or download the ZIP and extract it. / 或者下载 ZIP 解压。

### 2. Create the Virtual Environment / 创建虚拟环境

Double-click / 双击：

```
1_MAKE_VENV.bat
```

This step installs the matching PyTorch CUDA build into `venv\`. / 这一步把对应的 PyTorch CUDA 版本装进 `venv\`。

- RTX 50 series → `torch 2.7.1+cu128`
- RTX 20 / 30 / 40 series → `torch 2.6.0+cu124`
- No NVIDIA GPU → CPU torch (slow, for testing only)

### 3. Download All Models / 下载所有模型

Double-click / 双击：

```
2_install_all.bat
```

Downloads the 13 models (Stereo 3D v1.5 + v2.1, depth models, image encoder, VAE, face repair weights). Skips files that already exist. Never re-downloads, never writes to `checkpoints\`. / 下 13 个模型（立体 3D v1.5 + v2.1、深度模型、图像编码器、VAE、人脸修复权重）。已有的跳过，不重下，不写到 `checkpoints\`。

Check: `13/13 OK` = all models present. / 检查：`13/13 OK` = 13 个都齐了。

DA3 is **not** in those 13. Use the next step for DA3. / DA3 **不在**这 13 个里，DA3 用下一步。

### 4. Install DA3 / 装 DA3

Double-click / 双击：

```
3_Install_DA3.bat
```

Installs `depth_anything_3` code (`--no-deps`) + minimal deps, then checks DA3-S / B / L weights. / 装 `depth_anything_3` 代码（`--no-deps`）+ 最小依赖，然后检查 DA3-S / B / L 权重。

- Default: install code + check weights, do **not** auto-download weights. / 默认：装代码 + 检查权重，**不**自动下权重。
- To download missing DA3 weights: `venv\Scripts\python.exe Install_DA3.py --download-models` / 要下缺的 DA3 权重：`venv\Scripts\python.exe Install_DA3.py --download-models`
- Check only: `venv\Scripts\python.exe Install_DA3.py --check-only` / 只检查：`venv\Scripts\python.exe Install_DA3.py --check-only`

### 5. Install Distill (optional) / 装 Distill（可选）

Only if you use `Distill-B`. Double-click / 双击：

```
4_Install_Distill.bat
```

Installs Distill deps (only the missing ones) + auto-downloads the Distill-B weight (~390 MB) if missing. / 装 Distill 依赖（只装缺的）+ 权重缺了就自动下（约 390MB）。

### 6. Verify / 验证

```
CHECK_ENV.bat
```

You should see / 应该看到：

```
torch: 2.6.0+cu124 cuda: True
```

If `cuda: False`, you installed CPU torch: delete `venv\`, re-run `1_MAKE_VENV.bat`. If a package is missing, run `INSTALL_FIX.bat`. / 如果是 `cuda: False`，就是装成 CPU 版 torch 了：删掉 `venv\`，重跑 `1_MAKE_VENV.bat`。如果缺包，跑 `INSTALL_FIX.bat`。

### 7. Launch (daily use) / 启动（每天用）

Double-click / 双击：

```
Launch_Easy3D_GUI.bat
```

This is the last step. It uses the same `venv` python as `CHECK_ENV.bat`. Do NOT double-click the `.py` file, Windows may pick another Python. / 这是最后一步。它用的就是 `CHECK_ENV.bat` 那个 `venv` 里的 Python。不要直接双击 `.py` 文件，Windows 可能会挑错 Python。

---

## Get Help / Support · 求助

If install or run has any problem, post it at **Issues** or **Discussions** — do not just re-run the installers. / 安装 / 运行有任何问题，直接去 **Issues** 或 **Discussions** 发帖，别只反复重装。

- Issues: https://github.com/KolaKater/Easy3D-Converter/issues
- Discussions: https://github.com/KolaKater/Easy3D-Converter/discussions

When you post, include / 发帖时带上：

1. `CHECK_ENV.bat` output / `CHECK_ENV.bat` 的输出
2. The exact error text (screenshot or copy-paste, not “it doesn’t work”) / 完整的报错文字（截图或复制粘贴，别只写“不行”）
3. GPU + Windows version + which BAT failed (`1_MAKE_VENV` / `2_install_all` / `3_Install_DA3` / `4_Install_Distill` / `Launch`) / 显卡 + Windows 版本 + 哪一步失败的（`1_MAKE_VENV` / `2_install_all` / `3_Install_DA3` / `4_Install_Distill` / `Launch`）

---

## Troubleshooting · 故障排查

> Still stuck? Post at [Issues](https://github.com/KolaKater/Easy3D-Converter/issues) or [Discussions](https://github.com/KolaKater/Easy3D-Converter/discussions) with your `CHECK_ENV.bat` output + exact error text. / 还是不行？去 [Issues](https://github.com/KolaKater/Easy3D-Converter/issues) 或 [Discussions](https://github.com/KolaKater/Easy3D-Converter/discussions) 发帖，带上 `CHECK_ENV.bat` 输出 + 完整报错。

| Problem / 问题 | Fix / 解决办法 |
|---|---|
| `torch.cuda.is_available() == False` / CPU torch | You installed CPU torch. Delete `venv\`, re-run `1_MAKE_VENV.bat`. / 装成 CPU 版 torch 了。删掉 `venv\`，重跑 `1_MAKE_VENV.bat`。 |
| `pillow` / `numpy` errors | Do not upgrade them. `1_MAKE_VENV.bat` pins the right versions: `numpy 1.26.4`, `pillow 10.4.0`. Never force `numpy>=2` in the main venv. / 别升级它们。`1_MAKE_VENV.bat` 会锁定正确版本：`numpy 1.26.4`，`pillow 10.4.0`。主 venv 里永远不要强制升 `numpy>=2`。 |
| `INSTALL_FIX.bat` / `CHECK_ENV.bat` | `CHECK_ENV.bat` only checks, it installs nothing. `INSTALL_FIX.bat` re-installs `requirements_easy3d.txt` without touching torch. Use `CHECK_ENV.bat` first, `INSTALL_FIX.bat` only if a package is missing. / `CHECK_ENV.bat` 只检查，不装东西。`INSTALL_FIX.bat` 只重装 `requirements_easy3d.txt`，不动 torch。先跑 `CHECK_ENV.bat`，缺包才跑 `INSTALL_FIX.bat`。 |
| `GFPGAN loaded: False` | Check `gfpgan/weights/` is complete. / 检查 `gfpgan/weights/` 是不是齐的。 |
| OOM on 3070 8GB at Resolution > 3000 | First switch **MDE to Base / Small**, then lower **Resolution** and **Steps**. / 先把 **MDE 换成 Base / Small**，再降 **Resolution** 和 **Steps**。 |
| `Output/_deferred/` left over after Cancel | Normal on Cancel. Delete it and re-run, or re-run with the same output folder. / Cancel 后留下来是正常的。删掉重跑，或者用同一个输出文件夹重跑。 |
| Portrait image looks flatter than landscape | Expected. See FAQ. / 预期的。见 FAQ。 |
| `Flux.2 Klein` cannot be loaded | Only `Finetune-v1.5` / `Finetune-v2.1` are supported. Flux is a different architecture. / 只支持 `Finetune-v1.5` / `Finetune-v2.1`。Flux 是另一种结构。 |
| GUI shows `Low VRAM` but the old guide says `Low VRAM (3070 8GB)` | That is the v10.9 rename. Same option, same behaviour. / 那是 v10.9 改名。同一个选项，功能一样。 |
| `Save Warped Image` is gone | Removed in v10.9. No `_WARPED` output anymore. Use `Save Depth Map` for `_DEPTH.jpg`. / v10.9 删掉了。不再出 `_WARPED` 文件。要深度图用 `Save Depth Map`，出 `_DEPTH.jpg`。 |

---

## FAQ · 常见问题

**Q: Can I use it for video? / 能做视频吗？**
Yes. v10.9 supports image + video in one GUI. Video has restrictions (see Video Format above). / 可以。v10.9 一个 GUI 同时支持图片 + 视频。视频有自己的限制（见上面 Video Format）。

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

**Q: Why can't I change Resolution after selecting Fast-Decoupled? / 为什么一选 Fast-Decoupled 就不能选 Resolution 了？**
Because Fast-Decoupled and Resolution conflict. When Fast-Decoupled is selected, Resolution cannot be selected. / 因为 Fast-Decoupled 和 Resolution 会冲突。选了 Fast-Decoupled，就不可以选 Resolution。

**Q: My full-body photo's face looks blurry. / 全身像的脸很模糊，怎么办？**
Use **Decoupled C or D + Face Restore**. Full-body means the face is small in the frame, so it needs the finer mode + Face Restore. See [How to Pick](#how-to-pick-mode--resolution--怎么选模式--分辨率).
用 **Decoupled C 或 D + Face Restore**。全身像里脸本来就小，要用更细的模式 + Face Restore。见 [怎么选](#how-to-pick-mode--resolution--怎么选模式--分辨率)。

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
| **Depth Anything 3 (DA3)** | Depth Anything Team | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) — see its repo for license |
| **Distill Any Depth** | Distill Any Depth Authors | [xingyang1/Distill-Any-Depth](https://huggingface.co/xingyang1/Distill-Any-Depth) — see its repo for license |
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
