# img2img — Anime character pipeline

Two-step generation: **FLUX Kontext** (character-consistent base) → **ModelsLab Counterfeit V3** (anime refinement). Plus a dedicated **IP-Adapter + ControlNet** mode for transferring identity/style from one image onto another's structure.

## Modes

### 1. Free-angle (no pose ref)
```bash
python3 img2img.py --character vek.jpeg Fullbody.png --angle front
python3 img2img.py --character vek.jpeg --angle three-quarter
```
Angles: `front`, `back`, `side-left`, `side-right`, `three-quarter`, `low`, `high`, `closeup`.

### 2. Pose transfer
```bash
python3 img2img.py --character vek.jpeg Fullbody.png --pose imgpose.webp
```
Stitches pose ref (left) + character refs (right) into a single composite for FLUX Kontext.

### 3. Single-image edit
```bash
python3 img2img.py vek.jpeg --prompt "same character sitting on a bench"
```

### 4. Anime refinement (Counterfeit V3 by default)
Add `--refine` to any mode above to push the FLUX output through ModelsLab Counterfeit V3.
```bash
python3 img2img.py --character vek.jpeg --angle front --refine
python3 img2img.py --character vek.jpeg --angle front --refine --model anything
```
Aliases: `counterfeit`, `anything`, `pony`, `dreamshaper`. Default = Counterfeit V3.

### 5. IP-Adapter mode (NEW — character consistency)
Skips FLUX entirely. Goes direct to ModelsLab `/api/v5/controlnet` with IP-Adapter (identity) + ControlNet (structure).
```bash
python3 img2img.py TARGET.png --ip-image IDENTITY.png \
  --prompt "1girl, sitting at a desk, anime style"
```
- **`TARGET.png`** — provides structure (canny edges by default)
- **`--ip-image IDENTITY.png`** — provides identity/style ("from" character)

## IP-Adapter flags

| Flag | Default | Purpose |
|---|---|---|
| `--ip-image PATH` | — | identity reference (triggers IP-Adapter mode) |
| `--ip-adapter ID` | `ip-adapter-plus_sd15` | use `ip-adapter-plus-face_sd15` for face-only |
| `--ip-scale 0–1` | `0.7` | how strongly it matches the reference |
| `--controlnet NAME` | `canny` | structure model (comma-separate for multi) |
| `--controlnet-scale 0–1` | `0.7` | structure conditioning strength |
| `--ip-strength 0–1` | `0.6` | denoising strength |
| `--ip-width` / `--ip-height` | `768` / `1024` | output size |
| `--seed N` | random | deterministic output |

### ControlNet options
- `canny` — edge detection (good for outfits)
- `openpose` — copy the body pose
- `depth` — keep 3D structure
- `lineart` — best for clean anime lines
- `face_detector` — lock face features

### Multi-ControlNet (locks character hardest)
```bash
python3 img2img.py target.png --ip-image vek.jpeg \
  --controlnet "openpose,canny,face_detector"
```

### Face-only swap
```bash
python3 img2img.py target.png --ip-image vek.jpeg \
  --ip-adapter ip-adapter-plus-face_sd15 --controlnet openpose
```

## Refinement model presets

The `modelslab_refine` function auto-detects the model and picks tuned settings:

| Detector | Model family | Quality tags | Steps | CFG | Strength | Scheduler |
|---|---|---|---|---|---|---|
| `is_pony` | Pony Diffusion | `score_9, score_8_up, ...` | 35 | 6.5 | 0.52 | UniPC |
| `is_counterfeit` | Counterfeit V3 | `(masterpiece:1.3), (best quality:1.2), ...` | 30 | 7.5 | 0.35 | DPM++ 2M |
| else | anything-v5, dreamshaper, ... | `masterpiece, best quality, ...` | 40 | 7 | 0.20 | UniPC |

Counterfeit V3 uses NAI-style emphasis weights (`(tag:weight)`) and a comprehensive anatomy-focused negative prompt.

## Endpoints used

| Mode | Endpoint |
|---|---|
| FLUX Kontext (step 1) | `https://queue.fal.run/fal-ai/flux-pro/kontext` |
| ModelsLab refine (step 2) | `https://modelslab.com/api/v6/images/img2img` |
| ModelsLab IP-Adapter | `https://modelslab.com/api/v5/controlnet` |
| ModelsLab job fetch | `https://modelslab.com/api/v6/images/fetch/{id}` |

**Important:** IP-Adapter requires the **v5 controlnet** endpoint, not v6 img2img. The v6 img2img endpoint silently ignores `ip_adapter_*` parameters.

## Output
All outputs go to `./out/`:
- `fal_<stem>_<timestamp>_<n>.png` — raw FLUX
- `fal_<stem>_<timestamp>_<n>_anime.png` — FLUX + ModelsLab refine
- `fal_<stem>_<timestamp>_<n>_flux.png` — refine failed, raw FLUX saved as fallback
- `ipadapter_<target>_from_<ref>_<timestamp>.png` — IP-Adapter mode
