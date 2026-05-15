"""img2yuki — two-step anime pipeline.

Step 1: fal.ai FLUX Kontext  → character-consistent image
Step 2: ModelsLab anything-v5 → high-quality anime refinement

Modes:
  Free-angle (no pose ref):
    python3 img2img.py --character vek.jpeg Fullbody.png --angle front
    python3 img2img.py --character vek.jpeg --angle three-quarter

  Pose transfer:
    python3 img2img.py --character vek.jpeg Fullbody.png --pose imgpose.webp

  Single-image edit:
    python3 img2img.py vek.jpeg --prompt "same character sitting on a bench"

  Skip anime refinement:
    python3 img2img.py --character vek.jpeg --angle front --no-refine

Angles: front, back, side-left, side-right, three-quarter, low, high, closeup
"""
import argparse
import base64
import io
import sys
import time
from pathlib import Path

import requests

# ========================= CONFIG =========================
FAL_KEY = "e828c075-e360-4c93-92cc-8b707f855be7:9057c91b475b8f5b7b93dfae86615fc2"
FAL_MODEL_ID = "fal-ai/flux-pro/kontext"
FAL_SUBMIT_URL = f"https://queue.fal.run/{FAL_MODEL_ID}"
FAL_HEADERS = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}

MODELSLAB_KEY = "T2JYbud4YBmdRGYIBRMCKeuusg5tVUpKIqMQvd2G667fvJ7dtPA6lEqEFTcC"
MODELSLAB_MODEL = "counterfeit-v3-0-v3-0"
MODELSLAB_URL = "https://modelslab.com/api/v6/images/img2img"
MODELSLAB_FETCH_URL = "https://modelslab.com/api/v6/images/fetch"
MODELSLAB_CONTROLNET_URL = "https://modelslab.com/api/v5/controlnet"

REFINE_STRENGTH = 0.35   # low = preserve composition, high = more creative

# Model aliases — shorthand → real ModelsLab model_id
MODEL_ALIASES = {
    "pony":        "pony-diffusion-v6-xl",
    "counterfeit": "counterfeit-v3-0-v3-0",
    "aom3":        "anything-v5",       # aom3 not confirmed, fallback to anything-v5
    "anything":    "anything-v5",
    "dreamshaper": "dreamshaper-8",
}

# Pony Diffusion needs score tags + Danbooru tag format
PONY_QUALITY_TAGS = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
PONY_NEGATIVE_TAGS = "score_1, score_2, score_3, score_4, worst quality, low quality, bad anatomy, bad hands, missing fingers, extra fingers, deformed, ugly, blurry"

# Counterfeit V3 — NAI-style quality tags, emphasis syntax
COUNTERFEIT_QUALITY_TAGS = "(masterpiece:1.3), (best quality:1.2), (ultra-detailed:1.1), highres, absurdres"
COUNTERFEIT_NEGATIVE_TAGS = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
    "username, blurry, bad feet, deformed, ugly, extra limbs, fused fingers, too many fingers, "
    "long neck, mutation, poorly drawn face, poorly drawn hands, cloned face, malformed limbs, "
    "missing arms, missing legs, extra arms, extra legs, mutated hands, gross proportions"
)

# Fixed character tags for this character — appended automatically when using pony
CHAR_TAGS = (
    "1girl, solo, long hair, very long hair, blonde hair, wavy hair, light-colored hair, "
    "red eyes, crimson eyes, bandaid, bandaid on cheek, facial bandage, young woman, adult"
)

ANGLES = {
    "front":         "camera angle: straight-on front view, character facing directly toward the viewer",
    "back":          "camera angle: rear view, character facing away from the viewer",
    "side-left":     "camera angle: left profile view, character facing left",
    "side-right":    "camera angle: right profile view, character facing right",
    "three-quarter": "camera angle: three-quarter front view, character turned slightly to one side",
    "low":           "camera angle: low-angle shot, camera below looking up at the character",
    "high":          "camera angle: high-angle shot, camera above looking down at the character",
    "closeup":       "camera angle: upper-body close-up from waist to head",
}

REFINE_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry, "
    "deformed, ugly, extra limbs, fused fingers, too many fingers, long neck, mutation, "
    "brown square, brown pixel, missing bandage, no bandage, face scar, face mark, "
    "changed face, different face, altered face"
)

QUALITY_SUFFIX = (
    "Anime style, masterpiece, best quality, ultra-detailed, 8k. "
    "Anatomically correct hands, exactly five fingers per hand, natural finger proportions."
)

FRAMING = (
    "Tight framing: character fills 85% of the frame. "
    "Minimal background — simple, slightly blurred environment, character is the clear subject."
)

DEFAULT_PROMPT = (
    "same character, anime style, detailed face, soft lighting, high quality"
)
# =========================================================


def build_transfer_prompt(angle: str | None, has_pose: bool) -> str:
    angle_line = ANGLES.get(angle, "") if angle else ""

    if has_pose:
        pose_instruction = (
            "The LEFT PORTION of the reference image is a POSE REFERENCE — copy its body pose "
            "exactly: every limb angle, body stance, and framing. "
            "The RIGHT PORTION shows the CHARACTER to draw — use their face, hair, outfit, "
            "accessories, and color palette exactly. "
        )
    else:
        pose_instruction = (
            "This is a CHARACTER REFERENCE SHEET showing the same character from different angles. "
            "Draw this character in a natural standing pose, keeping all details identical: "
            "face, hair, outfit, accessories, and color palette. "
        )

    return (
        f"{pose_instruction}"
        "CRITICAL: The character is a YOUNG ADULT with fully adult body proportions and mature "
        "height — NOT a child, NOT chibi, NOT super-deformed. "
        f"{angle_line + '. ' if angle_line else ''}"
        f"{FRAMING} "
        f"{QUALITY_SUFFIX} "
        "Output a single standalone illustration of ONE character only."
    )


def stitch_images(paths: list[Path], target_height: int = 768) -> str:
    from PIL import Image

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        ratio = target_height / img.height
        img = img.resize((int(img.width * ratio), target_height), Image.LANCZOS)
        imgs.append(img)

    total_width = sum(i.width for i in imgs)
    composite = Image.new("RGB", (total_width, target_height), (255, 255, 255))
    x = 0
    for img in imgs:
        composite.paste(img, (x, 0))
        x += img.width

    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def image_to_data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime = "webp" if ext == "webp" else ("png" if ext == "png" else "jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"



def _progress(elapsed: float, eta_total: float, label: str = "fal") -> None:
    pct = min(99, int(100 * elapsed / max(eta_total, 1)))
    bar_len = 24
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {int(elapsed)}s / ~{int(eta_total)}s  [{label}]")
    sys.stdout.flush()


# ── FAL ───────────────────────────────────────────────────

def fal_submit(image_url: str, prompt: str, num_images: int, seed: int | None) -> dict:
    body = {
        "prompt": prompt,
        "image_url": image_url,
        "num_images": num_images,
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "output_format": "png",
        "safety_tolerance": "6",
    }
    if seed is not None:
        body["seed"] = seed
    r = requests.post(FAL_SUBMIT_URL, headers=FAL_HEADERS, json=body, timeout=120)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"fal submit failed ({r.status_code}): {r.text[:400]}")
    data = r.json()
    if not data.get("request_id"):
        raise RuntimeError(f"no request_id in fal response: {data}")
    return data


def fal_poll(submit_response: dict) -> dict:
    status_url = (
        submit_response.get("status_url")
        or f"{FAL_SUBMIT_URL}/requests/{submit_response['request_id']}/status"
    )
    result_url = (
        submit_response.get("response_url")
        or f"{FAL_SUBMIT_URL}/requests/{submit_response['request_id']}"
    )
    start = time.time()
    eta_total = 30.0
    deadline = start + 600
    last_status = None

    while time.time() < deadline:
        elapsed = time.time() - start
        if elapsed > eta_total:
            eta_total = elapsed * 1.4
        _progress(elapsed, eta_total, "FLUX")

        try:
            r = requests.get(status_url, headers=FAL_HEADERS, timeout=30)
        except Exception as e:
            sys.stdout.write(f"\n  network error: {e}, retrying ...\n")
            time.sleep(3)
            continue

        if r.status_code != 200:
            sys.stdout.write(f"\n  status HTTP {r.status_code}: {r.text[:200]}\n")
            time.sleep(3)
            continue

        data = r.json()
        status = data.get("status")
        if status != last_status:
            sys.stdout.write(f"\n  fal: {status}\n")
            last_status = status

        if status == "COMPLETED":
            sys.stdout.write(f"  [{'█' * 24}] 100%  done in {int(elapsed)}s\n")
            rr = requests.get(result_url, headers=FAL_HEADERS, timeout=60)
            if not rr.ok:
                raise RuntimeError(f"fal result fetch failed ({rr.status_code}): {rr.text[:600]}")
            return rr.json()
        if status in ("IN_PROGRESS", "IN_QUEUE"):
            time.sleep(3)
            continue
        sys.stdout.write("\n")
        raise RuntimeError(f"unexpected fal status {status}: {data}")

    sys.stdout.write("\n")
    raise RuntimeError(f"fal timed out after {int(time.time() - start)}s")


# ── MODELSLAB ─────────────────────────────────────────────

def modelslab_refine(fal_image_url: str, prompt: str, model: str = MODELSLAB_MODEL, lora: str | None = None, lora_strength: float = 0.8) -> bytes:
    """Run img2img through ModelsLab anime model. Returns raw PNG bytes."""
    real_model = MODEL_ALIASES.get(model, model)
    is_pony = "pony" in real_model.lower()
    is_counterfeit = "counterfeit" in real_model.lower()

    if is_pony:
        refine_prompt = f"{PONY_QUALITY_TAGS}, {CHAR_TAGS}, sunglasses, indoors, cozy room, laptop, desk lamp, warm lighting, painterly, detailed shading, subsurface scattering, soft shadows"
        negative = PONY_NEGATIVE_TAGS
        strength = 0.52
        steps = "35"
        guidance = 6.5
        scheduler = "UniPCMultistepScheduler"
    elif is_counterfeit:
        refine_prompt = (
            f"{COUNTERFEIT_QUALITY_TAGS}, {CHAR_TAGS}, "
            "1girl, solo, anime style, (beautiful detailed eyes:1.2), (crimson red eyes:1.3), "
            "(white bandage on cheek:1.2), (band-aid on cheek:1.2), "
            "soft cel shading, dynamic lighting, sharp focus, detailed face, "
            "perfect anatomy, slender figure"
        )
        negative = COUNTERFEIT_NEGATIVE_TAGS
        strength = REFINE_STRENGTH
        steps = "30"
        guidance = 7.5
        scheduler = "DPMSolverMultistepScheduler"
    else:
        refine_prompt = (
            f"{prompt[:300]} "
            "1girl, anime style, masterpiece, best quality, ultra-detailed, "
            "beautiful detailed eyes, crimson red eyes, white bandage on cheek, band-aid on cheek, "
            "soft cel shading, dynamic lighting, sharp focus, preserve face details"
        )
        negative = REFINE_NEGATIVE
        strength = REFINE_STRENGTH
        steps = "40"
        guidance = 7
        scheduler = "UniPCMultistepScheduler"

    payload = {
        "key": MODELSLAB_KEY,
        "model_id": real_model,
        "prompt": refine_prompt,
        "negative_prompt": negative,
        "init_image": fal_image_url,
        "strength": strength,
        "width": "1024",
        "height": "1024",
        "samples": "1",
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "scheduler": scheduler,
        "safety_checker": "no",
        "enhance_prompt": "no",
        "base64": "no",
    }
    if lora:
        payload["lora_model"] = lora
        payload["lora_strength"] = lora_strength

    print("  refining with ModelsLab anime model ...")
    r = requests.post(MODELSLAB_URL, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(f"ModelsLab request failed ({r.status_code}): {r.text[:400]}")

    data = r.json()
    status = data.get("status")

    if status == "error":
        raise RuntimeError(f"ModelsLab error: {data.get('message', data)}")

    if status == "processing":
        job_id = data.get("id")
        eta = data.get("eta", 20)
        print(f"  ModelsLab queued (id={job_id}, eta~{eta}s) ...")
        start = time.time()
        while True:
            time.sleep(5)
            elapsed = int(time.time() - start)
            _progress(elapsed, max(eta, elapsed + 5), "ModelsLab")
            fr = requests.post(
                f"{MODELSLAB_FETCH_URL}/{job_id}",
                json={"key": MODELSLAB_KEY},
                timeout=30,
            )
            if not fr.ok:
                continue
            fd = fr.json()
            if fd.get("status") == "success":
                sys.stdout.write(f"\n  ModelsLab done in {elapsed}s\n")
                data = fd
                break
            if fd.get("status") == "error":
                raise RuntimeError(f"ModelsLab fetch error: {fd.get('message', fd)}")

    output = data.get("output") or []
    if not output:
        raise RuntimeError(f"ModelsLab returned no output: {data}")

    img_url = output[0]
    resp = requests.get(img_url, timeout=60)
    resp.raise_for_status()
    return resp.content


def modelslab_ipadapter(
    init_image_uri: str,
    ip_image_uri: str,
    prompt: str,
    model: str = MODELSLAB_MODEL,
    ip_adapter_id: str = "ip-adapter-plus_sd15",
    ip_scale: float = 0.7,
    controlnet: str = "canny",
    controlnet_scale: float = 0.7,
    strength: float = 0.6,
    width: int = 768,
    height: int = 1024,
    seed: int | None = None,
) -> bytes:
    """IP-Adapter + ControlNet via ModelsLab /api/v5/controlnet.

    init_image_uri = structure source (canny/openpose target).
    ip_image_uri   = identity/style source ("from" image).
    """
    real_model = MODEL_ALIASES.get(model, model)
    is_pony = "pony" in real_model.lower()
    is_counterfeit = "counterfeit" in real_model.lower()

    if is_pony:
        full_prompt = f"{PONY_QUALITY_TAGS}, {CHAR_TAGS}, {prompt}"
        negative = PONY_NEGATIVE_TAGS
    elif is_counterfeit:
        full_prompt = f"{COUNTERFEIT_QUALITY_TAGS}, {prompt}"
        negative = COUNTERFEIT_NEGATIVE_TAGS
    else:
        full_prompt = f"masterpiece, best quality, ultra-detailed, {prompt}"
        negative = REFINE_NEGATIVE

    payload = {
        "key": MODELSLAB_KEY,
        "model_id": real_model,
        "prompt": full_prompt,
        "negative_prompt": negative,
        "init_image": init_image_uri,
        "ip_adapter_id": ip_adapter_id,
        "ip_adapter_image": ip_image_uri,
        "ip_adapter_scale": ip_scale,
        "controlnet_model": controlnet,
        "controlnet_conditioning_scale": controlnet_scale,
        "strength": strength,
        "width": str(width),
        "height": str(height),
        "samples": "1",
        "num_inference_steps": "30",
        "guidance_scale": 7.5,
        "scheduler": "DPMSolverMultistepScheduler",
        "safety_checker": "no",
        "enhance_prompt": "no",
        "base64": "no",
    }
    if seed is not None:
        payload["seed"] = seed

    print(f"  IP-Adapter: id={ip_adapter_id}, scale={ip_scale} | ControlNet: {controlnet} ({controlnet_scale})")
    print(f"  posting to {MODELSLAB_CONTROLNET_URL} ...")
    r = requests.post(MODELSLAB_CONTROLNET_URL, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(f"ModelsLab controlnet failed ({r.status_code}): {r.text[:400]}")

    data = r.json()
    status = data.get("status")
    if status == "error":
        raise RuntimeError(f"ModelsLab error: {data.get('message', data)}")

    if status == "processing":
        job_id = data.get("id")
        eta = data.get("eta", 25)
        print(f"  ModelsLab queued (id={job_id}, eta~{eta}s) ...")
        start = time.time()
        while True:
            time.sleep(5)
            elapsed = int(time.time() - start)
            _progress(elapsed, max(eta, elapsed + 5), "IP-Adapter")
            fr = requests.post(
                f"{MODELSLAB_FETCH_URL}/{job_id}",
                json={"key": MODELSLAB_KEY},
                timeout=30,
            )
            if not fr.ok:
                continue
            fd = fr.json()
            if fd.get("status") == "success":
                sys.stdout.write(f"\n  ModelsLab done in {elapsed}s\n")
                data = fd
                break
            if fd.get("status") == "error":
                raise RuntimeError(f"ModelsLab fetch error: {fd.get('message', fd)}")

    output = data.get("output") or []
    if not output:
        raise RuntimeError(f"ModelsLab returned no output: {data}")

    img_url = output[0]
    resp = requests.get(img_url, timeout=60)
    resp.raise_for_status()
    return resp.content


# ── MAIN ──────────────────────────────────────────────────

def resolve(p: str) -> Path:
    src = Path(p)
    if not src.is_absolute():
        src = (Path(__file__).parent / src).resolve()
    return src


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-step anime pipeline: FLUX Kontext → ModelsLab anime refinement."
    )
    parser.add_argument("image", nargs="?", help="Single-image mode: image to transform")
    parser.add_argument("--character", nargs="+", help="Character reference image(s)")
    parser.add_argument("--pose", help="Pose reference image (stitched left of character refs)")
    parser.add_argument(
        "--angle",
        choices=list(ANGLES.keys()),
        default=None,
        help="Camera angle preset",
    )
    parser.add_argument("--prompt", default=None, help="Override the full prompt")
    parser.add_argument("--samples", type=int, default=1, help="Number of outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--refine", action="store_true", help="Enable ModelsLab anime refinement (experimental)")
    parser.add_argument("--model", default=MODELSLAB_MODEL, help="ModelsLab model ID or alias: counterfeit, anything, pony, dreamshaper (default: counterfeit-v3-0-v3-0)")
    parser.add_argument("--lora", default=None, help="ModelsLab LoRA model ID to apply during refinement")
    parser.add_argument("--lora-strength", type=float, default=0.8, help="LoRA strength 0.0-1.0 (default: 0.8)")
    parser.add_argument("--ip-image", default=None, help="IP-Adapter reference image (identity/style source). Triggers IP-Adapter mode when used with positional IMAGE.")
    parser.add_argument("--ip-adapter", default="ip-adapter-plus_sd15", help="IP-Adapter model id (default: ip-adapter-plus_sd15; for faces use ip-adapter-plus-face_sd15)")
    parser.add_argument("--ip-scale", type=float, default=0.7, help="IP-Adapter scale 0.0-1.0 (default: 0.7)")
    parser.add_argument("--ip-strength", type=float, default=0.6, help="img2img strength in IP-Adapter mode (default: 0.6)")
    parser.add_argument("--controlnet", default="canny", help="ControlNet model: canny, openpose, depth, lineart, face_detector. Comma-separate for multi (default: canny)")
    parser.add_argument("--controlnet-scale", type=float, default=0.7, help="ControlNet conditioning scale 0.0-1.0 (default: 0.7)")
    parser.add_argument("--ip-width", type=int, default=768)
    parser.add_argument("--ip-height", type=int, default=1024)
    args = parser.parse_args()

    # ── IP-Adapter standalone mode: IMAGE + --ip-image, skips FLUX ──
    if args.image and args.ip_image and not args.character:
        src = resolve(args.image)
        ip_src = resolve(args.ip_image)
        for p in (src, ip_src):
            if not p.exists():
                print(f"file not found: {p}", file=sys.stderr)
                return 1
        print(f"IP-Adapter mode: init={src.name}, ip_ref={ip_src.name}")
        init_uri = image_to_data_uri(src)
        ip_uri = image_to_data_uri(ip_src)
        prompt = args.prompt or DEFAULT_PROMPT

        out_dir = Path(__file__).parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())

        try:
            final_bytes = modelslab_ipadapter(
                init_image_uri=init_uri,
                ip_image_uri=ip_uri,
                prompt=prompt,
                model=args.model,
                ip_adapter_id=args.ip_adapter,
                ip_scale=args.ip_scale,
                controlnet=args.controlnet,
                controlnet_scale=args.controlnet_scale,
                strength=args.ip_strength,
                width=args.ip_width,
                height=args.ip_height,
                seed=args.seed,
            )
        except Exception as e:
            print(f"IP-Adapter generation failed: {e}", file=sys.stderr)
            return 1

        p = out_dir / f"ipadapter_{src.stem}_from_{ip_src.stem}_{stamp}.png"
        p.write_bytes(final_bytes)
        print(f"  saved: {p.name}")
        return 0

    if args.character:
        char_paths = [resolve(c) for c in args.character]
        for p in char_paths:
            if not p.exists():
                print(f"file not found: {p}", file=sys.stderr)
                return 1

        has_pose = bool(args.pose)
        if has_pose:
            pose_path = resolve(args.pose)
            if not pose_path.exists():
                print(f"file not found: {pose_path}", file=sys.stderr)
                return 1
            stitch_paths = [pose_path] + char_paths
            mode_label = f"pose={pose_path.name}, refs={[p.name for p in char_paths]}"
        else:
            stitch_paths = char_paths
            mode_label = f"refs={[p.name for p in char_paths]}"

        angle_label = f", angle={args.angle}" if args.angle else ""
        print(f"{'transfer' if has_pose else 'free-angle'} mode: {mode_label}{angle_label}")
        print("  stitching reference images ...")
        image_url = stitch_images(stitch_paths)
        prompt = args.prompt or build_transfer_prompt(args.angle, has_pose)
        stem = f"{'posed' if has_pose else 'freeangle'}_{char_paths[0].stem}"
        if args.angle:
            stem += f"_{args.angle}"

    elif args.image:
        src = resolve(args.image)
        if not src.exists():
            print(f"file not found: {src}", file=sys.stderr)
            return 1
        print(f"single-image mode: {src.name}")
        image_url = image_to_data_uri(src)
        prompt = args.prompt or DEFAULT_PROMPT
        stem = src.stem
    else:
        parser.error("Provide either IMAGE or --character (with optional --pose and --angle)")

    print(f"  prompt: {prompt[:140]}{'...' if len(prompt) > 140 else ''}")
    print(f"[1/2] submitting to FLUX Kontext ...")
    sub = fal_submit(image_url, prompt, args.samples, args.seed)
    print(f"  request_id: {sub['request_id']}")

    result = fal_poll(sub)
    images = result.get("images") or []
    if not images:
        print(f"no images in fal result: {result}", file=sys.stderr)
        return 1

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())

    for i, img_data in enumerate(images, start=1):
        fal_url = img_data.get("url") if isinstance(img_data, dict) else img_data
        if not fal_url:
            continue

        if not args.refine:
            resp = requests.get(fal_url, timeout=60)
            resp.raise_for_status()
            final_bytes = resp.content
            suffix = ""
        else:
            print(f"\n[2/2] anime refinement (ModelsLab {args.model}) ...")
            try:
                final_bytes = modelslab_refine(fal_url, prompt, args.model, args.lora, args.lora_strength)
                suffix = "_anime"
            except Exception as e:
                print(f"  ModelsLab refinement failed: {e} — saving raw FLUX output", file=sys.stderr)
                resp = requests.get(fal_url, timeout=60)
                resp.raise_for_status()
                final_bytes = resp.content
                suffix = "_flux"

        p = out_dir / f"fal_{stem}_{stamp}_{i}{suffix}.png"
        p.write_bytes(final_bytes)
        print(f"  saved: {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
