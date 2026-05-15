#!/usr/bin/env python3
"""Anime image generation via ModelsLab API (primary) + HuggingFace Spaces (fallback).

Usage (CLI):
    python3 test_image_gen.py "1girl, long hair, blue sky"
    python3 test_image_gen.py "1girl" --negative "ugly, blurry"
    python3 test_image_gen.py "1girl" --lora yae-miko-genshin
    python3 test_image_gen.py "1girl" --model pony
    python3 test_image_gen.py --serve                          # start local API server

API server (port 8899):
    POST /generate       {"prompt": "...", "negative_prompt": "...", ...}
    POST /generate/raw   {"prompt": "..."}   -> raw image bytes
    GET  /health
    Docs: http://localhost:8899/docs
"""
import os
import sys
import time
import base64
import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

MODELSLAB_KEY = os.environ.get(
    "MODELSLAB_KEY",
    "pNaRZnXW6N7pzLOs3uu3MJdJD1jGt5597tOPwzlXBwhLpYtkmydJLDSZxamf",
)
MODELSLAB_URL = "https://modelslab.com/api/v6/images/text2img"

HF_TOKEN = os.environ.get("HF_TOKEN")
OUT_DIR = Path(__file__).resolve().parent / "test_generated"

DEFAULT_MODEL = "pony"
DEFAULT_LORA = "yae-miko-genshin"

QUALITY_PREFIX = (
    "score_9, score_8_up, score_7_up, "
    "source_anime, (anime screencap:1.3), flat color, cel shading, anime coloring, "
    "masterpiece, best quality, very aesthetic, absurdres, "
)
DEFAULT_NEGATIVE = (
    "(worst quality:2), (low quality:2), (normal quality:2), "
    "(jpeg artifacts), (blurry), (duplicate), (morbid), (mutilated), "
    "(out of frame), (extra limbs), (bad anatomy), (disfigured), (deformed), "
    "(cross-eye), (glitch), (oversaturated), (overexposed), (underexposed), "
    "(bad proportions), (bad hands), (bad feet), (cloned face), (long neck), "
    "(missing arms), (missing legs), (extra fingers), (fused fingers), "
    "(poorly drawn hands), (poorly drawn face), (mutation), (deformed eyes), "
    "watermark, text, logo, signature, grainy, tiling, censored, "
    "ugly, blurry eyes, noisy image, bad lighting, unnatural skin, asymmetry, "
    "source_cartoon, source_pony, source_furry, "
    "3d, realistic, photo, photorealistic, semi-realistic"
)

# HF Spaces fallback
SPACES = [
    {
        "id": "Sergidev/HD-Pony-Diffusion-v6",
        "api_name": "/generate_and_update_history",
        "type": "sergidev",
    },
    {
        "id": "artificialguybr/PonyDiffusion-XL-Free-DEMO",
        "api_name": "/run",
        "type": "full",
    },
    {
        "id": "K00B404/Pony_Diffusion_V6_XL",
        "api_name": "/predict",
        "type": "simple",
    },
]


# ── ModelsLab generation ──────────────────────────────────────────────────

def generate_modelslab(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    model_id: str = DEFAULT_MODEL,
    lora_model: str = DEFAULT_LORA,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    steps: int = 31,
    scheduler: str = "DPMSolverMultistepScheduler",
) -> Path:
    """Generate via ModelsLab API. Returns path to saved image."""
    import requests

    full_prompt = QUALITY_PREFIX + prompt

    payload = {
        "key": MODELSLAB_KEY,
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id,
        "lora_model": lora_model,
        "width": str(width),
        "height": str(height),
        "guidance_scale": str(guidance_scale),
        "num_inference_steps": str(steps),
        "scheduler": scheduler,
        "enhance_prompt": False,
    }

    print(f"    ModelsLab [{model_id}] + LoRA [{lora_model}]...", flush=True)
    resp = requests.post(
        MODELSLAB_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "error":
        raise RuntimeError(f"ModelsLab error: {data.get('message', data)}")

    # Handle async processing — poll until ready
    if data.get("status") == "processing":
        fetch_url = data.get("fetch_result")
        eta = data.get("eta", 30)
        print(f"    Processing (ETA ~{eta}s)...", flush=True)
        for _ in range(60):
            time.sleep(5)
            poll = requests.post(
                fetch_url,
                headers={"Content-Type": "application/json"},
                json={"key": MODELSLAB_KEY},
                timeout=30,
            ).json()
            if poll.get("status") == "success":
                data = poll
                break
            if poll.get("status") == "error":
                raise RuntimeError(f"ModelsLab error: {poll.get('message', poll)}")
        else:
            raise RuntimeError("ModelsLab timed out after 5 minutes")

    # Download image from URL
    img_url = None
    output = data.get("output")
    if isinstance(output, list) and output:
        img_url = output[0]
    elif isinstance(output, str):
        img_url = output

    if not img_url:
        raise RuntimeError(f"No image URL in response: {data}")

    print(f"    Downloading...", flush=True)
    img_resp = requests.get(img_url, timeout=60)
    img_resp.raise_for_status()

    # Save to temp file
    ext = ".png"
    if "webp" in img_resp.headers.get("content-type", ""):
        ext = ".webp"
    elif "jpeg" in img_resp.headers.get("content-type", ""):
        ext = ".jpg"

    tmp = OUT_DIR / f"_tmp{ext}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(img_resp.content)
    return tmp


# ── HF Spaces fallback ───────────────────────────────────────────────────

def _extract_path(result) -> Path:
    img = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(img, list) and img:
        img = img[0]
    if isinstance(img, dict):
        path = img.get("image") or img.get("path")
        if path:
            return Path(path)
    if isinstance(img, str):
        return Path(img)
    raise RuntimeError(f"Unexpected result format: {img}")


def generate_hf_spaces(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 4.5,
    steps: int = 28,
    sampler: str = "DPM++ 2M SDE Karras",
) -> Path:
    """Fallback: generate via free HuggingFace Spaces."""
    from gradio_client import Client

    full_prompt = QUALITY_PREFIX + prompt
    errors = []

    for space in SPACES:
        sid = space["id"]
        try:
            print(f"    Trying {sid}...", flush=True)
            client = Client(sid, token=HF_TOKEN)
            stype = space["type"]
            api = space["api_name"]

            if stype == "sergidev":
                result = client.predict(
                    full_prompt, negative_prompt, seed,
                    width, height, guidance_scale, steps, sampler,
                    f"{width} x {height}", False, 0.55, 1.5, "", 1,
                    api_name=api,
                )
            elif stype == "full":
                result = client.predict(
                    full_prompt, negative_prompt, seed,
                    width, height, guidance_scale, steps, sampler,
                    f"{width} x {height}", False, 0.55, 1.5,
                    api_name=api,
                )
            else:
                result = client.predict(full_prompt, api_name=api)

            return _extract_path(result)
        except Exception as e:
            short = str(e)[:120]
            print(f"    Failed: {short}")
            errors.append(f"{sid}: {short}")

    raise RuntimeError("All HF Spaces failed:\n  " + "\n  ".join(errors))


# ── Unified generation ────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    model_id: str = DEFAULT_MODEL,
    lora_model: str = DEFAULT_LORA,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    steps: int = 31,
) -> Path:
    """Try ModelsLab first, fall back to HF Spaces."""
    # Try ModelsLab
    if MODELSLAB_KEY:
        try:
            return generate_modelslab(
                prompt, negative_prompt, model_id, lora_model,
                width, height, guidance_scale, steps,
            )
        except Exception as e:
            print(f"    ModelsLab failed: {str(e)[:120]}")
            print(f"    Falling back to HF Spaces...")

    # Fallback to HF Spaces
    return generate_hf_spaces(prompt, negative_prompt, width=width, height=height)


def save_image(src_path: Path, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix or ".png"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"pony_{ts}{ext}"
    dest.write_bytes(src_path.read_bytes())
    # Clean up temp file
    if src_path.name.startswith("_tmp"):
        src_path.unlink(missing_ok=True)
    return dest


# ── FastAPI server ────────────────────────────────────────────────────────

def create_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel

    app = FastAPI(title="Pony Diffusion V6 XL API", version="2.0")

    class GenerateRequest(BaseModel):
        prompt: str
        negative_prompt: str = DEFAULT_NEGATIVE
        model_id: str = DEFAULT_MODEL
        lora_model: str = DEFAULT_LORA
        width: int = 1024
        height: int = 1024
        guidance_scale: float = 7.5
        steps: int = 31

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backend": "modelslab" if MODELSLAB_KEY else "hf_spaces",
            "model": DEFAULT_MODEL,
            "lora": DEFAULT_LORA,
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        try:
            start = time.time()
            src = generate_image(
                req.prompt, req.negative_prompt, req.model_id, req.lora_model,
                req.width, req.height, req.guidance_scale, req.steps,
            )
            elapsed = time.time() - start
            dest = save_image(src)
            img_bytes = dest.read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            mime = "image/webp" if dest.suffix == ".webp" else "image/png"
            return JSONResponse({
                "status": "ok",
                "image_base64": b64,
                "mime_type": mime,
                "file": str(dest),
                "size_kb": round(len(img_bytes) / 1024, 1),
                "elapsed_seconds": round(elapsed, 1),
            })
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    @app.post("/generate/raw")
    def generate_raw(req: GenerateRequest):
        try:
            src = generate_image(
                req.prompt, req.negative_prompt, req.model_id, req.lora_model,
                req.width, req.height, req.guidance_scale, req.steps,
            )
            dest = save_image(src)
            img_bytes = dest.read_bytes()
            mime = "image/webp" if dest.suffix == ".webp" else "image/png"
            return Response(content=img_bytes, media_type=mime)
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    return app


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv):
    """Parse CLI args: positional prompt + optional flags."""
    args = {
        "prompt": "",
        "negative": DEFAULT_NEGATIVE,
        "model": DEFAULT_MODEL,
        "lora": DEFAULT_LORA,
    }
    positional = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--serve":
            return "serve", args
        elif a == "--negative" and i + 1 < len(argv):
            i += 1
            args["negative"] = argv[i]
        elif a == "--model" and i + 1 < len(argv):
            i += 1
            args["model"] = argv[i]
        elif a == "--lora" and i + 1 < len(argv):
            i += 1
            args["lora"] = argv[i]
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    args["prompt"] = " ".join(positional)
    return "generate", args


def main():
    mode, args = _parse_args(sys.argv)

    if mode == "serve":
        from granian import Granian
        print(f"\n  Pony Diffusion API v2")
        print(f"  Backend:  {'ModelsLab' if MODELSLAB_KEY else 'HF Spaces'}")
        print(f"  Model:    {DEFAULT_MODEL}")
        print(f"  LoRA:     {DEFAULT_LORA}")
        print(f"  Docs:     http://localhost:8899/docs\n")
        Granian(
            "tests.test_image_gen:create_app",
            factory=True,
            address="0.0.0.0",
            port=8899,
            interface="asgi",
        ).serve()
        return

    prompt = args["prompt"]
    if not prompt:
        prompt = "1girl, long hair, blue sky, detailed face, detailed eyes"

    try:
        start = time.time()
        src = generate_image(
            prompt,
            negative_prompt=args["negative"],
            model_id=args["model"],
            lora_model=args["lora"],
        )
        elapsed = time.time() - start
        dest = save_image(src)
        size_kb = dest.stat().st_size / 1024
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Saved:   {dest} ({size_kb:.1f} KB)\n")
        if sys.platform == "darwin":
            os.system(f'open "{dest}"')
    except Exception as e:
        print(f"\n  Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
