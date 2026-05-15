#!/usr/bin/env python3
"""img2img Web UI — Flask backend. Run: python3 server.py [--port 7860]"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

import requests
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))
import img2img as _core
from img2img import (
    ANGLES, DEFAULT_PROMPT, FAL_KEY, MODEL_ALIASES,
    MODELSLAB_KEY, MODELSLAB_MODEL, MODELSLAB_URL, MODELSLAB_FETCH_URL,
    build_transfer_prompt, image_to_data_uri, stitch_images,
    REFINE_NEGATIVE, QUALITY_SUFFIX,
)

app = Flask(__name__)
OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="img2img_ui_"))

jobs: dict[str, dict] = {}

FAL_CATALOG = [
    {"id": "fal-ai/flux-pro/kontext",    "name": "FLUX Pro Kontext",      "tags": ["img2img", "character", "recommended"]},
    {"id": "fal-ai/flux-pro/v1.1-ultra", "name": "FLUX Pro v1.1 Ultra",   "tags": ["t2i", "premium", "high-res"]},
    {"id": "fal-ai/flux-pro/v1.1",       "name": "FLUX Pro v1.1",         "tags": ["t2i", "premium"]},
    {"id": "fal-ai/flux-pro",            "name": "FLUX Pro",              "tags": ["t2i"]},
    {"id": "fal-ai/flux/dev",            "name": "FLUX Dev",              "tags": ["t2i", "open"]},
    {"id": "fal-ai/flux/schnell",        "name": "FLUX Schnell",          "tags": ["t2i", "fast", "4-step"]},
    {"id": "fal-ai/flux-lora",           "name": "FLUX LoRA",             "tags": ["t2i", "lora"]},
    {"id": "fal-ai/flux-realism",        "name": "FLUX Realism",          "tags": ["t2i", "realism"]},
    {"id": "fal-ai/stable-diffusion-v3-medium", "name": "SD3 Medium",     "tags": ["t2i"]},
    {"id": "fal-ai/fast-sdxl",           "name": "Fast SDXL",             "tags": ["t2i", "fast"]},
    {"id": "fal-ai/hyper-sdxl",          "name": "Hyper SDXL",            "tags": ["t2i", "fast", "4-step"]},
    {"id": "fal-ai/stable-cascade",      "name": "Stable Cascade",        "tags": ["t2i"]},
    {"id": "fal-ai/pixart-sigma",        "name": "PixArt Sigma",          "tags": ["t2i"]},
    {"id": "fal-ai/aura-flow",           "name": "Aura Flow",             "tags": ["t2i"]},
    {"id": "fal-ai/kolors",              "name": "Kolors",                "tags": ["t2i", "colorful"]},
    {"id": "fal-ai/recraft-v3",          "name": "Recraft v3",            "tags": ["t2i", "design"]},
    {"id": "fal-ai/ideogram/v2",         "name": "Ideogram v2",           "tags": ["t2i", "text-in-image"]},
    {"id": "fal-ai/ideogram/v2/turbo",   "name": "Ideogram v2 Turbo",     "tags": ["t2i", "fast"]},
    {"id": "fal-ai/imagen4/preview",     "name": "Imagen 4 Preview",      "tags": ["t2i", "google"]},
    {"id": "fal-ai/ip-adapter-face-id",  "name": "IP-Adapter FaceID",     "tags": ["img2img", "face"]},
    {"id": "fal-ai/controlnet-sdxl",     "name": "ControlNet SDXL",       "tags": ["img2img", "control"]},
    {"id": "fal-ai/lcm",                 "name": "LCM",                   "tags": ["t2i", "fast"]},
    {"id": "fal-ai/sana",                "name": "Sana",                  "tags": ["t2i"]},
    {"id": "fal-ai/omnigen-v1",          "name": "OmniGen v1",            "tags": ["t2i", "multimodal"]},
    {"id": "fal-ai/lumina-next-t2i",     "name": "Lumina Next",           "tags": ["t2i"]},
    {"id": "fal-ai/stable-diffusion-xl", "name": "SDXL",                  "tags": ["t2i"]},
]


# ─── FAL wrapper (model-agnostic) ────────────────────────────
def fal_generate(model_id: str, image_url: str | None, prompt: str,
                 num_images: int, seed: int | None,
                 guidance: float, steps: int, log_fn=None) -> dict:
    url = f"https://queue.fal.run/{model_id}"
    headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}
    body = {
        "prompt": prompt,
        "num_images": num_images,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "output_format": "png",
        "safety_tolerance": "6",
    }
    if image_url:
        body["image_url"] = image_url
    if seed is not None:
        body["seed"] = seed

    if log_fn:
        log_fn(f"Submitting to {model_id}…")
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code not in (200, 202):
        raise RuntimeError(f"FAL submit failed ({r.status_code}): {r.text[:400]}")
    data = r.json()
    if not data.get("request_id"):
        raise RuntimeError(f"No request_id in FAL response: {data}")

    status_url = data.get("status_url") or f"{url}/requests/{data['request_id']}/status"
    result_url = data.get("response_url") or f"{url}/requests/{data['request_id']}"

    start = time.time()
    eta = 30.0
    deadline = start + 600
    last_status = None

    while time.time() < deadline:
        elapsed = time.time() - start
        if elapsed > eta:
            eta = elapsed * 1.4
        pct = min(99, int(100 * elapsed / max(eta, 1)))
        if log_fn:
            log_fn(f"progress:{pct}")

        try:
            r = requests.get(status_url, headers=headers, timeout=30)
        except Exception as e:
            if log_fn:
                log_fn(f"Network error: {e}, retrying…")
            time.sleep(3)
            continue

        if r.status_code != 200:
            time.sleep(3)
            continue

        d = r.json()
        status = d.get("status")
        if status != last_status and log_fn:
            log_fn(f"FAL: {status}")
            last_status = status

        if status == "COMPLETED":
            if log_fn:
                log_fn(f"FAL done in {int(elapsed)}s")
            rr = requests.get(result_url, headers=headers, timeout=60)
            if not rr.ok:
                raise RuntimeError(f"FAL result fetch failed: {rr.text[:400]}")
            return rr.json()
        if status in ("IN_PROGRESS", "IN_QUEUE"):
            time.sleep(3)
            continue
        raise RuntimeError(f"Unexpected FAL status: {status}")

    raise RuntimeError("FAL timed out")


def modelslab_refine_custom(fal_url: str, prompt: str, model: str,
                             lora: str | None, lora_strength: float,
                             strength: float, steps: int, guidance: float,
                             scheduler: str, log_fn=None) -> bytes:
    real_model = MODEL_ALIASES.get(model, model)
    is_pony = "pony" in real_model.lower()

    if is_pony:
        refine_prompt = (
            "score_9, score_8_up, score_7_up, 1girl, solo, long hair, blonde hair, "
            "red eyes, bandaid on cheek, sunglasses, indoors, cozy room, masterpiece"
        )
        negative = _core.PONY_NEGATIVE_TAGS
    else:
        refine_prompt = (
            f"{prompt[:300]} 1girl, anime style, masterpiece, best quality, ultra-detailed, "
            "beautiful detailed eyes, crimson red eyes, white bandage on cheek, "
            "soft cel shading, dynamic lighting, sharp focus"
        )
        negative = REFINE_NEGATIVE

    payload = {
        "key": MODELSLAB_KEY,
        "model_id": real_model,
        "prompt": refine_prompt,
        "negative_prompt": negative,
        "init_image": fal_url,
        "strength": strength,
        "width": "1024",
        "height": "1024",
        "samples": "1",
        "num_inference_steps": str(steps),
        "guidance_scale": guidance,
        "scheduler": scheduler,
        "safety_checker": "no",
        "enhance_prompt": "no",
        "base64": "no",
    }
    if lora:
        payload["lora_model"] = lora
        payload["lora_strength"] = lora_strength

    if log_fn:
        log_fn(f"Refining with ModelsLab {real_model}…")
    r = requests.post(MODELSLAB_URL, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(f"ModelsLab failed ({r.status_code}): {r.text[:400]}")

    data = r.json()
    status = data.get("status")

    if status == "error":
        raise RuntimeError(f"ModelsLab error: {data.get('message', data)}")

    if status == "processing":
        job_id = data.get("id")
        eta = data.get("eta", 20)
        if log_fn:
            log_fn(f"ModelsLab queued (id={job_id}, eta~{eta}s)…")
        start = time.time()
        while True:
            time.sleep(5)
            elapsed = int(time.time() - start)
            pct = min(99, int(50 + 50 * elapsed / max(eta * 2, elapsed + 1)))
            if log_fn:
                log_fn(f"progress:{pct}")
            fr = requests.post(
                f"{MODELSLAB_FETCH_URL}/{job_id}",
                json={"key": MODELSLAB_KEY},
                timeout=30,
            )
            if not fr.ok:
                continue
            fd = fr.json()
            if fd.get("status") == "success":
                if log_fn:
                    log_fn(f"ModelsLab done in {elapsed}s")
                data = fd
                break
            if fd.get("status") == "error":
                raise RuntimeError(f"ModelsLab fetch error: {fd.get('message', fd)}")

    output = data.get("output") or []
    if not output:
        raise RuntimeError(f"ModelsLab returned no output: {data}")

    resp = requests.get(output[0], timeout=60)
    resp.raise_for_status()
    return resp.content


# ─── Job runner ──────────────────────────────────────────────
def run_job(job_id: str, config: dict, image_paths: dict):
    q = jobs[job_id]["queue"]

    def log(msg: str):
        q.put({"type": "log", "text": msg})

    try:
        mode        = config.get("mode", "single")
        prompt      = config.get("prompt", "") or ""
        fal_model   = config.get("fal_model", "fal-ai/flux-pro/kontext").strip()
        guidance    = float(config.get("guidance_scale", 4.5))
        steps       = int(config.get("steps", 50))
        num_images  = int(config.get("num_images", 1))
        seed        = config.get("seed")
        seed        = int(seed) if seed not in (None, "", "null") else None
        angle       = config.get("angle") or None
        do_refine   = config.get("refine", False)
        ml_model    = config.get("ml_model", MODELSLAB_MODEL) or MODELSLAB_MODEL
        r_strength  = float(config.get("refine_strength", 0.2))
        r_steps     = int(config.get("refine_steps", 40))
        r_guidance  = float(config.get("refine_guidance", 7.0))
        scheduler   = config.get("scheduler", "UniPCMultistepScheduler")
        lora        = config.get("lora") or None
        lora_str    = float(config.get("lora_strength", 0.8))

        # Build input image URL
        if mode == "single":
            src = image_paths.get("single")
            if not src:
                raise RuntimeError("No image uploaded for single-image mode")
            image_url = image_to_data_uri(Path(src))
            if not prompt:
                prompt = DEFAULT_PROMPT
            stem = "single"

        elif mode == "character":
            char_paths = [Path(p) for p in (image_paths.get("characters") or [])]
            if not char_paths:
                raise RuntimeError("No character images uploaded")
            log(f"Stitching {len(char_paths)} reference image(s)…")
            image_url = stitch_images(char_paths)
            if not prompt:
                prompt = build_transfer_prompt(angle, False)
            stem = "freeangle"

        elif mode == "pose":
            char_paths = [Path(p) for p in (image_paths.get("characters") or [])]
            pose_p     = image_paths.get("pose")
            if not char_paths:
                raise RuntimeError("No character images uploaded")
            if not pose_p:
                raise RuntimeError("No pose reference uploaded")
            log(f"Stitching pose + {len(char_paths)} character image(s)…")
            image_url = stitch_images([Path(pose_p)] + char_paths)
            if not prompt:
                prompt = build_transfer_prompt(angle, True)
            stem = "posed"
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        log(f"Prompt: {prompt[:120]}{'…' if len(prompt) > 120 else ''}")
        q.put({"type": "phase", "phase": "fal"})

        result = fal_generate(fal_model, image_url, prompt, num_images, seed,
                              guidance, steps, log_fn=log)
        images_out = result.get("images") or []
        if not images_out:
            raise RuntimeError(f"No images in FAL result: {result}")

        stamp = int(time.time())
        saved = []

        for i, img_data in enumerate(images_out, start=1):
            fal_url = img_data.get("url") if isinstance(img_data, dict) else img_data
            if not fal_url:
                continue

            if not do_refine:
                resp = requests.get(fal_url, timeout=60)
                resp.raise_for_status()
                final_bytes = resp.content
                suffix = ""
            else:
                q.put({"type": "phase", "phase": "refine"})
                try:
                    final_bytes = modelslab_refine_custom(
                        fal_url, prompt, ml_model, lora, lora_str,
                        r_strength, r_steps, r_guidance, scheduler, log_fn=log,
                    )
                    suffix = "_refined"
                except Exception as e:
                    log(f"Refinement failed: {e} — saving raw FLUX output")
                    resp = requests.get(fal_url, timeout=60)
                    resp.raise_for_status()
                    final_bytes = resp.content
                    suffix = "_flux"

            fname = f"ui_{stem}_{stamp}_{i}{suffix}.png"
            (OUT_DIR / fname).write_bytes(final_bytes)
            saved.append(fname)
            log(f"Saved: {fname}")
            q.put({"type": "image", "filename": fname})

        q.put({"type": "done", "images": saved})
        jobs[job_id]["status"] = "done"
        jobs[job_id]["images"] = saved

    except Exception as e:
        log(f"Error: {e}")
        q.put({"type": "error", "message": str(e)})
        jobs[job_id]["status"] = "error"
    finally:
        q.put(None)


# ─── Routes ──────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(str(Path(__file__).parent / "ui.html"))


@app.route("/out/<path:filename>")
def serve_image(filename):
    return send_from_directory(str(OUT_DIR), filename)


@app.route("/api/preview")
def preview():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists() or not str(p).startswith(str(UPLOAD_DIR)):
        return "not found", 404
    return send_file(str(p))


@app.route("/api/upload", methods=["POST"])
def upload():
    saved = {}
    for field in ("image", "pose"):
        if field in request.files:
            f = request.files[field]
            if f.filename:
                ext = Path(f.filename).suffix or ".png"
                tmp = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
                f.save(str(tmp))
                saved[field] = str(tmp)
    chars = []
    for f in request.files.getlist("characters"):
        if f.filename:
            ext = Path(f.filename).suffix or ".png"
            tmp = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
            f.save(str(tmp))
            chars.append(str(tmp))
    if chars:
        saved["characters"] = chars
    return jsonify(saved)


@app.route("/api/search/fal")
def search_fal():
    q = request.args.get("q", "").lower()
    results = (
        [m for m in FAL_CATALOG
         if q in m["id"].lower() or q in m["name"].lower()
         or any(q in t for t in m.get("tags", []))]
        if q else FAL_CATALOG
    )
    return jsonify(results)


@app.route("/api/search/modelslab")
def search_modelslab():
    q = request.args.get("q", "").lower()
    try:
        r = requests.post(
            "https://modelslab.com/api/v6/images/models",
            json={"key": MODELSLAB_KEY},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            models = data if isinstance(data, list) else data.get("data", [])
            if q:
                models = [m for m in models if q in str(m).lower()]
            return jsonify({"models": models[:60]})
    except Exception:
        pass
    return jsonify({"models": [], "error": "ModelsLab API unavailable"})


@app.route("/api/search/lora")
def search_lora():
    q = request.args.get("q", "").lower()
    try:
        r = requests.post(
            "https://modelslab.com/api/v6/images/loras",
            json={"key": MODELSLAB_KEY},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            loras = data if isinstance(data, list) else data.get("data", [])
            if q:
                loras = [l for l in loras if q in str(l).lower()]
            return jsonify({"loras": loras[:60]})
    except Exception:
        pass
    return jsonify({"loras": [], "error": "ModelsLab API unavailable"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    config      = data.get("config", {})
    image_paths = data.get("images", {})
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "images": [], "queue": Queue()}
    threading.Thread(
        target=run_job, args=(job_id, config, image_paths), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>/stream")
def stream_job(job_id):
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404

    def event_stream():
        q = jobs[job_id]["queue"]
        while True:
            try:
                msg = q.get(timeout=60)
                if msg is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except Empty:
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/images")
def list_images():
    imgs = sorted(OUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([p.name for p in imgs[:100]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"img2img Studio → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
