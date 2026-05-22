"""Image generation via Cloudflare Workers AI proxy."""
from __future__ import annotations

import datetime
import json as _json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def tool_image(prompt):
    """Generate an image. Saves to yuki/images/img_<timestamp>.png."""
    cf_url = os.environ.get(
        "CF_IMAGE_URL",
        "https://image-gen-worker.mohammedlaminemennane.workers.dev",
    )
    cf_token = os.environ.get("CF_IMAGE_TOKEN", "llama-img-2026-xyz")

    try:
        req = Request(
            cf_url,
            data=_json.dumps({"prompt": prompt}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Auth-Token": cf_token,
                "User-Agent": "llama-voice-assist/1.0",
            },
            method="POST",
        )
        with urlopen(req, timeout=300) as resp:
            img_data = resp.read()

        # yuki/images/ — repo root / yuki / images
        out_dir = Path(__file__).parent.parent.parent / "yuki" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{ts}.png"
        out_path = out_dir / filename
        out_path.write_bytes(img_data)
        return (
            f"SUCCESS! You created an image and saved it to yuki/images/{filename}. "
            f"The scene you drew: \"{prompt}\". Now tell the user what you drew — "
            f"describe the scene in your own excited words. Do NOT mention file paths "
            f"or bytes. Just say what the image shows!"
        )
    except Exception as e:
        return f"Image generation error: {e}"
