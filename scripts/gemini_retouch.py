"""Retouch a headshot with Google Gemini's image model (auth key from a file).

Sends the photo + a constrained casting-retouch prompt and saves the returned
image. The auth key blocks models.list(), so we try known image models
newest-first and use the first that returns an image (logged, so it's never a
silent hardcode).

    .venv312/bin/python scripts/gemini_retouch.py inputs/headshot1.jpg out.png \
        --key ~/Desktop/gemini.txt --max-edge 1536
"""
from __future__ import annotations

import argparse
import io
import os

from google import genai
from google.genai import types
from PIL import Image

# Newest first; first that yields an image wins. Override with --model.
IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-exp-image-generation",
]

PROMPT = (
    "You are a professional photo retoucher. This is a real photograph of a man. "
    "Make a MINIMAL, strictly photorealistic skin retouch of THIS photograph for an "
    "acting headshot. Treat it as editing the existing photo, not generating a new one.\n"
    "DO (only these): remove temporary blemishes and spots anywhere on the skin, "
    "including the small raised pimple/blemish on his upper chest below the collarbone; "
    "gently reduce under-eye darkness and the crepey/scaly texture under the eyes and "
    "beside the nose; lightly even skin tone.\n"
    "ABSOLUTELY DO NOT: add any redness, blush, warmth, tan, or 'rouge' — if anything "
    "REDUCE redness and keep his skin tone exactly matched to the original; change his "
    "expression, smile, or where he is looking; alter face shape, bone structure, "
    "proportions, eyes, eye color, eyebrows, lips, hairline, hair, or stubble/beard; "
    "stylize, illustrate, smooth, or airbrush — it must keep real pores and look like an "
    "untouched photo; change lighting, background, framing, pose, or clothing.\n"
    "Output the SAME photograph, same expression and composition and aspect ratio, just "
    "with those few skin fixes — indistinguishable from the original to a casting "
    "director who knows his face."
)


def _first_image(resp):
    for cand in getattr(resp, "candidates", None) or []:
        for part in getattr(cand.content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


def retouch(image_path: str, out_path: str, key_path: str, max_edge: int, model: str | None,
            prompt: str | None = None):
    prompt = prompt or PROMPT
    key = open(os.path.expanduser(key_path)).read().strip()
    client = genai.Client(api_key=key)
    img = Image.open(image_path).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))

    models = [model] if model else IMAGE_MODELS
    configs = [
        types.GenerateContentConfig(response_modalities=["IMAGE"]),
        types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        None,
    ]
    last_err = None
    for m in models:
        for cfg in configs:
            try:
                resp = client.models.generate_content(model=m, contents=[prompt, img], config=cfg)
                data = _first_image(resp)
                if data:
                    out = Image.open(io.BytesIO(data)).convert("RGB")
                    out.save(out_path)
                    print(f"OK model={m} cfg={'none' if cfg is None else cfg.response_modalities} "
                          f"in={img.size} out={out.size} -> {out_path}")
                    return
                last_err = f"{m}: no image in response"
            except Exception as e:
                last_err = f"{m} ({'none' if cfg is None else cfg.response_modalities}): {type(e).__name__} {str(e)[:140]}"
        print("  tried", m, "->", last_err)
    raise SystemExit(f"No image produced. Last: {last_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--key", default="~/Desktop/gemini.txt")
    ap.add_argument("--max-edge", type=int, default=1536)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default=None, help="custom prompt (default: the headshot retouch prompt)")
    args = ap.parse_args()
    retouch(args.image, args.out, args.key, args.max_edge, args.model, args.prompt)
