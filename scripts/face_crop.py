"""Crop a generous face region (full-res) from a photo, for donor retouching.

For bodyshots / wide frames the face is small, so a whole-frame Gemini pass
barely touches it. Crop the face out at full resolution first, retouch THAT,
then surgical_paste registers it back (landmark affine handles the scale).

    .venv312/bin/python scripts/face_crop.py inputs/bodyshot6.jpg inputs/face6.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from retoucher import faceparse, image_io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--pad", type=float, default=0.6, help="padding as fraction of face size")
    args = ap.parse_args()

    rgb = image_io.load(Path(args.image)).pixels
    u8 = image_io.to_uint8(rgb)
    H, W = u8.shape[:2]
    g = faceparse.detect(rgb)
    if g is None:
        raise SystemExit("no face detected")
    ys, xs = np.where(g.face_oval > 0.5)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pw, ph = int(args.pad * (x1 - x0)), int(args.pad * (y1 - y0))
    crop = u8[max(0, y0 - ph):min(H, y1 + ph), max(0, x0 - pw):min(W, x1 + pw)]
    Image.fromarray(crop).save(args.out, quality=98)
    print(f"face crop {crop.shape[1]}x{crop.shape[0]} -> {args.out}")


if __name__ == "__main__":
    main()
