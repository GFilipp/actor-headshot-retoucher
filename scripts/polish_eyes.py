"""Layer 2: deterministic 'Photoshop' cleanup on top of the Gemini surgical paste.

Run AFTER surgical_paste. Cleans the residual the paste leaves around the eyes:
  1. eye-white correction (de-red / de-yellow / brighten the sclera)
  2. discoloration ring — even the redness/brownness skin around the orbit (a*/b*
     only; L kept, so form/texture survive)
  3. residual fine lines under the eye (high-frequency attenuation, feature-safe)

    .venv312/bin/python scripts/polish_eyes.py inputs/photo1_final.jpg \
        inputs/photo1_polished.jpg --whites 0.6 --discolor 0.6 --lines 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from retoucher import faceparse, image_io
from retoucher.blend import (
    reduce_discoloration, smooth_under_eye_texture, whiten_eye_whites,
)
from retoucher.image_io import to_uint8
from retoucher.mask import skin_mask


def _dilate(m, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px) * 2 + 1, int(px) * 2 + 1))
    return cv2.dilate(m.astype(np.float32), k)


def _feather(m, s):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=max(1.0, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--whites", type=float, default=0.6, help="eye-white correction 0..1")
    ap.add_argument("--discolor", type=float, default=0.6, help="periorbital discoloration evening 0..1")
    ap.add_argument("--lines", type=float, default=0.4, help="residual fine-line smoothing 0..1")
    ap.add_argument("--shadow", type=float, default=0.5, help="under-eye shadow/bag-darkness lift 0..1")
    args = ap.parse_args()

    rgb = image_io.load(Path(args.image)).pixels.astype(np.float32)
    H, W = rgb.shape[:2]
    spatial = max(H, W) / 1024.0
    sigma = 8.0 * spatial
    g = faceparse.detect(rgb)
    if g is None:
        raise SystemExit("no face geometry")
    skin = skin_mask(rgb)

    ys = np.where(g.eyes.max(axis=1) > 0.5)[0]
    eye_h = float(ys.max() - ys.min()) if ys.size else 0.08 * H
    # ROUNDED eye-area disc (under-eye, corners, crow's feet); no straight edges = no box.
    area = _dilate(g.eyes, 1.0 * eye_h) * g.face_oval
    protect_wide = _dilate(g.protect, max(2.0, 4.0 * spatial))    # SMOOTHING: keep off lashes
    protect_tight = _dilate(g.protect, max(1.0, 1.0 * spatial))   # DISCOLOR: reach the eye-opening rim
    ref = np.clip(g.face_oval - area - protect_wide, 0.0, 1.0)    # clean cheek/forehead skin to pull toward

    res = rgb
    res = whiten_eye_whites(res, g.eyes, args.whites)
    # de-red / de-brown the rim around the eye opening, toward clean skin (reaches the rim)
    disc = _feather(area * (1.0 - _feather(protect_tight, sigma * 0.3)), sigma * 0.6)
    res = reduce_discoloration(res, disc, args.discolor, ref)
    # fine-line smoothing, lashes-safe (wider protect)
    soft = _feather(area, max(60.0, 0.6 * eye_h)) * (1.0 - _feather(protect_wide, sigma * 0.5))
    res = smooth_under_eye_texture(res, soft, args.lines, sigma, protect=protect_wide)

    Image.fromarray(to_uint8(res)).save(args.out, quality=96)
    print(f"polished (whites={args.whites} discolor={args.discolor} lines={args.lines}) -> {args.out}")


if __name__ == "__main__":
    main()
