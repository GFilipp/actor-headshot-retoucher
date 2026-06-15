"""Surgically paste ONE region of a Gemini-retouched image onto the full-res original.

The generated image is only used as a donor for a specific region (default: the
under-eye / tear-trough). We:
  1. register the donor onto the original's face with a landmark affine,
  2. colour-match the donor to the original over face skin (so any added
     rouge/brightness is neutralised — we keep the original's tone),
  3. composite only the chosen region through a feathered mask.
Everything outside the region stays the original photo: real expression, real
eyes, full resolution.

    .venv312/bin/python scripts/surgical_paste.py inputs/headshot1.jpg \
        inputs/gemini_out.png inputs/surgical_out.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from retoucher import faceparse, image_io
from retoucher.image_io import clip01, to_float, to_uint8


def color_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shift src's per-channel LAB mean to ref's, measured over mask. Neutralises a
    global colour/brightness difference (e.g. added rouge) while keeping local detail."""
    sel = mask > 0.5
    if int(sel.sum()) < 50:
        return src
    s = rgb2lab(clip01(src)).astype(np.float32)
    r = rgb2lab(clip01(ref)).astype(np.float32)
    for c in range(3):
        s[..., c] += float(r[..., c][sel].mean() - s[..., c][sel].mean())
    return clip01(lab2rgb(s)).astype(np.float32)


def _dilate(m, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px) * 2 + 1, int(px) * 2 + 1))
    return cv2.dilate(m.astype(np.float32), k)


def build_region(g, name, grow=1.0):
    """Region of the ORIGINAL to replace with the (registered) donor.

    ``grow`` scales the periorbital reach (× eye height). Keep it modest (~0.5) for a
    RAW paste so it stays on the naturally-smooth eye area and doesn't turn the sharp
    cheek into a soft patch; go bigger only for low-freq transfer."""
    if name == "face":
        return g.face_oval.astype(np.float32)
    if name == "periorbital":
        # ROUNDED disc around BOTH orbits (under-eye, corners, crow's feet), not the eyeball.
        # No straight edges -> no faint rectangular tell.
        ys = np.where(g.eyes.max(axis=1) > 0.5)[0]
        eye_h = float(ys.max() - ys.min()) if ys.size else 0.08 * g.eyes.shape[0]
        peri = _dilate(g.eyes, grow * eye_h)
        return np.clip(peri * (1.0 - g.eyes) * g.face_oval, 0, 1)
    return g.under_eye.astype(np.float32)


REGION_NAMES = ("under_eye", "periorbital", "face")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("donor", help="Gemini-retouched image to borrow from")
    ap.add_argument("out")
    ap.add_argument("--region", default="under_eye", choices=list(REGION_NAMES))
    ap.add_argument("--strength", type=float, default=1.0, help="opacity 0..1")
    ap.add_argument("--feather", type=float, default=None, help="feather px (default ~1% of long edge)")
    ap.add_argument("--mode", default="paste", choices=["paste", "transfer", "luma"],
                    help="paste = replace pixels; transfer = donor low-freq tone only (no crepe); "
                         "luma = take donor LUMINANCE (carries crepe/texture fix) but KEEP original "
                         "colour (no colour distortion)")
    ap.add_argument("--transfer-sigma", type=float, default=None,
                    help="low-freq sigma for --mode transfer (default ~0.6%% of long edge)")
    ap.add_argument("--grow", type=float, default=1.0, help="periorbital reach x eye-height")
    args = ap.parse_args()

    loaded = image_io.load(Path(args.original))
    orig = loaded.pixels.astype(np.float32)
    H, W = orig.shape[:2]
    donor = to_float(np.array(Image.open(args.donor).convert("RGB")))

    lm_o = faceparse.landmarks(orig)
    lm_d = faceparse.landmarks(donor)
    if lm_o is None or lm_d is None:
        raise SystemExit("need face landmarks in both images (MediaPipe).")
    M, _ = cv2.estimateAffinePartial2D(lm_d, lm_o, method=cv2.RANSAC)
    donor_a = cv2.warpAffine(donor, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    geom = faceparse.detect(orig)
    if geom is None:
        raise SystemExit("no face geometry on the original.")
    region = build_region(geom, args.region, args.grow).astype(np.float32)

    # colour-match the donor to the original over face skin OUTSIDE the pasted region
    match = np.clip(geom.face_oval - region - geom.protect, 0, 1)
    donor_m = color_match(donor_a, orig, match)

    feather = args.feather if args.feather is not None else max(4.0, 0.01 * max(H, W))
    m = cv2.GaussianBlur(region, (0, 0), sigmaX=feather) * float(args.strength)
    # hard-protect brows / eyes / lips: the feathered paste must never land on them
    prot = cv2.GaussianBlur(_dilate(geom.protect, max(2.0, feather * 0.4)), (0, 0), sigmaX=feather * 0.4)
    m = (m * (1.0 - np.clip(prot, 0, 1)))[..., None]
    if args.mode == "transfer":
        s = args.transfer_sigma or max(8.0, 0.006 * max(H, W))
        delta = cv2.GaussianBlur(donor_m, (0, 0), sigmaX=s) - cv2.GaussianBlur(orig, (0, 0), sigmaX=s)
        out = clip01(orig + m * delta)           # donor's tone/darkness fix; original texture kept
    elif args.mode == "luma":
        lab_o = rgb2lab(clip01(orig)).astype(np.float32)
        lab_d = rgb2lab(clip01(donor_m)).astype(np.float32)
        w = m[..., 0]
        lab_o[..., 0] = lab_o[..., 0] * (1 - w) + lab_d[..., 0] * w   # donor luminance (crepe fix)
        out = clip01(lab2rgb(lab_o))                                  # original a*/b* kept (no colour shift)
    else:
        out = clip01(orig * (1 - m) + donor_m * m)

    Image.fromarray(to_uint8(out)).save(args.out, quality=96)
    print(f"pasted '{args.region}' (feather={feather:.0f}px, strength={args.strength}) "
          f"-> {args.out}  [{W}x{H}]")


if __name__ == "__main__":
    main()
