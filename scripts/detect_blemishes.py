"""Detect candidate blemishes on visible skin and mark them with numbers + coords.

Model-independent: redness (LAB a*) and dark-spot anomalies on skin, excluding
eyes/brows/lips (protect) and the under-eye (handled separately) and hair/shadow/
highlights. Prints each candidate's pixel coord so a chosen one can be healed.

    .venv312/bin/python scripts/detect_blemishes.py inputs/headshot5.jpg \
        inputs/blemish5.png --top 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from retoucher import faceparse, image_io
from retoucher.mask import skin_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    rgb = image_io.load(Path(args.image)).pixels.astype(np.float32)
    u8 = image_io.to_uint8(rgb)
    H, W = rgb.shape[:2]
    lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    L, a = lab[..., 0], lab[..., 1]

    region = skin_mask(rgb)
    geom = faceparse.detect(rgb)
    if geom is not None:
        protect = cv2.dilate(geom.protect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        region = region * (1.0 - np.clip(protect + geom.under_eye, 0, 1))
    region = region * ((L > 25) & (L < 92)).astype(np.float32)   # drop hair / shadow / blown highlights

    red = np.clip(a - cv2.GaussianBlur(a, (0, 0), 30), 0, None)
    dark = np.clip(cv2.GaussianBlur(L, (0, 0), 30) - L, 0, None)
    score = (red / 6.0 + dark / 12.0) * (region > 0.5)

    cand = (score > 0.6).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 8 <= area <= 1800:
            cx, cy = cent[i]
            blobs.append((float(score[labels == i].mean()) * area, int(cx), int(cy), int(area)))
    blobs.sort(reverse=True)
    blobs = blobs[: args.top]

    mark = u8.copy()
    coords = []
    for rank, (_, cx, cy, area) in enumerate(blobs, 1):
        r = max(18, int(round(1.6 * (area / 3.14) ** 0.5)))
        cv2.circle(mark, (cx, cy), r + 10, (0, 255, 0), 4)
        cv2.putText(mark, str(rank), (cx + r + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        coords.append((rank, cx, cy, r))
        print(f"#{rank}: ({cx},{cy}) r~{r} area={area}")
    Image.fromarray(mark).save(args.out)
    print(f"marked {len(coords)} candidates -> {args.out}")


if __name__ == "__main__":
    main()
