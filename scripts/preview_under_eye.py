"""Preview the under-eye corrector on a real headshot (model-independent, no API key).

Per eye it shows: original | one column per texture strength | mask overlay,
cropped to the tear-trough. The mask overlay tints the treated region red on the
original so you can confirm it lands beside the nose. Use it to judge the
smoothing level and check nothing is overbaked, at full resolution:

    .venv312/bin/python scripts/preview_under_eye.py inputs/headshot.jpg \
        --texture 0.6,0.7 --out under_eye_preview.png

Runs on the original's own pixels (the texture fix is model-independent), so it
needs MediaPipe but no OpenAI key. By default it processes at FULL resolution
(--cap-mp 0); pass --cap-mp 8 to mirror the shipped 8 MP processing cap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from retoucher import faceparse, image_io
from retoucher.blend import correct_under_eye, smooth_under_eye_texture
from retoucher.config import PipelineConfig
from retoucher.image_io import to_uint8


def _eye_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    m = (mask > 0.5).astype(np.uint8)
    num, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, num)
             if stats[i, cv2.CC_STAT_AREA] >= 50]
    return sorted(boxes, key=lambda b: b[0])  # left to right


def _label(panel_bgr: np.ndarray, text: str) -> None:
    cv2.rectangle(panel_bgr, (0, 0), (max(70, 9 * len(text)), 26), (0, 0, 0), -1)
    cv2.putText(panel_bgr, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--shadow", type=float, default=None, help="under_eye_strength override (0..1)")
    ap.add_argument("--texture", default="0.6", help="comma list of texture strengths, e.g. 0.6,0.7")
    ap.add_argument("--sigma", type=float, default=None, help="absolute texture sigma px (default: scaled freq_sigma)")
    ap.add_argument("--cap-mp", type=float, default=0.0, help="cap working megapixels (0 = full resolution)")
    ap.add_argument("--out", default="under_eye_preview.png")
    args = ap.parse_args()

    strengths = [float(x) for x in args.texture.split(",") if x.strip() != ""]
    cfg = PipelineConfig()
    shadow = args.shadow if args.shadow is not None else cfg.under_eye_strength

    loaded = image_io.load(Path(args.image))
    rgb = loaded.pixels
    if args.cap_mp > 0:
        rgb, _ = image_io.resize_to_megapixels(rgb, args.cap_mp)
    spatial = max(rgb.shape[:2]) / float(cfg.reference_dim)
    sigma = args.sigma if args.sigma is not None else cfg.freq_sigma * spatial

    geom = faceparse.detect(rgb)
    if geom is None:
        raise SystemExit("No face geometry (MediaPipe unavailable or no face found).")
    region = geom.under_eye
    if float(region.max()) <= 0:
        raise SystemExit("Under-eye region is empty for this image.")
    # Match the pipeline's dilated feature protection so the preview is faithful.
    pr = max(1, int(round(cfg.protect_dilate_px * spatial)))
    protect = cv2.dilate(geom.protect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pr * 2 + 1, pr * 2 + 1)))

    lift = correct_under_eye(rgb, region, shadow, sigma)
    cols = [("original", rgb)]
    for t in strengths:
        cols.append((f"texture {t:.2f}", smooth_under_eye_texture(lift, region, t, sigma, protect=protect)))
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * region, 0, 1)
    cols.append(("mask", overlay))

    rows = []
    for (x, y, w, h) in _eye_boxes(region):
        px, py = int(0.6 * w), int(0.9 * h)
        sl = (slice(max(0, y - py), min(rgb.shape[0], y + h + py)),
              slice(max(0, x - px), min(rgb.shape[1], x + w + px)))
        panels = []
        for name, img in cols:
            p = cv2.cvtColor(to_uint8(img[sl]), cv2.COLOR_RGB2BGR)
            _label(p, name)
            panels.append(p)
        rows.append(np.hstack(panels))

    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"wrote {args.out}  (sigma={sigma:.1f}px, cap={args.cap_mp or 'full'}, {len(rows)} eye row(s))")


if __name__ == "__main__":
    main()
