"""Diagnose the protected_features QA gate on a real headshot (mock generator).

Splits the protect region into the feature CORE (raw landmarks) vs the dilated
RING, and reports which edit kind overlaps the dilated-protect zone. Tells us
whether features are genuinely changing (core low) or it's a dilation/edit-bleed
artifact (ring low) plus an over-strict threshold.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from retoucher import faceparse, image_io, prompts
from retoucher.align import align_to_reference
from retoucher.blend import composite
from retoucher.config import PipelineConfig
from retoucher.diff import compute_touch_up_map
from retoucher.generate import MockGenerator
from retoucher.mask import build_masks


def gray(x):
    return cv2.cvtColor(image_io.to_uint8(x), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def main(path: str):
    cfg = PipelineConfig()
    loaded = image_io.load(Path(path))
    original, _ = image_io.resize_to_megapixels(loaded.pixels, cfg.max_process_mp)
    spatial = max(original.shape[:2]) / float(cfg.reference_dim)
    cfg = replace(
        cfg,
        freq_sigma=cfg.freq_sigma * spatial, feather_px=cfg.feather_px * spatial,
        protect_dilate_px=cfg.protect_dilate_px * spatial, skin_erode_px=cfg.skin_erode_px * spatial,
        guided_radius=max(1, int(round(cfg.guided_radius * spatial))),
    )
    geom = faceparse.detect(original)
    if geom is None:
        raise SystemExit("no geometry")
    gen = MockGenerator()
    gi, _ = image_io.resize_to_megapixels(original, cfg.generator_max_mp)
    target = gen.edit(gi, prompts.prompt_for(cfg.mode))
    al = align_to_reference(original, target, iterations=cfg.ecc_iterations, epsilon=cfg.ecc_epsilon)
    tmap = compute_touch_up_map(original, al.warped, cfg.freq_sigma, neutralize_cast=cfg.neutralize_global_cast)
    masks = build_masks(original, tmap, cfg, geom=geom)
    result = composite(original, tmap, masks, cfg)

    _, smap = structural_similarity(gray(original), gray(result), data_range=1.0, full=True)
    raw = geom.protect > 0.5
    dil = masks.protect > 0.5
    raw_grow = cv2.dilate(raw.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = dil & ~raw_grow
    print(f"protect CORE  SSIM = {float(smap[raw].mean()):.4f}  (px={int(raw.sum())})")
    print(f"protect RING  SSIM = {float(smap[ring].mean()):.4f}  (px={int(ring.sum())})" if ring.sum() else "ring empty")
    print(f"protect DILAT SSIM = {float(smap[dil].mean()):.4f}  (gate value; thr 0.985)")
    print(f"luma |delta| mean in dilated-protect = {float(np.abs(gray(original)[dil]-gray(result)[dil]).mean()):.4f}")
    for k, m in masks.by_kind.items():
        print(f"  edit '{k}': overlap dilated-protect frac>0.05 = {float((m[dil]>0.05).mean()):.3f}, max = {float(m[dil].max()):.3f}")
    print(f"  under_eye: overlap dilated-protect max = {float(masks.under_eye[dil].max()):.3f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "inputs/headshot1.jpg")
