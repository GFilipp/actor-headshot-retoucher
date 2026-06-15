"""Unit tests for the individual pipeline stages (v2.1)."""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from retoucher.align import align_to_reference
from retoucher.blend import apply_tone, composite, correct_under_eye
from retoucher.config import PipelineConfig
from retoucher.diff import compute_touch_up_map, frequency_separate
from retoucher.mask import build_masks

from _synth import (
    BLEMISHES, CORNER, RED_BLEMISH, UNDER_EYE,
    disk, fake_geometry, luma, translate,
)


# --- alignment ---------------------------------------------------------------

def test_align_recovers_known_shift(original):
    res = align_to_reference(original, translate(original, 3, -2))
    assert res.success and res.method.startswith("ecc")
    interior = (slice(24, 232), slice(24, 232))
    assert float(np.abs(res.warped[interior] - original[interior]).mean()) < 0.02


# --- frequency separation / touch-up map ------------------------------------

def test_frequency_separation_reconstructs(original):
    low, high = frequency_separate(original, 8.0)
    assert np.allclose(low + high, original, atol=1e-5)


def test_map_flags_under_eye_and_neutralizes_cast(original, target):
    tmap = compute_touch_up_map(original, target, 8.0)
    reg = UNDER_EYE[0]
    assert tmap.low_delta[reg[0], reg[1], 0].mean() < -0.01     # removes red discoloration
    assert float(np.abs(tmap.low_delta[CORNER]).mean()) < 0.01  # cast neutralized


def test_map_flags_dark_and_red_blemishes(original, target):
    tmap = compute_touch_up_map(original, target, 8.0)
    dcy, dcx = BLEMISHES[0]
    assert tmap.mark_score[disk(dcy, dcx, 4)].max() > 0.05      # dark spot (model-gated)
    rcy, rcx = RED_BLEMISH
    assert tmap.red_score[disk(rcy, rcx, 4)].max() > 0.30       # red spot (model-independent)
    # the reddish spot is NOT dark, so the luma detector alone would miss it
    assert tmap.mark_score[disk(rcy, rcx, 4)].max() < 0.06


# --- masks (with injected geometry) -----------------------------------------

def test_masks_confine_and_protect(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    geom = fake_geometry()
    masks = build_masks(original, tmap, cfg, geom=geom)

    assert masks.untouched()[CORNER].mean() > 0.9              # corner protected
    assert float((masks.by_kind["tone"] * geom.brows).max()) < 0.2   # tone off brows
    assert float((masks.by_kind["tone"] * geom.eyes).max()) < 0.2    # tone off eyes
    # reddish chest blemish (outside the face oval) is still reachable by heal
    rcy, rcx = RED_BLEMISH
    assert masks.by_kind["heal"][disk(rcy, rcx, 4)].max() > 0.2


def test_forced_mark_creates_heal(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    forced = np.zeros(original.shape[:2], np.float32)
    forced[disk(45, 210, 5)] = 1.0                             # clear skin, no auto-defect
    masks = build_masks(original, tmap, cfg, geom=fake_geometry(), forced=forced)
    assert masks.by_kind["heal"][disk(45, 210, 4)].max() > 0.2


# --- tone transfer: form (luminance) preserved, colour moved ----------------

def test_apply_tone_preserves_luminance_and_reduces_redness(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    masks = build_masks(original, tmap, cfg, geom=fake_geometry())
    toned = apply_tone(original, tmap, masks.by_kind["tone"], cfg.tone_strength, cfg)

    reg = UNDER_EYE[0]
    lab0 = rgb2lab(np.clip(original, 0, 1))
    lab1 = rgb2lab(np.clip(toned, 0, 1))
    # Luminance (3D form) preserved in the edited region...
    assert float(np.abs(lab1[..., 0][reg[0], reg[1]] - lab0[..., 0][reg[0], reg[1]]).mean()) < 1.0
    # ...while redness (a*) is reduced.
    assert lab1[..., 1][reg[0], reg[1]].mean() < lab0[..., 1][reg[0], reg[1]].mean() - 0.5


# --- composite: heals marks, protects features ------------------------------

def test_composite_heals_marks_and_protects_features(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    geom = fake_geometry()
    masks = build_masks(original, tmap, cfg, geom=geom)
    result = composite(original, tmap, masks, cfg)

    dcy, dcx = BLEMISHES[0]
    assert luma(result)[disk(dcy, dcx, 3)].mean() > luma(original)[disk(dcy, dcx, 3)].mean() + 0.03
    rcy, rcx = RED_BLEMISH
    a0 = rgb2lab(np.clip(original, 0, 1))[..., 1][disk(rcy, rcx, 3)].mean()
    a1 = rgb2lab(np.clip(result, 0, 1))[..., 1][disk(rcy, rcx, 3)].mean()
    assert a1 < a0 - 1.0                                       # reddish blemish de-reddened
    for feat in (geom.brows > 0.5, geom.eyes > 0.5):           # features untouched
        assert float(np.abs(result[feat] - original[feat]).mean()) < 0.01


# --- deterministic under-eye corrector --------------------------------------

def test_under_eye_corrector_lifts_shadow(original):
    geom = fake_geometry()
    ue = geom.under_eye > 0.5
    shadow = original.copy()
    shadow[ue] *= 0.7                                          # simulate tear-trough shadow
    out = correct_under_eye(shadow, geom.under_eye, strength=0.6, sigma=8.0)
    assert luma(out)[ue].mean() > luma(shadow)[ue].mean() + 0.02
