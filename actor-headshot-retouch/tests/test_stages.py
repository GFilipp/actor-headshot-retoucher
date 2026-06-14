"""Unit tests for the individual pipeline stages."""
from __future__ import annotations

import numpy as np

from retoucher.align import align_to_reference
from retoucher.blend import composite
from retoucher.config import PipelineConfig
from retoucher.diff import compute_touch_up_map, frequency_separate
from retoucher.mask import build_masks

from _synth import BLEMISHES, CORNER, UNDER_EYE, disk, luma, translate


# --- alignment ---------------------------------------------------------------

def test_align_recovers_known_shift(original):
    moving = translate(original, 3, -2)
    res = align_to_reference(original, moving)
    assert res.success and res.method.startswith("ecc")
    interior = (slice(24, 232), slice(24, 232))
    err = float(np.abs(res.warped[interior] - original[interior]).mean())
    assert err < 0.02, f"alignment residual too high: {err}"


def test_align_handles_size_mismatch(original):
    import cv2
    small = cv2.resize(original, (128, 128))
    res = align_to_reference(original, small)
    assert res.warped.shape == original.shape


# --- frequency separation / touch-up map ------------------------------------

def test_frequency_separation_reconstructs(original):
    low, high = frequency_separate(original, 8.0)
    assert np.allclose(low + high, original, atol=1e-5)


def test_map_flags_under_eye_and_neutralizes_cast(original, target):
    tmap = compute_touch_up_map(original, target, 8.0)
    reg = UNDER_EYE[0]
    # Removing brown/purple discoloration means a negative R delta there.
    assert tmap.low_delta[reg[0], reg[1], 0].mean() < -0.01
    # The uniform warm cast is neutralized, so a clean corner is ~unchanged.
    assert float(np.abs(tmap.low_delta[CORNER]).mean()) < 0.01


def test_map_flags_marks(original, target):
    tmap = compute_touch_up_map(original, target, 8.0)
    cy, cx = BLEMISHES[0]
    assert tmap.mark_score[disk(cy, cx, 4)].max() > 0.05


# --- masks -------------------------------------------------------------------

def test_masks_confine_edits(original, target):
    tmap = compute_touch_up_map(original, target, 8.0)
    masks = build_masks(original, tmap, PipelineConfig())
    edited, untouched = masks.edited(), masks.untouched()
    reg = UNDER_EYE[0]
    assert edited[reg[0], reg[1]].mean() > 0.3      # under-eye is edited
    assert untouched[CORNER].mean() > 0.9           # clean corner is protected
    cy, cx = BLEMISHES[0]
    assert masks.by_kind["heal"][disk(cy, cx, 4)].max() > 0.2


# --- blend -------------------------------------------------------------------

def test_blend_reduces_discoloration_and_preserves_texture(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    masks = build_masks(original, tmap, cfg)
    result = composite(original, tmap, masks, cfg)

    reg = UNDER_EYE[0]
    before = original[reg[0], reg[1], 0].mean()
    after = result[reg[0], reg[1], 0].mean()
    assert after < before - 0.01, "under-eye discoloration not reduced"

    # Texture (high-frequency energy) is largely preserved -> no plastic skin.
    def hf_energy(img):
        _, high = frequency_separate(img, cfg.freq_sigma)
        return float(np.mean(high ** 2))
    assert hf_energy(result) > 0.5 * hf_energy(original)

    # A clean corner does not drift (cast neutralized, mask confined).
    assert float(np.abs(result[CORNER] - original[CORNER]).mean()) < 0.02


def test_blend_heals_marks(original, target):
    cfg = PipelineConfig()
    tmap = compute_touch_up_map(original, target, cfg.freq_sigma)
    masks = build_masks(original, tmap, cfg)
    result = composite(original, tmap, masks, cfg)
    cy, cx = BLEMISHES[0]
    spot = disk(cy, cx, 3)
    assert luma(result)[spot].mean() > luma(original)[spot].mean() + 0.03
