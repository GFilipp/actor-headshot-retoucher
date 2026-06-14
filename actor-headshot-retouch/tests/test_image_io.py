"""I/O: never-overwrite versioning, load round-trip, generator downscale."""
from __future__ import annotations

import numpy as np

from retoucher import image_io

from _synth import make_original


def test_versioned_save_never_overwrites(tmp_path):
    img = make_original()
    p1 = image_io.save_versioned(img, tmp_path, "shot")
    p2 = image_io.save_versioned(img, tmp_path, "shot")
    assert p1.name == "shot_retouch_v1.jpg"
    assert p2.name == "shot_retouch_v2.jpg"
    assert p1.exists() and p2.exists()


def test_load_roundtrip(tmp_path):
    img = make_original()
    p = image_io.save_versioned(img, tmp_path, "shot", ext=".png")
    loaded = image_io.load(p)
    assert loaded.pixels.shape == img.shape
    assert loaded.pixels.dtype == np.float32
    assert float(np.abs(loaded.pixels - img).mean()) < 0.01


def test_resize_to_megapixels_downscales():
    big = np.zeros((2000, 2000, 3), np.float32)
    small, scale = image_io.resize_to_megapixels(big, 1.0)
    assert small.shape[0] * small.shape[1] <= 1.05e6
    assert scale < 1.0


def test_resize_to_megapixels_keeps_small_image():
    small_in = np.zeros((256, 256, 3), np.float32)
    out, scale = image_io.resize_to_megapixels(small_in, 1.5)
    assert scale == 1.0
    assert out.shape == small_in.shape
