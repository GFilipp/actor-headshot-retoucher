"""Image loading, colour-managed saving, and versioned output.

Processing happens in float32 RGB in [0, 1] so bit depth is purely an IO
concern. ICC profile and EXIF are preserved on save when present. The original
file is never overwritten: outputs are written as ``<stem>_retouch_vN.<ext>``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Pillow >= 9 exposes Resampling; fall back for older installs.
try:  # pragma: no cover - trivial shim
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _LANCZOS = Image.LANCZOS

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".rw2", ".dng", ".orf"}


@dataclass
class LoadedImage:
    pixels: np.ndarray  # float32 RGB, [0, 1], shape (H, W, 3)
    icc: bytes | None = None
    exif: bytes | None = None
    source_path: Path | None = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.pixels.shape[:2]
        return w, h

    @property
    def megapixels(self) -> float:
        h, w = self.pixels.shape[:2]
        return (h * w) / 1e6


def clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def to_float(arr: np.ndarray) -> np.ndarray:
    """Convert a uint8/uint16 image to float32 in [0, 1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return clip01(arr.astype(np.float32))


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (clip01(arr) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(arr: np.ndarray) -> np.ndarray:
    return (clip01(arr) * 65535.0 + 0.5).astype(np.uint16)


def _load_raw(path: Path) -> np.ndarray:
    import rawpy  # optional; only needed for RAW sources

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(no_auto_bright=True, output_bps=16, gamma=(2.222, 4.5))
    return to_float(rgb)


def load(path: str | Path) -> LoadedImage:
    path = Path(path).expanduser()
    if path.suffix.lower() in RAW_EXTENSIONS:
        return LoadedImage(pixels=_load_raw(path), source_path=path)

    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        exif = im.info.get("exif")
        im = im.convert("RGB")
        arr = np.asarray(im)
    return LoadedImage(pixels=to_float(arr), icc=icc, exif=exif, source_path=path)


def resize_to_megapixels(pixels: np.ndarray, max_mp: float) -> tuple[np.ndarray, float]:
    """Downscale so the image is at most ``max_mp`` megapixels.

    Returns the resized float image and the scale factor applied (1.0 if none).
    """
    h, w = pixels.shape[:2]
    mp = (h * w) / 1e6
    if mp <= max_mp:
        return pixels, 1.0
    scale = (max_mp / mp) ** 0.5
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = Image.fromarray(to_uint8(pixels)).resize((new_w, new_h), _LANCZOS)
    return to_float(np.asarray(im)), scale


def resize_to(pixels: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resample ``pixels`` to an exact (width, height)."""
    w, h = size_wh
    im = Image.fromarray(to_uint8(pixels)).resize((w, h), _LANCZOS)
    return to_float(np.asarray(im))


def _next_version_path(out_dir: Path, stem: str, suffix: str, ext: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = out_dir / f"{stem}_{suffix}_v{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def save_versioned(
    pixels: np.ndarray,
    out_dir: str | Path,
    stem: str,
    *,
    suffix: str = "retouch",
    ext: str = ".jpg",
    icc: bytes | None = None,
    exif: bytes | None = None,
    quality: int = 96,
) -> Path:
    """Write a never-overwriting versioned output, preserving ICC/EXIF."""
    out_dir = Path(out_dir).expanduser()
    target = _next_version_path(out_dir, stem, suffix, ext)
    im = Image.fromarray(to_uint8(pixels))
    ext_l = ext.lower()
    save_kwargs: dict = {}
    if icc and ext_l in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        save_kwargs["icc_profile"] = icc
    # EXIF support varies by format/Pillow version; only attach where it is safe.
    if exif and ext_l in {".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        save_kwargs["exif"] = exif
    if ext_l in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=quality, subsampling=0)  # 4:4:4, chroma-safe
    im.save(target, **save_kwargs)
    return target


def save_master_tiff(pixels: np.ndarray, path: str | Path) -> Path:
    """Write a 16-bit lossless TIFF master (via OpenCV, BGR order)."""
    import cv2

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(to_uint16(pixels), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return path
