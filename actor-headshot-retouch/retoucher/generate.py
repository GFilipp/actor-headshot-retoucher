"""The retouch-target generator.

The generative model proposes "what good looks like". The deterministic
pipeline does the surgical transfer back onto the original. The model is
therefore hidden behind a one-method interface so OpenAI (the current choice)
can be swapped for FLUX.1 Kontext, Gemini, or a local model without touching
the pipeline.
"""
from __future__ import annotations

import base64
import io
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from .image_io import to_float, to_uint8


class Generator(Protocol):
    """Anything that turns an image + instruction into a retouch target."""

    def edit(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        """Return a retouched RGB float image in [0, 1]."""
        ...


def _default_mock_transform(image_rgb: np.ndarray) -> np.ndarray:
    """A plausible stand-in retouch for offline / --dry-run runs.

    Evens low-frequency tone and lifts shadows slightly while preserving
    high-frequency texture, so the pipeline has a real (if mild) delta to
    transfer. Not a real retouch; just enough to exercise the wiring.
    """
    import cv2

    bgr = cv2.cvtColor(to_uint8(image_rgb), cv2.COLOR_RGB2BGR)
    # Edge-preserving smooth approximates a retoucher evening out blotches.
    smoothed = cv2.bilateralFilter(bgr, d=9, sigmaColor=40, sigmaSpace=9)
    out = to_float(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB))
    # Gentle shadow lift.
    out = np.clip(out * 0.97 + 0.015, 0.0, 1.0)
    return out.astype(np.float32)


class MockGenerator:
    """No-API generator for tests and ``--dry-run``.

    Pass ``transform`` to control the output exactly (tests inject a known
    target); the default applies a mild synthetic retouch.
    """

    def __init__(self, transform: Callable[[np.ndarray], np.ndarray] | None = None):
        self._transform = transform or _default_mock_transform

    def edit(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        return self._transform(image_rgb).astype(np.float32)


# gpt-image-1 only emits these three sizes. Requesting the bucket closest to the
# input aspect keeps the returned target close in framing, so alignment stays
# clean (a wrong aspect would force an anisotropic resize before diffing).
GPT_IMAGE_SIZES = {"1024x1024": 1.0, "1536x1024": 1536 / 1024, "1024x1536": 1024 / 1536}


def closest_gpt_image_size(width: int, height: int) -> str:
    aspect = width / max(1, height)
    return min(GPT_IMAGE_SIZES, key=lambda s: abs(GPT_IMAGE_SIZES[s] - aspect))


class OpenAIGenerator:
    """OpenAI image-edit backend (gpt-image-1 by default).

    Kept deliberately thin. Per the plan this is the current choice for
    accessibility; the noisy-diff and deprecation risk are mitigated by (a) the
    pipeline never trusting these pixels globally and (b) this seam making a
    later model swap a small change.
    """

    def __init__(self, model: str = "gpt-image-1", quality: str = "high", size: str = "auto"):
        self.model = model
        self.quality = quality  # "high" gives a more faithful target (costs more)
        self.size = size  # "auto" -> closest aspect bucket; or pass an explicit size

    def edit(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'openai' package is required for OpenAIGenerator. "
                "Install it (pip install 'actor-headshot-retoucher[openai]') "
                "or run with --dry-run / a MockGenerator."
            ) from exc

        h, w = image_rgb.shape[:2]
        size = closest_gpt_image_size(w, h) if self.size == "auto" else self.size

        buf = io.BytesIO()
        Image.fromarray(to_uint8(image_rgb)).save(buf, format="PNG")
        # Tuple form (name, bytes, mime) is the most SDK-version-robust.
        image_arg = ("source.png", buf.getvalue(), "image/png")

        client = OpenAI()  # reads OPENAI_API_KEY from the environment
        result = client.images.edit(
            model=self.model, image=image_arg, prompt=prompt,
            size=size, quality=self.quality, n=1,
        )
        data = getattr(result, "data", None) or []
        b64 = getattr(data[0], "b64_json", None) if data else None
        if not b64:
            raise RuntimeError("OpenAI image edit returned no image data.")
        with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
            arr = np.asarray(im.convert("RGB"))
        return to_float(arr)


def get_generator(name: str = "openai", **kwargs) -> Generator:
    """Factory so the CLI can pick a backend by name."""
    name = (name or "openai").lower()
    if name in {"mock", "dry-run", "dryrun", "none"}:
        return MockGenerator()
    if name == "openai":
        return OpenAIGenerator(**kwargs)
    raise ValueError(f"Unknown generator backend: {name!r}")
