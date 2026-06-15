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
import os
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


# Last-resort only, used if the live models API can't be reached. The OPERATIVE
# default is auto-discovered at runtime (see discover_latest_image_model), so a
# new release like gpt-image-3 is picked up with zero code change.
FALLBACK_OPENAI_IMAGE_MODEL = "gpt-image-2"

# The three fixed sizes the legacy gpt-image-1 supports. gpt-image-2 takes
# flexible sizes and "auto", so this is only used when you pin gpt-image-1 and
# pass size="bucket".
GPT_IMAGE_1_SIZES = {"1024x1024": 1.0, "1536x1024": 1536 / 1024, "1024x1536": 1024 / 1536}


def closest_gpt_image_size(width: int, height: int) -> str:
    """Closest legacy gpt-image-1 size bucket to the input aspect ratio."""
    aspect = width / max(1, height)
    return min(GPT_IMAGE_1_SIZES, key=lambda s: abs(GPT_IMAGE_1_SIZES[s] - aspect))


def discover_latest_image_model(client) -> str:
    """Newest non-mini ``gpt-image*`` model the account can see.

    Picks by the model's ``created`` timestamp, so new releases are adopted
    automatically with no code change. Mini variants are excluded as the
    cheaper, lower-fidelity tier.
    """
    models = list(client.models.list().data)
    candidates = [
        m for m in models
        if str(getattr(m, "id", "")).startswith("gpt-image") and "mini" not in str(getattr(m, "id", ""))
    ]
    if not candidates:
        raise RuntimeError("no gpt-image models available to this account")
    return max(candidates, key=lambda m: getattr(m, "created", 0) or 0).id


def pinned_openai_model(model: str | None = None) -> str | None:
    """A model the user pinned via arg or $OPENAI_IMAGE_MODEL, else None (auto-discover)."""
    return model or os.environ.get("OPENAI_IMAGE_MODEL") or None


class OpenAIGenerator:
    """OpenAI image-edit backend.

    By default it auto-discovers OpenAI's latest gpt-image model at call time, so
    a new release needs no code change. Pin a specific model with the ``model``
    arg or ``$OPENAI_IMAGE_MODEL``. The pipeline never trusts these pixels
    globally, and this one-method seam makes swapping to FLUX.1 Kontext / Gemini
    / a local model trivial.
    """

    def __init__(self, model: str | None = None, quality: str = "high", size: str = "auto"):
        self.pinned_model = pinned_openai_model(model)  # None => auto-discover latest
        self.quality = quality  # "high" gives a more faithful target (costs more)
        self.size = size  # "auto" (default), explicit "WxH", or "bucket" (legacy gpt-image-1)
        self._discovered: str | None = None

    @property
    def model(self) -> str | None:
        """The model that will be used: the pinned id, or the discovered one once
        resolved. None means discovery has not run yet (it runs at edit time)."""
        return self.pinned_model or self._discovered

    def _model_for(self, client) -> str:
        if self.pinned_model:
            return self.pinned_model
        if not self._discovered:
            try:
                self._discovered = discover_latest_image_model(client)
            except Exception:
                self._discovered = FALLBACK_OPENAI_IMAGE_MODEL
        return self._discovered

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
        # gpt-image-2 handles "auto"; "bucket" maps to a legacy gpt-image-1 size.
        size = closest_gpt_image_size(w, h) if self.size == "bucket" else self.size

        buf = io.BytesIO()
        Image.fromarray(to_uint8(image_rgb)).save(buf, format="PNG")
        # Tuple form (name, bytes, mime) is the most SDK-version-robust.
        image_arg = ("source.png", buf.getvalue(), "image/png")

        client = OpenAI()  # reads OPENAI_API_KEY from the environment
        result = client.images.edit(
            model=self._model_for(client), image=image_arg, prompt=prompt,
            size=size, quality=self.quality, n=1,
        )
        data = getattr(result, "data", None) or []
        b64 = getattr(data[0], "b64_json", None) if data else None
        if not b64:
            raise RuntimeError("OpenAI image edit returned no image data.")
        with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
            arr = np.asarray(im.convert("RGB"))
        return to_float(arr)


# Google Gemini image models, newest-first. The Generative Language *auth* keys
# (AQ.* prefix) block models.list, so we try known ids in order rather than discover.
FALLBACK_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-exp-image-generation",
]
DEFAULT_GEMINI_KEY_FILE = "~/Desktop/gemini.txt"


class GeminiGenerator:
    """Google Gemini image-edit backend behind the same one-method seam.

    Key resolution: ``$GEMINI_API_KEY`` / ``$GOOGLE_API_KEY``, else a key file
    (default ``~/Desktop/gemini.txt``). Prompt-agnostic — the caller supplies the
    persona/instruction, so there is no photo-specific prompt baked in here.
    """

    def __init__(self, model: str | None = None, key_path: str = DEFAULT_GEMINI_KEY_FILE,
                 max_edge: int = 1536):
        self.pinned_model = model
        self.key_path = key_path
        self.max_edge = max_edge

    def _key(self) -> str:
        for env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if os.environ.get(env):
                return os.environ[env]
        path = os.path.expanduser(self.key_path)
        if os.path.exists(path):
            return open(path).read().strip()
        raise RuntimeError(
            "No Gemini key: set $GEMINI_API_KEY/$GOOGLE_API_KEY or place it in "
            f"{self.key_path}."
        )

    def edit(self, image_rgb: np.ndarray, prompt: str) -> np.ndarray:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'google-genai' package is required for GeminiGenerator "
                "(pip install google-genai), or run with --dry-run / a MockGenerator."
            ) from exc

        client = genai.Client(api_key=self._key())
        img = Image.fromarray(to_uint8(image_rgb))
        if self.max_edge and max(img.size) > self.max_edge:
            s = self.max_edge / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)))

        models = [self.pinned_model] if self.pinned_model else GEMINI_IMAGE_MODELS
        configs = [
            types.GenerateContentConfig(response_modalities=["IMAGE"]),
            types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            None,
        ]
        last = ""
        for m in models:
            for cfg in configs:
                try:
                    resp = client.models.generate_content(model=m, contents=[prompt, img], config=cfg)
                    for cand in getattr(resp, "candidates", None) or []:
                        for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                            inl = getattr(part, "inline_data", None)
                            if inl and getattr(inl, "data", None):
                                with Image.open(io.BytesIO(inl.data)) as im:
                                    return to_float(np.asarray(im.convert("RGB")))
                    last = f"{m}: no image in response"
                except Exception as e:  # pragma: no cover - network/SDK dependent
                    last = f"{m}: {type(e).__name__} {str(e)[:120]}"
        raise RuntimeError(f"Gemini produced no image. Last: {last}")


def edit_n(generator: Generator, image_rgb: np.ndarray, prompt: str, n: int = 1) -> list[np.ndarray]:
    """Draw ``n`` candidates from a generator. Gemini is stochastic and some samples
    carry artifacts (stipple) — sample, then let the self-audit pick the cleanest."""
    return [generator.edit(image_rgb, prompt) for _ in range(max(1, int(n)))]


def get_generator(name: str = "openai", **kwargs) -> Generator:
    """Factory so the CLI / router can pick a backend by name."""
    name = (name or "openai").lower()
    if name in {"mock", "dry-run", "dryrun", "none"}:
        return MockGenerator()
    if name == "openai":
        return OpenAIGenerator(**kwargs)
    if name in {"gemini", "google"}:
        return GeminiGenerator(**kwargs)
    raise ValueError(f"Unknown generator backend: {name!r}")
