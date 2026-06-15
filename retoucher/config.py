"""Tunable thresholds and per-run pipeline settings.

Every knob that affects retouch strength or quality gating lives here, so a
non-engineer can tune behaviour in one place and tests can assert against named
constants instead of magic numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Regions the skill spec enumerates (see SKILL.md / retouch_workflows.md).
REGIONS = (
    "under_eye",
    "eye_whites",
    "eyelids",
    "skin_marks",
    "neck",
    "hand",
    "flyaways",
)


@dataclass
class QAThresholds:
    """Pass/fail gates. A gate whose optional backend is unavailable is reported
    as ``skipped``, never silently treated as a pass."""

    identity_min_cosine: float = 0.60        # InsightFace / ArcFace (optional)
    untouched_min_ssim: float = 0.92         # off-edit regions stay near-identical
    untouched_max_lpips: float = 0.10        # perceptual, optional (torch)
    edited_min_delta_e: float = 1.5          # an edit must be visible
    edited_max_delta_e: float = 14.0         # an edit must not be a cartoon
    max_hf_energy_loss: float = 0.45         # skin must keep texture
    protected_min_ssim: float = 0.985        # brows/eyes/lips must NOT change


@dataclass
class PipelineConfig:
    mode: str = "hybrid-map"
    # Spatial params are expressed at this reference width and scaled to the
    # image's actual size by the pipeline, so behaviour is resolution-independent.
    reference_dim: int = 1024
    freq_sigma: float = 8.0
    feather_px: float = 12.0

    # --- transfer strengths (per edit kind), each in [0, 1] ---
    tone_strength: float = 0.85              # colour/tone (chroma-only) transfer
    under_eye_strength: float = 0.6          # deterministic tear-trough lightening

    # --- form-preserving tone transfer ---
    neutralize_global_cast: bool = True
    # Move only chroma (a*/b*); clamp the per-pixel chroma shift (LAB units) so a
    # broad recolor can't flatten facial form.
    max_chroma_delta: float = 12.0
    guided_radius: int = 8                   # edge-aware blend of the correction
    guided_eps: float = 1e-3

    # --- blemish detection / healing ---
    mark_luma_thresh: float = 0.06           # dark spot the model also removed
    mark_red_thresh: float = 0.30            # model-independent redness anomaly (0..1)
    mark_max_blob_frac: float = 0.01         # marks are small; skip big shadows / moles

    # --- feature protection / confinement ---
    protect_dilate_px: float = 4.0           # grow the never-edit feature mask
    skin_erode_px: float = 6.0               # pull edits in from skin / feature boundaries
    edge_gate: float = 0.25                  # downweight tone edits near hard edges

    generator_max_mp: float = 1.5
    ecc_iterations: int = 200
    ecc_epsilon: float = 1e-5
    jpeg_quality: int = 96
    qa: QAThresholds = field(default_factory=QAThresholds)

    def to_dict(self) -> dict:
        return asdict(self)
