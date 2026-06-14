"""Tunable thresholds and per-run pipeline settings.

Every knob that affects retouch strength or quality gating lives here, so a
non-engineer can tune behaviour in one place and tests can assert against named
constants instead of magic numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Regions the skill spec enumerates (see SKILL.md / retouch_workflows.md).
# Each maps to a mask the pipeline can build and edit independently.
REGIONS = (
    "under_eye",
    "eye_whites",
    "eyelids",
    "skin_marks",
    "neck",
    "hand",
    "flyaways",
)

# Region edit kind. "tone" transfers the low-frequency colour/tone delta from
# the aligned target. "heal" removes a local mark using the ORIGINAL's
# surrounding texture (never the generated pixels).
REGION_KIND = {
    "under_eye": "tone",
    "eye_whites": "tone",
    "eyelids": "tone",
    "neck": "tone",
    "hand": "tone",
    "skin_marks": "heal",
    "flyaways": "heal",
}


@dataclass
class QAThresholds:
    """Pass/fail gates.

    A gate whose optional backend is unavailable is reported as ``skipped``,
    never silently treated as a pass.
    """

    # Identity preservation (InsightFace / ArcFace cosine). Skipped without backend.
    identity_min_cosine: float = 0.60
    # Regions that must NOT change should stay near-identical (SSIM, 0..1).
    untouched_min_ssim: float = 0.92
    # Optional perceptual check on untouched regions (LPIPS). Skipped without torch.
    untouched_max_lpips: float = 0.10
    # An edit must be visible: mean CIEDE2000 inside edited regions >= this.
    edited_min_delta_e: float = 1.5
    # An edit must not be a cartoon: mean CIEDE2000 inside edited regions <= this.
    edited_max_delta_e: float = 14.0
    # Skin must keep texture: high-frequency energy may drop at most this fraction.
    max_hf_energy_loss: float = 0.45


@dataclass
class PipelineConfig:
    # hybrid-map | light-retouch | light-regen
    mode: str = "hybrid-map"
    # Spatial params below (freq_sigma, feather_px) are expressed at this
    # reference width; the pipeline scales them to the image's actual size so
    # behaviour is the same on a 512px and a 4096px file. Never assume a fixed
    # resolution.
    reference_dim: int = 1024
    # Frequency separation: Gaussian sigma (px at reference_dim) that splits
    # low-freq tone from high-freq texture. Larger = coarser tone layer.
    freq_sigma: float = 8.0
    # Per-region transfer strength in [0, 1]. 1.0 takes the full mapped delta.
    strength: float = 0.85
    # Remove a uniform colour cast the model may add, so complexion never drifts.
    neutralize_global_cast: bool = True
    # Feather (px at reference_dim) applied to every mask edge to avoid seams.
    feather_px: float = 12.0
    # Max megapixels sent to the generator; the original is restored afterwards.
    generator_max_mp: float = 1.5
    # Alignment convergence.
    ecc_iterations: int = 200
    ecc_epsilon: float = 1e-5
    # Output.
    jpeg_quality: int = 96
    qa: QAThresholds = field(default_factory=QAThresholds)

    def to_dict(self) -> dict:
        return asdict(self)
