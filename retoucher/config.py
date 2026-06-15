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
class CalibrationConfig:
    """Policy thresholds for `calibrate.py` — the per-photo generative-vs-deterministic
    split. Encodes the rulebook this project learned the hard way: small/low-res faces
    can't tolerate a raw paste (texture distorts at pixel zoom), pigmentation is chromatic
    so deterministic 'barely dents' it (generative-led), mild unevenness is deterministic-
    only, stray hair is generative-only, and identity-sensitive ops cap the generative share.
    """

    large_face_frac: float = 0.05                          # >= this AND high-res -> raw paste tolerated
    paste_resolution_classes: tuple[str, ...] = ("native_high",)
    identity_gen_cap: float = 0.5                          # identity-sensitive ops cap generative share
    mild_unevenness_sev: float = 0.5                       # below -> deterministic-only
    feather_frac: float = 0.10                             # feather px = region-bbox diagonal * this
    mask_grow: float = 1.1                                 # organic mask spread (no straight edges)
    gen_weight_strong: float = 0.8                         # crepe / under-eye on a big face
    gen_weight_pigment: float = 0.7                        # pigmentation / discoloration
    gen_weight_small_face: float = 0.5                     # any generative op on a small/low-res face
    det_strength_floor: float = 0.25                       # deterministic strength at severity 0
    det_strength_ceiling: float = 0.7                      # deterministic strength at severity 1

    def to_dict(self) -> dict:
        return asdict(self)


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
    under_eye_texture_strength: float = 0.6  # soften crepey tear-trough texture (HF attenuation)
    skin_even_strength: float = 0.5          # even skin colour blotches (a*/b* only; L/form/texture kept)

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
    # Cap the working/output resolution so a 20MP+ original can't make the
    # full-image align/diff/mask/blend/QA stages hang (esp. CPU-only sandboxes).
    # 8 MP (~3500x2300) is plenty for casting/marketing delivery.
    max_process_mp: float = 8.0
    ecc_iterations: int = 200
    ecc_epsilon: float = 1e-5
    jpeg_quality: int = 96
    qa: QAThresholds = field(default_factory=QAThresholds)

    def to_dict(self) -> dict:
        return asdict(self)
