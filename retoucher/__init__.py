"""Actor headshot retouch: a dynamic, holistic, hybrid, self-auditing system.

The v3 engine analyzes the whole photo, builds a per-region retouch map, calibrates
generative-vs-deterministic per region, composites surgically, and self-audits at native
resolution before it will deliver. See `orchestrator.retouch` (and METHODOLOGY.md).

The legacy v2 deterministic-transfer pipeline (`pipeline.retouch_image`) is still here for
backward compatibility; its result type is `pipeline.RetouchResult`.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# v3 system (primary)
from .analyze import analyze
from .audit import audit_map, audit_region, identity_gate
from .calibrate import calibrate, escalate
from .config import AuditThresholds, CalibrationConfig, PipelineConfig, QAThresholds
from .generate import (
    GeminiGenerator, Generator, MockGenerator, OpenAIGenerator, get_generator,
)
from .orchestrator import RetouchOutcome, retouch
# v2 legacy pipeline
from .pipeline import retouch_image, retouch_path
from .schema import (
    CalibrationRecord, PhotoAssessment, RegionVerdict, RetouchMap, RetouchOp,
)
from .surgical import SurgicalResult, surgical_retouch

# Single source of truth is pyproject.toml; read it from installed metadata.
try:
    __version__ = _pkg_version("actor-headshot-retoucher")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+source"

__all__ = [
    # surgical engine (the proven real-photo recipe)
    "surgical_retouch",
    "SurgicalResult",
    # v3 system
    "retouch",
    "RetouchOutcome",
    "analyze",
    "calibrate",
    "escalate",
    "audit_map",
    "audit_region",
    "identity_gate",
    "PhotoAssessment",
    "RetouchMap",
    "RetouchOp",
    "CalibrationRecord",
    "RegionVerdict",
    # generators + config
    "Generator",
    "MockGenerator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "get_generator",
    "PipelineConfig",
    "CalibrationConfig",
    "AuditThresholds",
    "QAThresholds",
    # v2 legacy pipeline
    "retouch_image",
    "retouch_path",
]
