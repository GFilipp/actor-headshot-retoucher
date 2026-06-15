"""P6: the orchestrator spine, end to end and fully offline (MockAssessor + MockGenerator).
Asserts coverage == map, complete telemetry, audit-gated delivery, and graceful refusal."""
from __future__ import annotations

import json

import numpy as np

from retoucher.generate import MockGenerator
from retoucher.orchestrator import RetouchResult, retouch

from _synth import fake_geometry, make_original


class _StubAssessor:
    def __init__(self, payload):
        self.payload = payload

    def assess(self, rgb):
        return self.payload


def test_dry_run_offline_full_spine_and_telemetry():
    rgb = make_original()
    res = retouch(rgb, generator=MockGenerator(), geom=fake_geometry(), samples=2, max_escalate=1)
    assert isinstance(res, RetouchResult)
    assert res.image.shape == rgb.shape                       # same canvas, no resize
    assert len(res.verdicts) == len(res.retouch_map.ops) > 0  # coverage == map
    for k in ("assessment", "map", "calibrations", "selected_sample", "sample_scores",
              "calibrations_final", "escalations", "verdicts", "identity", "delivered"):
        assert k in res.report, k
    assert res.report["samples"] == 2                         # audit-driven sampling honored
    assert res.identity["status"] in ("pass", "fail")         # identity never skipped-as-clean
    assert isinstance(res.delivered, bool)
    json.dumps(res.report)                                    # fully serializable telemetry


def test_unhandleable_photo_refuses_without_crashing():
    blank = np.full((300, 300, 3), 0.5, np.float32)           # no detectable face
    res = retouch(blank, generator=MockGenerator())
    assert res.delivered is False and not res.assessment.handleable
    assert res.image.shape == blank.shape                     # returns the original untouched
    assert res.verdicts == []


def test_deterministic_only_path_runs_without_a_generator():
    # eye-white cast is deterministic-only -> the spine must run with generator=None.
    stub = _StubAssessor({"shot_type": "headshot", "lighting": "soft", "face_count": 1,
                          "defects": [{"region": "eye_area", "defect": "eye_white_cast",
                                       "severity": 0.5, "bbox": [40, 40, 90, 60]}]})
    res = retouch(make_original(), generator=None, assessor=stub, geom=fake_geometry())
    assert res.report["samples"] == 1                         # no generative donor needed
    assert len(res.verdicts) == 1
    assert res.calibrations[0].composite_mode == "none"
