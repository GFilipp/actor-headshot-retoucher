"""P3: the Analyze contract — handleable single face, honest refusal otherwise,
and a JSON-serializable assessment/map. Offline via MockAssessor."""
from __future__ import annotations

import json

import numpy as np

from retoucher.analyze import analyze

from _synth import fake_geometry, make_original


class _StubAssessor:
    def __init__(self, payload):
        self.payload = payload

    def assess(self, rgb):
        return self.payload


def test_analyze_single_face_is_handleable_with_map():
    # Inject geometry: the synthetic test image isn't a real MediaPipe-detectable face.
    a, m, geom = analyze(make_original(), geom=fake_geometry())
    assert geom is not None
    assert a.handleable is True and a.face_count == 1
    assert a.shot_type in ("headshot", "three_quarter", "bodyshot")
    assert len(m.ops) > 0                                   # found things to fix
    assert any(o.region == "eye_area" for o in m.ops)
    assert any(s.kind == "face" for s in a.subjects)
    json.dumps(a.to_dict()); json.dumps(m.to_dict())        # report-serializable


def test_analyze_no_face_refuses_not_crashes():
    blank = np.full((400, 400, 3), 0.5, np.float32)
    a, m, geom = analyze(blank)
    assert geom is None
    assert a.handleable is False and "no clear frontal face" in a.reason
    assert "no-face/occluded/profile" in a.out_of_scope


def test_analyze_multi_face_flagged_out_of_scope():
    stub = _StubAssessor({"shot_type": "headshot", "lighting": "soft",
                          "face_count": 2, "defects": []})
    a, m, geom = analyze(make_original(), assessor=stub, geom=fake_geometry())
    assert a.handleable is False
    assert "multi-person" in a.out_of_scope and "2 faces" in a.reason


def test_analyze_out_of_scope_region_flagged_not_mapped():
    stub = _StubAssessor({"shot_type": "headshot", "lighting": "soft", "face_count": 1,
                          "defects": [{"region": "background", "defect": "blemish",
                                       "severity": 0.5, "bbox": [0, 0, 10, 10]}]})
    a, m, geom = analyze(make_original(), assessor=stub, geom=fake_geometry())
    assert "background" in a.out_of_scope
    assert all(o.region != "background" for o in m.ops)     # not silently treated


def test_vlm_bboxes_sanitized_scaled_clamped_or_dropped():
    # The ghost-blob bug: VLM boxes must never build a mask in the wrong place.
    img = make_original()
    h, w = img.shape[:2]
    stub = _StubAssessor({"shot_type": "headshot", "lighting": "soft", "face_count": 1,
                          "defects": [
                              {"region": "face", "defect": "blemish", "severity": 0.9,
                               "bbox": [0.25, 0.25, 0.5, 0.5]},          # normalized 0-1
                              {"region": "face", "defect": "blemish", "severity": 0.8,
                               "bbox": [-50, -50, 99999, 99999]},        # out of bounds
                              {"region": "hands", "defect": "pigmentation", "severity": 0.7,
                               "bbox": [10, 10, 10, 10]},                # degenerate
                          ]})
    a, m, geom = analyze(img, assessor=stub, geom=fake_geometry())
    by_id = {o.op_id: o for o in m.ops}
    assert by_id["op0"].bbox == (int(0.25 * w), int(0.25 * h), int(0.5 * w), int(0.5 * h))
    assert by_id["op1"].bbox == (0, 0, w - 1, h - 1)
    assert "op2" not in by_id                                # dropped, not silently masked
    assert "hands: invalid bbox" in a.out_of_scope           # ...and flagged


def test_vlm_zero_faces_cv_geometry_wins_with_recorded_disagreement():
    stub = _StubAssessor({"shot_type": "headshot", "lighting": "soft",
                          "face_count": 0, "defects": []})
    a, m, geom = analyze(make_original(), assessor=stub, geom=fake_geometry())
    assert a.handleable is True and a.face_count == 1        # masks come from CV geometry
    assert "CV wins" in a.reason                             # conflict recorded, not silent


def test_vlm_parse_failure_refuses_not_delivers():
    # face_count=-1 is the assessor's parse-failure sentinel; it must NOT be treated as a
    # clean "nothing to fix" (which would deliver an untouched original as success).
    stub = _StubAssessor({"shot_type": "unknown", "lighting": "unknown",
                          "face_count": -1, "defects": []})
    a, m, geom = analyze(make_original(), assessor=stub, geom=fake_geometry())
    assert a.handleable is False
    assert "assessment-failed" in a.out_of_scope and "failed to parse" in a.reason
