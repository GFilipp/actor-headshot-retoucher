# Methodology: the v3 north-star retouch system

This is the system of record. If a script, prompt, or habit contradicts this file, this
file wins. It exists because the project kept relearning the same lessons and shipping the
same artifacts; the rules below are now code, not memory.

## North star

For an arbitrary actor photo, a best-in-class film-retoucher persona produces a
casting-grade, identity-true retouch and a trustworthy clean/not-clean verdict, with no
human pixel-peeping required. Model-agnostic, holistic (the whole picture, not just the
face), and calibrated per photo.

That is the goal the v3 automation builds toward, not a claim it has earned: on its one
real-photo run it over-edited (many small auto-composites) and was correctly caught by its
own audit. Today the supported real-photo path is **the surgical session** (below) — the
recipe that actually delivered the 20260509 shoot — with the audit as the check.

## The four contracts

Each is a typed object that survives into the JSON telemetry report, so every run is
auditable and replayable.

1. **Analyze** (`retoucher/analyze.py`, `vision.py`, `schema.py`). Runs first, before any
   pixel moves. Produces a `PhotoAssessment`: shot type, face size (`face_px_frac`),
   resolution class, lighting, subjects, per-region clean-skin references, and a
   `RetouchMap` of defects with severity. Hybrid: a VLM proposes the whole-photo
   inventory; local CV corroborates and supplies geometry and the face-count guard. If the
   photo cannot be handled, the contract is refuse-and-flag, never crash.
2. **RetouchMap** (`schema.py`). An ordered list of `RetouchOp(region, defect, severity,
   bbox, identity_sensitive)` spanning every in-scope region, sorted highest-severity
   first (identity sensitivity is enforced downstream by Calibrate, which caps the
   generative share on identity-sensitive ops).
3. **Calibrate** (`retoucher/calibrate.py`). Per op, a decided `CalibrationRecord`: the
   generative-vs-deterministic split, composite mode, mask shape, feather, strengths, and a
   written rationale. It is a pure policy function of the defect, severity, face size,
   resolution, and identity sensitivity; not a fixed recipe.
4. **Verdict** (`retoucher/audit.py`). A native-resolution self-audit whose coverage equals
   the map. Only clean regions ship; a failing region is escalated (bounded) or flagged.

## The decision tree (the rulebook)

Encoded in `calibrate.py`; thresholds in `config.CalibrationConfig`.

| Defect | Treatment | Why |
|---|---|---|
| Stray / flyaway hair | Generative only (paste) | Deterministic code cannot remove a hair. |
| Pigmentation / discoloration | Generative-led + deterministic de-discolor toward clean skin | Pigment is chromatic; deterministic alone barely dents it. |
| Under-eye / crepe | Generative carries the texture fix + deterministic smooth and de-discolor | The regenerated smooth region is what actually fixes bags/crepe. |
| Eye-white cast | Deterministic only (sclera de-cast) | Never run generative near the eyeball. |
| Mild skin unevenness | Deterministic only (even a*/b*, keep form) | A field regenerate is overkill and risks plastic skin. |
| Isolated blemish | Targeted deterministic heal | A small spot does not need a regenerate. |
| Small / low-resolution face | Never a raw paste; lighter luma | Pasting low-res texture distorts at pixel zoom. |
| Identity-sensitive op | Cap the generative share; downgrade paste to luma | Generative must not restructure features. |

Composite modes (`regions.composite_region`):
- `paste`: raw donor pixels. Carries texture and color; can box on sharp edges and drag the
  donor's color. Use on large, high-resolution faces, always color-matched to clean skin.
- `transfer`: low-frequency tone delta only. Seamless; keeps the original texture. Use when
  a paste would smear (the blur escalation switches to this).
- `luma`: donor luminance, original chroma kept. No color distortion. The default for small
  or low-resolution faces.

## Non-negotiable invariants

These are the anti-corner-cutting gates. Each has a test that fails if it regresses.

- **The audit runs at nearest-neighbor native resolution.** Verifying on interpolated zoom
  is the exact illusion that hid every artifact. `audit._assert_native` rejects a shape
  mismatch at runtime; a source-scan test bans interpolating resize in `audit.py`.
- **Audit coverage equals map coverage.** Every mapped region is audited. A region that
  cannot be checked is reported skipped-and-not-clean, never a silent pass.
- **Audit-gated delivery.** Draw K candidates, audit each at native res, ship the cleanest;
  if none pass, refuse and report. Do not ship the least-bad.
- **Identity is required for delivery.** ArcFace when InsightFace is present, else a defined
  SSIM fallback. Never reported skipped-and-assumed-clean.
- **Per-region clean-skin reference and texture baseline.** A hand's clean tone is not the
  face's; texture thresholds come from the region's own adjacent-skin annulus.
- **Decision telemetry.** The JSON report logs the assessment, the full map, the calibration
  and rationale per region, candidate scores, the selected sample, escalations, and the
  per-region verdict.
- **Single-source persona.** `prompts.build_edit_prompt` is the one place the persona lives;
  the defect list is built from the map, so the prompt is subject-agnostic.
- **Model-agnostic proposer.** Gemini is one adapter behind the `generate.Generator`
  protocol, selected by `router.py`. Swappable, mockable.

## The spine

`orchestrator.retouch()` runs: ingest -> analyze -> map -> calibrate -> execute -> audit ->
deliver. Execute is per region: propose a generative donor (once, K samples), register it
to the original, composite the calibrated region, then apply deterministic follow-ups. The
audit picks the cleanest candidate; a failing region is escalated and re-executed, bounded.

## Scope

In: face and eye-area, hands and fingers, neck and chest, stray/flyaway hair.
Out (flagged in the report, handled gracefully, never crashed): multi-person, profile and
three-quarter angles, heavy occlusion, background, other body skin.

## Known limitations

- **Hand / body-skin discoloration: reduce, not erase.** A strong discoloration on a hand,
  next to the nail and the silhouette, cannot be fully erased without artifacts — generated
  pixels flatten the skin texture and rewrite the nail, and pasting across the silhouette
  halos. The tool ships an artifact-free reduction (frequency split: generated tone + original
  texture); erasing to zero is out of scope and reads as fake. See
  `references/retouch_learnings.md` §11.
- **Defect location on hands is unreliable.** Detectors grab sweater fabric and hair, and a
  chroma threshold grabs the whole warm hand. Point at the spot (the blemish workflow) rather
  than relying on auto-detection. The face / eye-area locates reliably via landmarks; the hand
  does not.

## The surgical session (the supported real-photo path)

One targeted pass, operator in the loop — the recipe that delivered the 20260509 shoot,
runnable as `--engine surgical` (`retoucher/surgical.py`):

1. **Donor.** Gemini regenerates the photo (K samples — the model is stochastic). Flaws in
   regions you are not harvesting are irrelevant.
2. **Register + color-match.** Landmark-affine (or ECC) registration onto the original,
   then LAB color-match to clean face skin so the donor's cast (rouge) never enters.
3. **Composite ONE region.** An organic rounded mask (`periorbital` by default), wide
   feather, features re-protected AFTER feathering. Mode per the photo:
   - `paste` — the donor's pixels; erases texture defects (crepe, bags). Large/high-res
     faces only; never across a silhouette edge; never over nails or features.
   - `luma` — the donor's luminance, the original's color. Crepe fix without color risk.
   - `transfer` — low-frequency tone only, original texture kept. Small/low-res faces.
4. **Polish.** Light deterministic finish: eye whites, de-discoloration toward clean
   reference skin, residual fine lines (`--whites/--discolor/--lines`).
5. **Audit as a check.** The region is audited at nearest-neighbor native resolution
   (seam/texture/color/residual/lashes); the cleanest of the K donors is kept, the
   verdict is reported, and the operator judges the image. Refine, or accept.

```bash
# Real run (Gemini donor; reads ~/Desktop/gemini.txt or $GEMINI_API_KEY):
.venv312/bin/python -m retoucher INPUT --engine surgical --samples 2 --out-dir out
# knobs: --region periorbital|under_eye|face   --composite paste|luma|transfer
#        --whites/--discolor/--lines            defaults = the proven recipe
```

Hands, neck, chest: work from a tight crop so the donor is sharp at that scale, keep masks
inside the silhouette, and expect reduction rather than erasure (see Known limitations).

## Running it

```bash
# Offline, no API, no cost (mock generator + mock assessor):
.venv312/bin/python -m retoucher INPUT --engine surgical --dry-run --out-dir out

# Recommended real run — the surgical session:
.venv312/bin/python -m retoucher INPUT --engine surgical --samples 2 --out-dir out

# Experimental whole-photo automation (audit-gated; not yet trusted unattended):
.venv312/bin/python -m retoucher INPUT --engine v3 --samples 3 --out-dir out
```

Run locally, not in a sandboxed environment (it denies the GPU and MediaPipe aborts).
See `RULES.md` for the pre-delivery checklist and `references/retouch_learnings.md` for the
failure log that produced these rules.
