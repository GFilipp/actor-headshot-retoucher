# Changelog

Versions follow the GitHub releases. The package version (`pyproject.toml`) is
the single source of truth.

## v3.1.0

A fresh-eyes QA sweep (three independent reviewers over code, docs, and repo hygiene)
plus the memorialization of the workflow that actually delivered real photos.

New — the surgical engine (`--engine surgical`):
- `retoucher/surgical.py`: the proven one-region recipe as a tested, runnable path —
  Gemini donor -> register to the original -> LAB color-match to clean face skin ->
  composite ONE organic region (paste/luma/transfer, features protected after feathering)
  -> light polish -> native-resolution audit of the region as a CHECK (operator judges).
  Draws K donors and keeps the audit-cleanest. Graceful refusal without a frontal face.
- CLI: `--region`, `--composite`, `--whites/--discolor/--lines`; always writes the image
  (suffix `surgical`) + a telemetry report. Defaults are the recipe that delivered the
  20260509 shoot.

Fixes:
- VLM bbox sanitization (the ghost-blob bug): normalized 0-1 boxes are scaled, everything
  is clamped to the frame, degenerate boxes are dropped AND flagged in `out_of_scope`; the
  VLM prompt now requests fractional coords. A mask is never built from a bogus box.
- `--max-process-mp` now reaches the v3 engine (was silently ignored on that path).
- Audit perf: per-image arrays (gray/LAB/edit-delta gradient/Laplacians) are computed once
  per (original, retouched) pair and shared across regions — was O(regions x full-image),
  ~13 minutes on a real 26-region map; equivalence-tested (identical verdicts).
- Analyze: when the VLM reports 0 faces but CV geometry found one, CV wins and the
  disagreement is recorded (was silently stored as face_count=0).
- CLI errors now print exception type + traceback; dropped an unused parameter from
  `build_edit_prompt`.

Docs truth-pass:
- README/SKILL/METHODOLOGY now say what is true: the surgical engine is the recommended
  real-photo path; v3 automation is experimental (its one real-photo run over-edited and
  was correctly caught by its own audit); the RetouchMap is sorted highest-severity first
  (the "identity-safe first" claim contradicted the code); the CLI default remains `v2`.
- METHODOLOGY gains "The surgical session" (the recipe, mode selection, knobs).
- `references/retouch_workflows.md` is explicitly marked legacy-v2 and SKILL points at
  METHODOLOGY for current workflow.
- `.gitignore` covers `*.report.json` and `out*/`.

## v3.0.1

Hardening + cleanup after running v3 on a real photo (it over-flagged because the audit
was synthetic-tuned) and a full loose-ends sweep.

Real-photo audit calibration:
- Color gate now measures the color the EDIT introduced (don't add red a* / warmth b* vs the
  original region), not an absolute match to a clean cheek. The cheek-match version flagged
  every naturally-different region (hand, neck, under-eye) and intended de-discoloration.
- Seam gate uses a boundary-vs-interior gradient RATIO (robust to real skin texture); an
  absolute threshold over-flagged real photos.
- Per-region de-discoloration reference: a hand/neck/chest pulls toward its OWN adjacent skin,
  not the face cheek.
- Low-res donor guard: a Gemini donor much lower-res than the working image is texture-unsafe
  to paste/luma (injects upscaling stipple), so those modes downgrade to transfer.
- 8 MP working-resolution cap (mirrors v2) so a 20 MP file no longer grinds for ~15 min; the
  native-res audit invariant is preserved (single pre-edit downscale, audit at working native).
- CLI real run wires the Gemini vision assessor, so the whole photo is inventoried
  (hands/neck/chest/hair), not just the face-derived CV inventory.

Claude-only + loose ends:
- Removed forward-facing Codex references (SKILL/README/METHODOLOGY/RULES/learnings + code
  comments); the GPU-sandbox notes are now environment-general. CHANGELOG history is retained.
- `__init__` now exposes the v3 system (`retouch`, `analyze`, `calibrate`, audit, schema,
  `GeminiGenerator`, configs); it previously exported only the v2 API.
- Renamed the v3 result `orchestrator.RetouchResult` -> `RetouchOutcome` to resolve the name
  collision with the legacy `pipeline.RetouchResult`.
- Removed a dead config field, dead imports/locals; retired the 7 superseded (and partly
  broken) legacy `scripts/` one-offs — the tested package modules are the maintained versions.
  `scripts/check_readiness.py` stays (the v2 CLI preflight uses it).

## v3.0.0

The north-star rebuild: a dynamic, holistic, hybrid, model-agnostic retouch system that
analyzes the whole photo, builds a per-region retouch map, calibrates generative-vs-
deterministic per region, surgically composites, and self-audits at native resolution
before it will deliver. The method this project actually used (Gemini regenerate ->
surgical paste -> deterministic polish) is now a tested system, not uncommitted scripts
driven by eye and shell history. `--engine v2` keeps the legacy deterministic pipeline.

Four typed contracts, each surviving into the JSON telemetry (auditable, replayable):

- **Analyze** (`analyze.py`, `vision.py`, `schema.py`): hybrid CV + VLM whole-photo
  assessment (`PhotoAssessment`) and defect map (`RetouchMap`) across face/eye-area,
  hands, neck, chest, hair. VLM proposes, CV corroborates and guards face-count.
  Unhandleable photos (multi-person, profile, no-face) are flagged, never crashed.
- **Calibrate** (`calibrate.py`): a pure policy function -> `CalibrationRecord` per op
  with a written rationale. Encodes the rulebook (small/low-res face never raw-pastes;
  pigment is generative-led; mild unevenness is deterministic-only; hair is generative-
  only; identity-sensitive caps the generative share). `escalate()` does audit-driven
  re-calibration. Thresholds in `config.CalibrationConfig`.
- **Self-audit** (`audit.py`): native-resolution detectors (seam/box, blur/plastic,
  stipple, color cast, residual mark, faded lashes) with a per-region annulus texture
  baseline. Coverage equals the map; identity is REQUIRED (ArcFace or a defined SSIM
  fallback, never skipped-as-clean). The nearest-neighbor invariant is enforced by a
  runtime guard and a source-scan test. Thresholds in `config.AuditThresholds`.
- **Orchestrator** (`orchestrator.py`): the spine (ingest -> analyze -> map -> calibrate
  -> execute -> audit -> deliver) with audit-driven sampling and audit-gated delivery
  (refuse rather than ship least-bad).

Also: `regions.py` (surgical compositor: register + organic masks + paste/transfer/luma +
color-match), `cleanup.py`, `detect.py`, `crop.py`, `GeminiGenerator` + `edit_n` behind
the `Generator` protocol with `router.py`, a single-source subject-agnostic persona
(`prompts.build_edit_prompt`), `--engine v3` / `--samples` / `--max-escalate` in the CLI,
and the docs of record (`METHODOLOGY.md`, `RULES.md`, `references/retouch_learnings.md`).
The photo-3 hand (blur + missed mark) and photo-2 glasses shadow are committed as a
synthetic golden regression set the audit must flag.

## v2.1.4

Stabilizes the proven delivery ops + the Gemini surgical-paste toolkit out of the
uncommitted working tree (the method actually used to retouch the 20260509 shoot),
ahead of the v3 north-star rebuild that promotes them into a tested system.

- `blend.py`: `smooth_under_eye_texture` (high-frequency attenuation), `even_skin_tone`
  (a*/b* evening, L kept so form/texture survive), `whiten_eye_whites` (sclera de-red/
  yellow + brighten), `reduce_discoloration` (pull red/brown toward a clean skin
  reference, reduce excess only).
- `faceparse.py`: `landmarks()` returns the raw 478-pt mesh, for registering a donor/
  edit back onto the original (surgical paste outside the deterministic pipeline).
- `config.py` / `cli.py`: `under_eye_texture_strength`, `skin_even_strength` +
  `--under-eye-texture` / `--skin-even`.
- `scripts/`: the Gemini surgical-paste toolkit — `gemini_retouch.py`, `surgical_paste.py`
  (paste/transfer/luma modes, organic rounded masks, colour-match, ECC for non-face
  regions), `polish_eyes.py`, `face_crop.py`, `detect_blemishes.py`. Ad-hoc one-offs;
  promoted into the package + tested in v3.0.0.

## v2.1.3

Fixes a crash and a hang seen running the skill inside a sandbox (Codex):

- **Crash:** MediaPipe aborted natively initializing its Metal/GPU helper where there's no GPU. Now forces `MEDIAPIPE_DISABLE_GPU=1` before MediaPipe loads, so it runs on CPU (verified) and avoids the abort; the subprocess probe remains the fallback.
- **Hang:** full-resolution originals (20MP+) made the full-image align/diff/mask/blend/QA stages run for minutes. Processing/output is now capped at `max_process_mp` (default 8 MP, `--max-process-mp` to change), and the blemish-component labeling is vectorized (no per-component Python loop). Note: this means very large originals are downsampled to the cap.

## v2.1.2

Robustness pass from a full bug audit:

- CLI no longer dumps a traceback on bad `--mark` / `--mark-box` / `--strength`; invalid input is a clean error. An image-less directory reports clearly instead of silently "succeeding".
- RAW input without the optional `rawpy` gives a clear install hint instead of `ModuleNotFoundError`, and one bad file in a batch is skipped instead of aborting the whole run.
- EXIF orientation is honored on load (no more sideways processing / missed faces on phone shots).
- `guided_radius` now scales with resolution like the other spatial params (correct edge-aware blend at non-reference sizes).
- The MediaPipe probe logs *why* it fell back, so a silent quality drop in a sandbox is debuggable.
- Tiny-crop border suppression can no longer zero the whole blemish map. Added robustness tests (probe/degraded-path/CLI-errors/RAW) — the class of gap that let the sandbox crash through.

## v2.1.1

- Headless / sandbox safety. MediaPipe can **abort natively** (not a catchable Python exception) during graphics setup in some sandboxes, e.g. Codex, which crashed the whole process. The face parser is now validated in an isolated subprocess before any in-process use; if it would crash, the pipeline silently falls back to the no-geometry path instead of dying. Override with `RETOUCH_FACE_PARSER=off` (force fallback) or `on` (trust it, skip the probe).

## v2.1.0

Fixes from real-headshot A/B testing: feature protection, form-preserving colour, and corrections that no longer depend on the model.

- **Feature protection (new `faceparse.py`, bundled MediaPipe model).** Brows, eyes/lashes, lips, and nostrils are never edited; tone edits are confined to the eroded face oval (ears/hair/neck excluded) and heals to skin near the face (skin-coloured clothing is left alone). New QA gate fails if protected features change.
- **Form-preserving tone (chroma-only).** Colour correction moves only a*/b* in LAB and keeps the original luminance, so facial contour (e.g., the nose) is no longer flattened; an edge-aware guided filter removes visible patch edges.
- **Model-independent corrections.** Reddish/pigmented blemishes are detected by a LAB a* anomaly (not just darkness) and healed, and a dedicated under-eye / tear-trough corrector lifts shadow regardless of what the model proposed.
- **Forced fixes.** `--mark x,y[,r]` and `--mark-box` force a fix at a location the user points to.
- **Robustness.** The OpenAI backend fails fast with a clear message when the SDK or `OPENAI_API_KEY` is missing. Pinned to Python 3.12 (MediaPipe wheels); `opencv-contrib-python-headless` for the guided filter.

## v2.0.0

Major rebuild from an instruction-only skill into a deterministic retouch pipeline.

- The image model now only proposes a retouch *target*; deterministic code transfers validated, local fixes back onto the original: align (ECC / ORB), frequency-separated touch-up map, region masks, blend with the original's own texture, heal marks from surrounding pixels, neutralize any global colour cast.
- Automated QA gates: identity (optional), untouched-region SSIM / LPIPS, edit visibility (CIEDE2000), and texture-loss.
- The OpenAI model is auto-discovered at runtime; nothing is hard-pinned. Pin with `$OPENAI_IMAGE_MODEL` or `--model`.
- Spatial parameters scale with image resolution, so behaviour is consistent from 512px to 4096px.
- Adds a CLI, batch processing, packaging, an offline test suite, CI, and a reproducible example.

Breaking: behaviour and outputs differ from the v1.x prose workflow. The `retoucher` package and CLI are new.

## v1.2.0

Added hybrid-map as the explicit default workflow and readiness support for it (prose skill).

## v1.1.0

Added the minimum-viable-edit threshold and before/after QA expectations (prose skill).

## v1.0.0

Initial actor headshot retouch skill: prose workflow plus the readiness checker.
