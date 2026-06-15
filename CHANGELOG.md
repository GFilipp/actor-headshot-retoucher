# Changelog

Versions follow the GitHub releases. The package version (`pyproject.toml`) is
the single source of truth.

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
