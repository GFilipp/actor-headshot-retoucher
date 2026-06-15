# Actor Headshot Retouch

A retouching tool for actor headshots, body shots, and marketing assets that fixes the usual barbell problem with AI retouching.

- Full AI regeneration looks fake and shifts identity. Casting directors punish that.
- Minimal automated touch-ups miss the real defects or smear them.

This tool takes a third path. A generative model proposes a retouched **target** (what good looks like). Deterministic code then transfers only the validated, local fixes back onto the original high-resolution file. The model never paints the final pixels. Texture, composition, lighting, and likeness stay the original's.

![retouch direction map](examples/retouch-map.png)

*Top: original. Bottom: the retouch directions the pipeline targets — lid discoloration, under-eye brownness, tear-trough shadows, crepey texture, and eye whites. Each is corrected as a subtle, masked, texture-preserving edit, never a regeneration.*

## Why it stays real

The generated target is treated as **direction, not pixels**.

- Colour/tone fixes move only chroma (a*/b* in LAB) and keep the original luminance, so 3D form (the nose, cheekbones) is never flattened. An edge-aware guided filter blends them with no visible patch.
- Marks and blemishes are healed from the original's own surrounding texture, never the generated image. Reddish/pigmented blemishes are caught too, not just dark spots.
- Facial features are protected: brows, eyes, lashes, lips, and nostrils are never edited; tone edits stay inside the face, so ears, hair, and clothing are left alone.
- Under-eye / tear-trough shadow is lifted by a dedicated corrector that runs regardless of what the model proposed, so it is actually addressed.
- Any uniform colour cast the model adds is removed before transfer, so complexion never drifts.

## How it works

```
original ──▶ generate target ──▶ align to original ──▶ frequency-separated
                (OpenAI)            (ECC / ORB)          touch-up map
                                                              │
   versioned output ◀── QA gates ◀── blend onto original ◀── region masks
   + contact sheet      (identity,     (masked tone delta,    (tone vs heal)
   + JSON report         SSIM, ΔE,      heal from original
                         texture)       texture)
```

Each stage is a small, testable module in [`retoucher/`](retoucher).

Every run also writes a before/after contact sheet (full frame plus 100% crops) and a JSON report, so you can judge the result without pixel-peeping.

## Quality gates

Every run is graded automatically. A gate whose optional backend is missing is reported as `skipped`, never silently passed. The verdict is `reject` if any gate fails; the artifact is still written for inspection.

| Gate | Checks | Backend |
| --- | --- | --- |
| `identity` | ArcFace cosine similarity before/after (no identity drift) | InsightFace (optional) |
| `untouched_ssim` | regions that should not change stay near-identical | core |
| `untouched_lpips` | perceptual distance off-edit | LPIPS/torch (optional) |
| `edited_delta_e` | the edit is visible (CIEDE2000) and not a cartoon | core |
| `texture` | high-frequency energy not lost (no plastic skin) | core |
| `protected_features` | brows / eyes / lips unchanged (no feature damage) | MediaPipe |

## Install

Requires **Python 3.12** (MediaPipe ships wheels there; 3.14 does not yet). The core install includes the bundled face parser used for feature protection and the under-eye corrector — no model download at runtime.

```bash
python -m pip install .
```

Optional heavier gates (identity check, perceptual check, RAW input):

```bash
python -m pip install ".[advanced]"
```

For real runs against OpenAI, add the SDK and set a key:

```bash
python -m pip install ".[openai]"
export OPENAI_API_KEY=sk-...
```

## Use

```bash
# Offline smoke (mock generator, no API key, no cost):
retoucher headshot.jpg --dry-run --out-dir out

# Real run against OpenAI:
retoucher headshot.jpg --mode hybrid-map --out-dir out

# Force a fix where you point (a blemish the model missed):
retoucher headshot.jpg --mark 980,1420 --out-dir out

# Batch a whole shoot:
retoucher ./shoot --out-dir ./shoot-retouched
```

Each run writes a versioned image (the original is never overwritten), a before/after contact sheet, and a JSON report with the alignment method and every QA gate.

Python API:

```python
from retoucher import OpenAIGenerator, PipelineConfig
from retoucher.pipeline import retouch_image

res = retouch_image("headshot.jpg", "out", OpenAIGenerator(), PipelineConfig(mode="hybrid-map"))
print(res.qa.verdict, res.report["qa"])
```

## The model is swappable

By default the OpenAI backend **auto-discovers the latest `gpt-image` model at runtime** (via the models API), so a new release like `gpt-image-3` is adopted with no code change. Nothing is hard-pinned. Pin a specific model if you want to:

```bash
export OPENAI_IMAGE_MODEL=gpt-image-3   # or pass --model gpt-image-3
```

It falls back to a known model only if the models API is unreachable. The generator is still the weakest part of the chain for this workflow (it tends to regenerate and recolor), but the pipeline never trusts its pixels globally, so the model matters less than in a regenerate-everything tool. To swap in FLUX.1 Kontext, Gemini, or a local model, implement the one-method `Generator` interface in [`retoucher/generate.py`](retoucher/generate.py); nothing else changes.

## Limitations

- Alignment can fail when the generated target differs a lot in pose or expression. The pipeline falls back (ECC, then ORB), flags low confidence, and the identity gate catches drift.
- The generator must keep the same crop and framing for a clean transfer. The prompts ask for this; a model that recrops will produce weaker results.
- The quality path needs Python 3.12 + MediaPipe (bundled model) for feature protection and the under-eye corrector. Without a face parser the tool degrades to a skin + edge-gated fallback and flags `face_geometry: false` in the report.
- Headless / sandboxed environments (e.g. Codex) where MediaPipe aborts during graphics setup are detected automatically (an isolated subprocess probe) and fall back to that path instead of crashing. Force it with `RETOUCH_FACE_PARSER=off`, or `=on` to skip the probe where you know MediaPipe works.
- Real before/after quality depends on the source photo and the generator. Heals are confined to the face and the area just below it (neck / open-collar chest); for a blemish further down, point at it with `--mark`.
- Privacy: a real run uploads your image to the generator's API (OpenAI by default); review their data policy first. Offline `--dry-run` and the test suite never leave your machine.

## Repository layout

This is both a runnable Python package and a Codex/agent skill.

```
actor-headshot-retouch-skill/   (clone this; the repo itself is the skill)
├── SKILL.md            agent behaviour spec
├── retoucher/          the pipeline (align, diff, mask, blend, qa, cli)
├── scripts/            check_readiness.py preflight
├── references/         retouch standards and prompts
├── tests/              offline test suite (mock generator)
├── examples/           reproducible demo + retouch-map illustration
├── pyproject.toml
└── CHANGELOG.md
```

To use it as a Codex skill, clone or copy this repository into your skills directory so `SKILL.md` sits at the skill's root, then invoke it; the skill runs the same pipeline described here.

## Develop

```bash
python -m pip install -e ".[dev]"
pytest tests -q
```

Tests run fully offline with a mock generator (no API key or network). They bootstrap the package onto the path, so `pytest` and `python examples/make_example.py` work straight from a clone. Requires Python 3.12 (MediaPipe wheels); CI runs on 3.12.
