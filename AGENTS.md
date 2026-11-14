# AGENTS.md

Start here if you are an AI coding agent. Codex reads this file directly; Claude Code loads it through `CLAUDE.md`. Humans can start at `README.md`.

## What this is

A retoucher for actor and model headshots. Gemini regenerates the photo as a "donor". Deterministic Python then registers the donor to the full-resolution original, composites only the one region that improved, polishes lightly, and audits the result at native resolution. Texture, likeness and lighting stay real.

## Read first

1. `README.md`: engines, install, CLI.
2. `METHODOLOGY.md`: the system, its invariants, and Known limitations.
3. `RULES.md`: the pre-delivery checklist.
4. `references/retouch_learnings.md`: every artifact the project shipped and the rule that now prevents it.
5. `CHANGELOG.md`: what changed in each release.

## Setup

Python 3.12 only. MediaPipe has no wheels for newer versions.

```bash
python3.12 -m venv .venv312
.venv312/bin/pip install -e ".[dev,gemini]"
```

Gemini key: set `$GEMINI_API_KEY` (or `$GOOGLE_API_KEY`). Without it, the code reads `~/Desktop/gemini.txt`. Keys never go inside this repo or into a commit; `.gitignore` blocks `.secrets/`, `*.key` and `gemini.txt` as a backstop.

## Run

```bash
# Real photo, recommended engine (draws 2 donors, keeps the audit-cleanest):
.venv312/bin/python -m retoucher INPUT --engine surgical --samples 2 --out-dir out

# Offline smoke (mock generator, no key, no cost):
.venv312/bin/python -m retoucher INPUT --engine surgical --dry-run --out-dir out
```

- `surgical`: recommended for real photos. The operator judges the output.
- `v3`: experimental whole-photo automation. Never run it unattended.
- `v2`: legacy pipeline, still the CLI default for back-compat.

Real photos need a local machine: MediaPipe face detection can abort in headless or GPU-less sandboxes. Cloud sandboxes can still run the tests.

## Test

```bash
.venv312/bin/python -m pytest tests -q
```

Fully offline (mock generator, synthetic geometry). CI runs the same suite on Python 3.12 for every push and pull request (`.github/workflows/ci.yml`).

## Rules

- This repo is public. Never commit real photos, outputs or keys. `inputs/`, `out*/` and `*.report.json` are gitignored; the owner supplies input photos from outside the repo.
- Gemini image model IDs are a hand-maintained, newest-first list in `retoucher/generate.py`, because the key type blocks model listing. Refresh it when Google retires a model. OpenAI models are auto-discovered.
- Hand and body-skin discoloration can only be reduced, not erased, without artifacts (`METHODOLOGY.md`, Known limitations). Don't re-grind it.
- Ship changes as: branch, pull request with CI green, merge. Then bump `pyproject.toml` and `CHANGELOG.md`, tag `vX.Y.Z`, and publish a GitHub release.
