# Actor Headshot Retouch Skill

This is a Codex skill for actor, model, casting, agency, and commercial headshot retouching.

It is built for two workflows:

- **Light retouching:** for photos that are already strong and only need subtle polish.
- **Light regen:** for photos that need visible rescue work around tired eyes, under-eyes, discoloration, eye whites, neck, hands, or skin fatigue.

The skill starts with a readiness checklist before doing any edit. It is designed to preserve identity and avoid fake, AI-looking results.

## What To Install

Copy the entire folder named:

```text
actor-headshot-retouch
```

into your Codex skills folder.

Do not copy only `SKILL.md`. The whole folder matters because it includes the readiness checker and workflow guide.

## Mac / Linux Install

Copy and paste this into Terminal from the folder where you downloaded this repo:

```bash
mkdir -p ~/.codex/skills
cp -R actor-headshot-retouch ~/.codex/skills/
```

Then restart Codex or open a new Codex thread.

Use it by typing:

```text
Use $actor-headshot-retouch on this headshot.
```

## Windows Install

Copy and paste this into PowerShell from the folder where you downloaded this repo:

```powershell
New-Item -ItemType Directory -Force $env:USERPROFILE\.codex\skills
Copy-Item -Recurse actor-headshot-retouch $env:USERPROFILE\.codex\skills\
```

Then restart Codex or open a new Codex thread.

Use it by typing:

```text
Use $actor-headshot-retouch on this headshot.
```

## Optional Photo Tools

For local, non-regenerative retouching, install:

- Python 3.12
- ImageMagick
- libvips
- ExifTool

Then install the Python image packages:

```bash
python -m venv photo-retouch
python -m pip install --upgrade pip wheel setuptools
python -m pip install numpy pillow opencv-python scikit-image mediapipe rawpy pyvips
```

You do **not** need PyYAML to use this skill.

## What The Skill Checks

Before editing, the skill checks:

- Python imaging stack
- ImageMagick
- libvips
- ExifTool
- whether image generation is available for light regen
- whether the source image can be read
- whether the output folder can be written

If something required is missing, the skill stops before editing.
