"""Retouch prompts, relocated from references/retouch_workflows.md.

These are the source of truth that the generator imports. The markdown
reference now points here so the wording is versioned in code and cannot drift
from what the pipeline actually sends.
"""
from __future__ import annotations

LIGHT_RETOUCH = """\
Perform a conservative, non-regenerative retouch on this actor headshot.

Use the original image as the only canvas. Do not regenerate, replace, or invent facial features. Preserve identity, pose, expression, crop, hair shape, clothing texture, arms, hands, background, and lighting.

Inspect the image like a professional airbrusher and create an internal defect map. Prioritize tired eyes, under-eye fine lines and scaling, discoloration around the eyes, beige/off-color eye whites, small skin marks, neck discoloration, thumb/hand discoloration, and distracting flyaway hair.

Make changes noticeable enough to improve actor marketing value, but keep them natural. Preserve real skin texture, stubble, brows, lashes, moles, and natural facial structure.

Before finishing, compare before/after crops for both eyes, neck, thumb/hand, and full frame. Do not present the result if the marked issues are still obvious.
"""

HYBRID_MAP = """\
Create a high-end actor headshot retouch map from this clean original.

Preserve the actor's exact identity, face shape, eye shape, expression, crop, pose, hair character, hands, wardrobe, background, and studio lighting. Do not create a new portrait.

Show the desired retouch direction clearly enough to guide final work on the original full-resolution file: healthier eyes, cleaner whites, reduced red/brown eyelid discoloration, smoother under-eye scaling and fine lines including the inner corner / tear trough right next to the nose, cleaner small skin marks on the face AND visible chest/neck, improved neck/thumb/hand discoloration when present, and only distracting flyaway reduction.

Keep real pores, stubble, asymmetry, fabric texture, and masculine character. Avoid waxy skin, broad cheek bleaching, painted eye whites, changed facial proportions, or AI-looking softness.
"""

LIGHT_REGEN = """\
Edit the clean unannotated actor headshot as the target image.

Use the annotated image only as a retouch map showing problem areas. Remove all annotation marks completely. Do not include black marker lines or any visible markup.

Preserve the actor's exact identity, facial structure, expression, pose, crop, camera angle, hairstyle, brows, lashes, stubble, moles, sweater, arms, hand shape, background, and original studio lighting. Do not create a new portrait. Do not beautify globally. This is a high-end actor headshot retouch, not a face replacement.

Retouch only the real photographic issues:
- reduce under-eye crepey texture, scaling, fine lines, puffiness, and dark brown/purple discoloration on both eyes, especially the inner corner / tear trough right next to the nose
- clean small blemishes and marks on visible chest and neck skin, not just the face
- brighten and neutralize the whites of the eyes while keeping them realistic
- reduce eyelid discoloration without changing eye shape
- clean small forehead, bridge, cheek, and skin marks
- reduce gray/brown/purple neck discoloration and harsh neck crease shadows
- reduce thumb/hand discoloration and dry crease texture
- tame only distracting flyaway hair
- keep natural pores, skin texture, stubble, and masculine character

The result should look like a world-class commercial actor/model retouch: rested, clean, polished, healthy, and marketable, while still unmistakably the same person and photograph.
"""

PROMPTS = {
    "light-retouch": LIGHT_RETOUCH,
    "hybrid-map": HYBRID_MAP,
    "light-regen": LIGHT_REGEN,
}


def prompt_for(mode: str) -> str:
    """Return the prompt text for a mode, defaulting to the hybrid-map map prompt."""
    return PROMPTS.get(mode, HYBRID_MAP)


# --- v3 single-source persona + dynamic, subject-agnostic edit prompt --------------
# The prompts above are subject-specific (legacy, "masculine character"). The v3 system
# builds its generation prompt from the actual RetouchMap so it is per-photo and never
# baked to one person or gender. This is the ONE place the persona lives.

RETOUCH_PERSONA = (
    "You are a best-in-class film-industry retoucher preparing an actor's photo for "
    "casting. Work like a high-end commercial retoucher: the result should look rested, "
    "clean, polished, and marketable, while remaining unmistakably the same person and "
    "the same photograph."
)

_IDENTITY_GUARD = (
    "Preserve exact identity, face shape, bone structure, features, expression, pose, "
    "crop, camera angle, hair character, wardrobe, background, and the original lighting. "
    "Keep real pores, skin texture, stubble, asymmetry, moles, brows, and lashes. Do NOT "
    "create a new portrait, beautify globally, reshape features, or add AI-looking softness."
)

_DEFECT_PHRASING = {
    "under_eye": "reduce under-eye crepey texture, scaling, fine lines and puffiness, "
                 "including the inner-corner tear trough right next to the nose",
    "crepe": "smooth crepey skin texture while keeping real pores",
    "discoloration": "neutralize red/brown/purple discoloration",
    "pigmentation": "even out pigmentation and blotches toward the surrounding clean skin",
    "blemish": "clean small blemishes and skin marks",
    "skin_unevenness": "even skin tone without flattening form or texture",
    "eye_white_cast": "brighten and neutralize the whites of the eyes, keeping them realistic",
    "flyaway": "tame only distracting flyaway / stray hairs",
}

_REGION_LABEL = {
    "eye_area": "eyes", "face": "face", "neck": "neck", "chest": "chest",
    "hands": "hands / fingers", "hair": "hair",
}


def build_edit_prompt(assessment, retouch_map) -> str:
    """Dynamic, subject-agnostic generation prompt: persona + identity guard + ONLY the
    defects the analyze stage actually mapped (grouped by region). Per-photo, never baked
    to a specific subject."""
    lines: list[str] = []
    for op in retouch_map.ops:
        phr = _DEFECT_PHRASING.get(op.defect)
        if not phr:
            continue
        line = f"- on the {_REGION_LABEL.get(op.region, op.region)}: {phr}"
        if line not in lines:
            lines.append(line)
    if not lines:
        lines = ["- perform a light, conservative clean-up of visible skin only"]
    return (
        f"{RETOUCH_PERSONA}\n\n{_IDENTITY_GUARD}\n\n"
        "Retouch only these real photographic issues:\n" + "\n".join(lines) + "\n\n"
        "Make the changes noticeable enough to improve casting value but natural. Before "
        "finishing, compare before/after at 100%: do not present the result if any listed "
        "issue is still obvious."
    )
