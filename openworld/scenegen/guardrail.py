"""Guardrail prompt-rewriting layer: plain instruction -> nanobanana edit prompt.

The nanobanana / Gemini image editor produces far more consistent robot scenes
when the edit instruction is phrased a specific way: framed as an edit to a
robot's top-down wrist-camera view, scoped to *only* the described objects, with
an explicit "keep everything else (and the gripper) the same", and with no
occlusion between objects. End users, however, just type something like
``"put a carrot in the bowl"`` or ``"add a red apple to the left of a plate"``.

This module sits *between* the user's instruction and the nanobanana call and
reshapes the former into the latter. Two backends:

  - ``"gemini"`` (default): ask ``gemini-2.5-flash`` (text) to rewrite the
    instruction following the rules in ``SYSTEM_PROMPT``. Needs ``GOOGLE_API_KEY``
    and the ``google-genai`` package (``uv sync --extra scenegen``). Falls back to
    the template on any error so a missing key never hard-stops a run.
  - ``"template"``: a deterministic string wrapper. No network, fully
    reproducible, but no real understanding of the instruction.

The only public entry point is :func:`build_edit_prompt`.
"""

from __future__ import annotations

import os

# Default model for the LLM-backed guardrail. Text-only Gemini (not the image
# model) — it reuses the same GOOGLE_API_KEY / google-genai SDK as nanobanana.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Rules the guardrail enforces. Mirrors the constraints the redteam task-proposer
# was told were "handled downstream" — here they actually get applied.
SYSTEM_PROMPT = """\
You rewrite a robot task/scene description into a single image-EDIT instruction \
for an image-editing model. The image being edited is a robot's first-person, \
top-down (bird's-eye) wrist-camera view of a tabletop.

Rewrite the user's instruction into ONE concise edit instruction that obeys ALL \
of these rules:
- Start by stating the view: "This is a robot's first-person top-down \
(bird's-eye) wrist-camera view of a tabletop."
- Describe ONLY the objects to place/modify on the table: their color, size, \
count, and where each sits relative to the others. ALWAYS state that each \
object must be rendered from this same overhead top-down viewpoint (its top \
surface seen from directly above, not its side profile), matching the scene's \
perspective, scale, lighting, and shadows so it stays consistent with the rest \
of the image.
- Keep every object fully visible and non-overlapping. No object may hide, \
cover, or partially block another, and none may be hidden by the gripper. Do \
NOT introduce occlusion.
- End with: "Keep everything else the same, and do not change the robot gripper \
or arm."
- Do not mention cameras other than this one, lighting, or photographic style. \
Do not invent objects the user did not ask for.

Output ONLY the rewritten edit instruction as plain text — no preamble, no \
quotes, no markdown, no explanation."""


def template_edit_prompt(instruction: str) -> str:
    """Deterministic fallback wrapper — no LLM, fully reproducible."""
    instruction = instruction.strip().rstrip(".")
    return (
        "This is a robot's first-person top-down (bird's-eye) wrist-camera view "
        "of a tabletop, looking straight down at the table from directly above. "
        "Edit the image to "
        f"{instruction}. "
        "Render any added or modified object from this exact same overhead "
        "top-down viewpoint, as if photographed straight down from above: show "
        "its top surface as seen from directly overhead, not its side profile. "
        "Match the scene's perspective, foreshortening, scale, lighting, and "
        "shadow direction so the object looks consistent with the rest of this "
        "top-down image. "
        "Keep every object fully visible and non-overlapping, keep everything "
        "else the same, and do not change the robot gripper or arm."
    )


def _gemini_edit_prompt(instruction: str, model: str, api_key_env: str) -> str:
    """Rewrite via text Gemini. Raises on any failure (caller decides fallback)."""
    from google import genai  # lazy: only needed for the gemini backend
    from google.genai import types

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=instruction.strip(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty rewrite")
    return text


def build_edit_prompt(
    instruction: str,
    *,
    backend: str = "gemini",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key_env: str = "GOOGLE_API_KEY",
    verbose: bool = True,
) -> str:
    """Turn a plain instruction into a nanobanana-optimized edit prompt.

    Args:
        instruction: the user's plain-language scene/edit instruction.
        backend: ``"gemini"`` (LLM rewrite) or ``"template"`` (deterministic).
        model: Gemini text model id used by the ``"gemini"`` backend.
        api_key_env: env var holding the Gemini API key.
        verbose: print which backend produced the prompt.

    Returns:
        The rewritten edit instruction (always a non-empty string). The
        ``"gemini"`` backend silently falls back to the template on any error.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        raise ValueError("instruction must be a non-empty string")

    if backend == "template":
        prompt = template_edit_prompt(instruction)
        if verbose:
            print(f"[guardrail] template -> {prompt!r}")
        return prompt

    if backend == "gemini":
        try:
            prompt = _gemini_edit_prompt(instruction, model, api_key_env)
            if verbose:
                print(f"[guardrail] gemini({model}) -> {prompt!r}")
            return prompt
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to template
            prompt = template_edit_prompt(instruction)
            if verbose:
                print(
                    f"[guardrail] gemini backend failed ({exc}); "
                    f"falling back to template -> {prompt!r}"
                )
            return prompt

    raise ValueError(f"unknown guardrail backend '{backend}' (expected 'gemini' or 'template')")
