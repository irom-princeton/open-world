"""scenegen: build world-model test cases from a language instruction + image.

Given a plain-language task instruction and an initial (wrist-camera) image,
this package produces an Initialization *suite* — the universal handoff format
consumed by ``scripts/generate_videos.py`` / ``scripts/run_evaluation.py`` — by
chaining three stages:

  1. guardrail  -- rewrite the user's plain instruction into the prompt format
     the nanobanana (Gemini 2.5 Flash Image) editor understands best
     (``openworld.scenegen.guardrail``).
  2. scene edit + multiview  -- spawn the bundled diffusers pipeline
     (``external/diffusers/.../multiview_droid_with_nanobanana.py``): nanobanana
     edits the wrist view, then FLUX.2-klein completes the two side views.
  3. suite assembly  -- resize the three views to the world-model resolution and
     write one ``initialization.yaml`` per case, cloning the robot start state
     from a template (``openworld.scenegen.pipeline``).

This is the *directed* counterpart to the autonomous ``openworld.redteam`` loop:
you supply the instruction and image instead of having an LLM propose them. The
guardrail layer (stage 1) is the piece the redteam loop assumes is "handled
downstream" but never actually applied — here it is explicit and reusable.

Entry points:
  - CLI:    ``python scripts/generate_test_case.py --instruction ... --init-image ...``
  - module: ``from openworld.scenegen.pipeline import generate_test_cases``
  - prompt: ``from openworld.scenegen.guardrail import build_edit_prompt``
"""

from openworld.scenegen.guardrail import build_edit_prompt
from openworld.scenegen.nanobanana import (
    build_suite,
    build_suite_from_spec,
    nanobanana_edit,
    resolve_base,
)

__all__ = [
    "build_edit_prompt",
    "nanobanana_edit",
    "resolve_base",
    "build_suite",
    "build_suite_from_spec",
]
