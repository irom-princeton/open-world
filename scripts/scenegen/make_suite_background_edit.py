"""Batch suite authoring — "background / lighting edit" mode (nanobanana-only).

Unlike the multiview add-object mode, nothing is added or removed: each of the
three real views is edited independently with the same theme prompt, so the
scene content (objects, robot, framing) stays put and only the surroundings /
lighting change. This authored init_8.. of the 0617_generated suite; the THEMES
list below is that recipe — edit it (or the env paths) for your own suite.

    GOOGLE_API_KEY=... python scripts/scenegen/make_suite_background_edit.py

Env overrides: SUITE (output suite dir), ORIG (dir of the 3 source views),
TPL (template initialization.yaml). Requires GOOGLE_API_KEY + `uv sync --extra scenegen`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanobanana_edit import nanobanana_edit  # noqa: E402

VIEWS = ["wrist", "exterior_left", "exterior_right"]
SUITE = Path(os.environ.get("SUITE", "/scratch/gpfs/AM43/yy4041/open-world/data/benchmark/0617_generated"))
ORIG = Path(os.environ.get("ORIG", str(SUITE / "_base_original")))
TPL = Path(os.environ.get("TPL", "configs/scenegen/template_vid15.yaml"))

# Shared instruction suffix: pin everything except the targeted surroundings.
KEEP = (
    "Keep the blue mug, the white container/bin, the robot arm and gripper, the "
    "camera viewpoint, framing, perspective, scale, and all object positions "
    "exactly the same. Do not move, remove, add, warp, or recolor any object."
)

# (case_idx, policy instruction, label, nanobanana edit prompt)
THEMES = [
    (8, "put the mug in the white container", "green_tabletop",
     "Change ONLY the wooden tabletop surface to a solid, uniform matte green "
     "tabletop (a clean, even green color). " + KEEP),
    (9, "put the mug in the white container", "forest_background",
     "Replace ONLY the room walls and background surrounding the table with an "
     "outdoor forest scene — green trees, foliage, and soft natural daylight — "
     "as if this table sits in a forest. Keep the tabletop itself unchanged. " + KEEP),
    (10, "put the mug in the white container", "restaurant_background",
     "Replace ONLY the room walls and background surrounding the table with the "
     "interior of a warm, cozy restaurant — distant dining tables, chairs, and "
     "ambient restaurant lighting and decor. Keep the tabletop itself "
     "unchanged. " + KEEP),
    (11, "put the mug in the white container", "red_tone_lighting",
     "Relight the entire scene with a warm red/amber tone, as if lit by a red "
     "light source, casting a reddish color tint over everything while keeping "
     "all details visible. Change ONLY the lighting color/tone across the whole "
     "image. " + KEEP),
]


def main() -> None:
    template_state = yaml.safe_load(open(TPL))["initial_state"]
    for idx, instruction, label, prompt in THEMES:
        case = SUITE / f"init_{idx}"
        case.mkdir(parents=True, exist_ok=True)
        for v in VIEWS:
            nanobanana_edit(str(ORIG / f"{v}.png"), str(case / f"{v}.png"), prompt)
        init = {
            "initial_state": template_state,
            "instruction": instruction,
            "metadata": {
                "suite": SUITE.name,
                "scene": "vid15_nanobanana",
                "task_type": "manipulation",
                "case_id": f"init_{idx}",
                "state_length": 7,
                "edit_mode": "nanobanana_all_views",
                "edit_label": label,
                "edit_prompt": prompt,
                "source_episode_id": 15,
            },
        }
        with open(case / "initialization.yaml", "w") as f:
            yaml.safe_dump(init, f, sort_keys=False)
        print(f"init_{idx} ({label}) done -> {case}")
    print("background-edit suite done.")


if __name__ == "__main__":
    main()
