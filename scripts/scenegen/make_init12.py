"""Author init_12 of the 0617_generated suite — a combined scene edit.

Same nanobanana-only, all-views mode as make_suite_background_edit.py (init_8..11),
but this case intentionally edits scene *content*, not just the surroundings:
  - replace the background with a sunny beach,
  - recolor the wooden tabletop to a darker brown,
  - turn the white plastic container/bin into a cardboard box.
So it carries its own KEEP clause (the shared one pins the table + container, which
we are changing here). The blue mug, robot, viewpoint, and all positions stay put.

    GOOGLE_API_KEY=... <fork-venv>/python scripts/scenegen/make_init12.py

Env overrides: SUITE (output suite dir), ORIG (dir of the 3 source views),
TPL (template initialization.yaml).
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

IDX = 12
LABEL = "beach_darkwood_cardboard"
INSTRUCTION = "put the mug in the cardboard box"

# Pin everything we are NOT changing. Note: unlike init_8..11 this DOES allow the
# tabletop and the container to change, so they are deliberately left out of KEEP.
KEEP = (
    "Keep the blue mug, the robot arm and gripper, the camera viewpoint, framing, "
    "perspective, scale, and all object positions exactly the same. Do not move, "
    "remove, or add any object, and do not recolor or warp the blue mug or the robot."
)

PROMPT = (
    "Make three edits to this scene. "
    "(1) Replace the room, walls, and background surrounding the table with an "
    "outdoor sunny beach — golden sand, blue ocean, and a bright clear sky under "
    "warm midday sunlight. "
    "(2) Change the wooden tabletop surface to a darker, rich dark-brown wood, "
    "keeping the same wood grain and top-down framing. "
    "(3) Transform the white plastic container/bin on the table into a plain brown "
    "corrugated cardboard box of the same size and shape, sitting open in the exact "
    "same position. " + KEEP
)


def main() -> None:
    template_state = yaml.safe_load(open(TPL))["initial_state"]
    case = SUITE / f"init_{IDX}"
    case.mkdir(parents=True, exist_ok=True)
    for v in VIEWS:
        nanobanana_edit(str(ORIG / f"{v}.png"), str(case / f"{v}.png"), PROMPT)
    init = {
        "initial_state": template_state,
        "instruction": INSTRUCTION,
        "metadata": {
            "suite": SUITE.name,
            "scene": "vid15_nanobanana",
            "task_type": "manipulation",
            "case_id": f"init_{IDX}",
            "state_length": 7,
            "edit_mode": "nanobanana_all_views",
            "edit_label": LABEL,
            "edit_prompt": PROMPT,
            "source_episode_id": 15,
        },
    }
    with open(case / "initialization.yaml", "w") as f:
        yaml.safe_dump(init, f, sort_keys=False)
    print(f"init_{IDX} ({LABEL}) done -> {case}")


if __name__ == "__main__":
    main()
