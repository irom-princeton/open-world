"""Remove a named object from each view of an init to make an empty-table base.

Object-removal nanobanana edit (camera/everything-else held fixed), applied to
each view of a source init dir, producing an "empty base" that
``make_suite_add_object.sh`` then populates with new objects. This is how the
``_base_no_mug`` base for the 0617 suite was made (object="green mug").

    GOOGLE_API_KEY=... python scripts/scenegen/remove_object.py \\
        --object "green mug" --src-dir <suite>/_base_original --dst-dir <suite>/_base_no_mug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanobanana_edit import nanobanana_edit  # noqa: E402

PROMPT_TMPL = (
    "This is a fixed camera frame. Remove ONLY the {obj} from the table and "
    "inpaint that spot with the surrounding table surface so it looks like empty "
    "table. This is an object-removal edit only: do NOT change the camera angle, "
    "zoom, framing, or perspective; do NOT move, warp, or re-render anything else. "
    "Keep absolutely everything else pixel-identical: the robot arm and gripper, "
    "any shelf/stand, the floor, the background, and the lighting. Do not add or "
    "recolor any object."
)


def main() -> None:
    p = argparse.ArgumentParser(description="Remove an object from each view of an init.")
    p.add_argument("--object", required=True, help='e.g. "green mug"')
    p.add_argument("--src-dir", required=True, help="dir with the per-view PNGs")
    p.add_argument("--dst-dir", required=True, help="output dir for the edited views")
    p.add_argument("--views", nargs="+", default=["wrist", "exterior_left", "exterior_right"])
    args = p.parse_args()

    prompt = PROMPT_TMPL.format(obj=args.object)
    for v in args.views:
        src = Path(args.src_dir) / f"{v}.png"
        if not src.exists():
            print(f"  WARN: missing {src}, skipping"); continue
        nanobanana_edit(str(src), str(Path(args.dst_dir) / f"{v}.png"), prompt)
    print("remove_object done.")


if __name__ == "__main__":
    main()
