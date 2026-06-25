#!/usr/bin/env python
"""Build a benchmark suite from a base view set + a YAML list of scene edits.

This is the usable front-end to the nanobanana all-views scene-edit mode: pick a
*base* (``tri`` or ``irom`` under ``assets/``, or any directory of three views),
list the edits you want, and get one ``init_<i>`` case per edit written under
``data/benchmark/<name>/`` — no GPU, no FLUX, just ``GOOGLE_API_KEY``.

    GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
        --spec configs/scenegen/suites/example.yaml

The spec is a YAML file (see configs/scenegen/suites/example.yaml):

    base: tri                 # assets/<base> or a path to a dir of 3 views
    name: my_suite            # output -> data/benchmark/my_suite
    keep: |                   # optional shared clause appended to every edit prompt
      Keep the robot arm, camera viewpoint, framing and all object positions
      exactly the same; do not move, remove, add, warp, or recolor any object.
    edits:
      - label: green_tabletop
        instruction: put the mug in the white container
        prompt: Change ONLY the wooden tabletop to a solid matte green tabletop.
      - label: forest
        instruction: put the mug in the white container
        prompt: Replace ONLY the background walls with an outdoor forest scene.

Anything on the command line (``--base``, ``--name``/``--out-dir``,
``--start-index``) overrides the matching spec key.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openworld.scenegen.nanobanana import build_suite_from_spec


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a benchmark suite from a base + a YAML list of scene edits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec", required=True, help="YAML suite spec (base + edits list).")
    p.add_argument("--base", default=None,
                   help="Override the spec's base (assets/<name> or a path to 3 views).")
    p.add_argument("--name", default=None,
                   help="Override the suite name (output -> data/benchmark/<name>).")
    p.add_argument("--out-dir", default=None,
                   help="Explicit output directory (overrides --name and the spec).")
    p.add_argument("--start-index", type=int, default=None,
                   help="First case index (use to append to an existing suite).")
    args = p.parse_args()

    overrides = {}
    if args.base is not None:
        overrides["base"] = args.base
    if args.name is not None:
        overrides["name"] = args.name
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.start_index is not None:
        overrides["start_index"] = args.start_index

    build_suite_from_spec(args.spec, **overrides)


if __name__ == "__main__":
    main()
