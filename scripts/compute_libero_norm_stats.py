"""Compute LIBERO action / state percentile statistics for the world model.

Reads annotation JSONs produced by ``scripts/preprocess_libero_for_wm.py``
and writes a stat.json compatible with the DROID format.

Example:
    python scripts/compute_libero_norm_stats.py \\
        --processed_root data/libero_processed \\
        --suite libero_10 \\
        --output dataset_meta_info/libero/stat.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def collect(processed_root: Path, suite: str) -> np.ndarray:
    annotation_dir = processed_root / suite / "annotation"
    if not annotation_dir.exists():
        raise FileNotFoundError(f"No annotation dir at {annotation_dir}")
    rows = []
    for split in ("train", "val"):
        split_dir = annotation_dir / split
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob("*.json")):
            with open(jf) as f:
                ann = json.load(f)
            cart = np.asarray(ann["observation.state.cartesian_position"], dtype=np.float32)  # (T, 6)
            grip = np.asarray(ann["observation.state.gripper_position"], dtype=np.float32)
            if grip.ndim == 1:
                grip = grip[:, None]
            rows.append(np.concatenate([cart, grip], axis=-1))
    if not rows:
        raise RuntimeError(f"No episodes found under {annotation_dir}")
    return np.concatenate(rows, axis=0)  # (sumT, 7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", type=Path, required=True)
    ap.add_argument(
        "--suite",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="One or more processed LIBERO suite names. Stats are pooled.",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pooled = []
    for suite in args.suite:
        try:
            pooled.append(collect(args.processed_root, suite))
        except FileNotFoundError as e:
            print(f"[skip] {e}")
    if not pooled:
        raise SystemExit("No LIBERO data found.")
    data = np.concatenate(pooled, axis=0)
    p01 = np.quantile(data, 0.01, axis=0).tolist()
    p99 = np.quantile(data, 0.99, axis=0).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"state_01": p01, "state_99": p99}, f, indent=2)
    print(f"wrote {args.output} from {len(data)} timesteps across {len(pooled)} suite(s)")


if __name__ == "__main__":
    main()
