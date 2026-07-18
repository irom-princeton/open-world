"""Data migration: shift already-preprocessed action conditioning from the Wan-VAE
temporal group's LAST frame (RGB 4i) to its CENTER (~4i-1), removing the ~1.4-frame
pose->pixel *lead* baked into data preprocessed before the ``align_actions_to_latent``
fix (see openworld/autoregressive/data/encode.py).

WHY interpolation (not a raw re-read): the raw per-RGB-frame ``states`` may no longer
be on disk -- only the Lf-subsampled actions survive in the .pt / sidecars. The stored
samples ARE dense samples (one per ~4 RGB frames) of a smooth EEF trajectory, so we
recover the group-center pose by interpolating each interior sample 0.25 of a gap back
toward its predecessor. Residual error is the sub-4-frame trajectory curvature
(sub-mm / sub-mrad), far below the ~1 cm lead being removed. Frame 0 (RGB 0, no
preceding group) is left unchanged, matching the fixed ``align_actions_to_latent``.

Interpolation rules per representation:
  - translation / gripper / joint / 6D-rotation dims: plain lerp -- all wrap-free.
  - Euler-XYZ rotation dims (``--euler-dims LO HI``, e.g. ``3 6`` for the DROID
    [xyz, euler-xyz, grip] layout): SLERP (shortest-arc) -- lerp would corrupt across
    the +/-pi wrap. Omit for wrap-free layouts (e.g. rot6d).

Delivery (sidecar, no bulk .pt rewrite):
  - cartesian: write NEW ``<split>_actions_aligned.npy`` (dict ep_id -> [Lf,D]); the
    dataloader (ARLatentDataset) + replay.load_full_episode prefer it over the .pt's
    baked ``action`` when present. Non-destructive; .pt latents untouched.
  - joint: overwrite the (small) existing sidecar in place, keeping a one-time
    ``<name>.orig.npy`` backup. It already loads through a sidecar, so no loader
    change is needed.

Idempotent: a cartesian run skips if the aligned sidecar exists; joint skips if a
.orig backup exists. Pass --force to redo.

    .venv/bin/python scripts/realign_actions_center.py \
        --latent-root data/droid_ar_latents --euler-dims 3 6 --parts cartesian joint
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

TEMPORAL_FACTOR = 4          # Wan VAE temporal compression
FRAC = ((TEMPORAL_FACTOR - 1) // 2) / TEMPORAL_FACTOR   # 0.25: gap fraction to shift back


def recenter(arr: np.ndarray, euler_slice=None, frac: float = FRAC) -> np.ndarray:
    """Shift each interior row of ``arr`` [Lf, D] back by ``frac`` of a gap toward the
    previous row. Row 0 unchanged. ``euler_slice=(lo,hi)`` slerps those dims instead of
    lerping (shortest-arc, wrap-safe)."""
    Lf = arr.shape[0]
    out = arr.astype(np.float64).copy()
    if Lf <= 1:
        return out.astype(arr.dtype)
    prev, cur = arr[:-1].astype(np.float64), arr[1:].astype(np.float64)
    # default: lerp every dim (new = (1-frac)*cur + frac*prev)
    out[1:] = (1.0 - frac) * cur + frac * prev
    if euler_slice is not None:
        lo, hi = euler_slice
        rp = R.from_euler("xyz", prev[:, lo:hi])
        rc = R.from_euler("xyz", cur[:, lo:hi])
        rel = (rp.inv() * rc).as_rotvec()                 # shortest-arc prev->cur
        rnew = rp * R.from_rotvec((1.0 - frac) * rel)     # move (1-frac) of the way to cur
        out[1:, lo:hi] = rnew.as_euler("xyz")
    return out.astype(arr.dtype)


def do_cartesian(root, split, euler_slice, force):
    out_path = os.path.join(root, f"{split}_actions_aligned.npy")
    if os.path.exists(out_path) and not force:
        print(f"[cartesian {split}] {os.path.basename(out_path)} exists -- skip (use --force)")
        return
    pts = sorted(glob.glob(os.path.join(root, split, "*.pt")))
    aligned = {}
    tr_deltas = []
    for i, p in enumerate(pts):
        ep = os.path.basename(p)[:-3]
        # mmap: fault in only the (tiny) action tensor, not the ~2MB latent per file.
        a = np.asarray(torch.load(p, map_location="cpu", mmap=True)["action"])
        b = recenter(a, euler_slice=euler_slice)
        aligned[ep] = b.astype(np.float32)
        tr_deltas.append(np.linalg.norm(b[1:, :3] - a[1:, :3], axis=1).mean() if a.shape[0] > 1 else 0.0)
        if (i + 1) % 20000 == 0:
            print(f"[cartesian {split}] {i+1}/{len(pts)}", flush=True)
    np.save(out_path, np.array(aligned, dtype=object), allow_pickle=True)
    md = float(np.mean(tr_deltas) * 1000) if tr_deltas else 0.0
    print(f"[cartesian {split}] wrote {len(aligned)} eps -> {os.path.basename(out_path)} "
          f"(mean |Dtrans|={md:.2f} mm)", flush=True)


def _backup(path):
    bak = path[:-4] + ".orig.npy"
    if not os.path.exists(bak):
        os.rename(path, bak)          # move original aside once
        return bak, True
    return bak, False


def do_joint(root, split, force):
    path = os.path.join(root, f"{split}_joint_actions.npy")
    if not os.path.exists(path):
        bak = path[:-4] + ".orig.npy"
        if os.path.exists(bak) and not force:
            print(f"[joint {split}] already migrated (.orig backup present) -- skip")
        else:
            print(f"[joint {split}] no {os.path.basename(path)} -- skip")
        return
    bak, moved = _backup(path)
    if not moved and not force:
        print(f"[joint {split}] backup exists -- skip (use --force)")
        return
    src = np.load(bak, allow_pickle=True).item()
    out = {ep: recenter(np.asarray(v, np.float32), euler_slice=None).astype(np.float32)
           for ep, v in src.items()}
    np.save(path, np.array(out, dtype=object), allow_pickle=True)
    print(f"[joint {split}] re-aligned {len(out)} eps (orig -> {os.path.basename(bak)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-root", required=True,
                    help="preprocessed latent root (contains <split>/ dirs + sidecars)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--parts", nargs="+", default=["cartesian", "joint"],
                    choices=["cartesian", "joint"])
    ap.add_argument("--euler-dims", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="cartesian dims [LO, HI) holding Euler-XYZ angles (slerped, "
                         "wrap-safe), e.g. '3 6' for [xyz, euler, grip]. Omit for "
                         "wrap-free layouts (rot6d etc.).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.latent_root
    if not os.path.isdir(root):
        raise SystemExit(f"no latent root {root}")
    euler = tuple(args.euler_dims) if args.euler_dims else None
    print(f"=== re-align {root}  parts={args.parts}  splits={args.splits} "
          f"(shift {FRAC} gap back, euler_slice={euler}) ===")
    for split in args.splits:
        if "cartesian" in args.parts:
            do_cartesian(root, split, euler, args.force)
        if "joint" in args.parts:
            do_joint(root, split, args.force)
    print("REALIGN DONE")


if __name__ == "__main__":
    main()
