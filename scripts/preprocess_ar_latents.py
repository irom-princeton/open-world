"""Precompute 16-ch Wan-VAE latents for the AR world model.

Reads a raw dataset through a format adapter (RGB videos + video-rate 7-d
actions + text) and writes a standard latent layout the shared ARLatentDataset
trains on:

    <out>/<split>/<episode_id>.pt   # {latent: f16[V,16,Lf,h,w], action: f32[Lf,7], text, num_latent_frames}
    <out>/<split>_sample.json       # [{ep_id, num_latent_frames}, ...]
    <out>/stats.json                # {state_01, state_99} action percentiles (from train)

Run on a compute node (GPU + offline cache):

    sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/preprocess_ar_latents.py \
        --format droid_ctrl_world --root /scratch/.../data/droid_ctrl_world \
        --out data/droid_ar_latents --splits train val --num-views 3 [--limit 8]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from openworld.autoregressive.data.formats import build_format
from openworld.autoregressive.data.encode import VaeLatentEncoder, align_actions_to_latent


def _load_vae(vae_dir: str, dtype):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(vae_dir, subfolder="vae", torch_dtype=dtype)


def process_split(fmt, enc, split, out, *, limit, min_latent_frames):
    ep_ids = fmt.list_episodes(split)
    if limit:
        ep_ids = ep_ids[:limit]
    os.makedirs(os.path.join(out, split), exist_ok=True)
    sample_list, action_pool = [], []
    for i, ep_id in enumerate(ep_ids):
        ep = fmt.load_episode(ep_id, split)
        Lf = enc.latent_frames(ep.num_frames)
        if Lf < min_latent_frames:
            continue
        cams = []
        for vp in ep.video_paths:
            rgb = fmt.read_frames(vp)                       # [T,H,W,3] uint8
            cams.append(enc.encode_video(rgb))             # [16,Lf,h,w]
        latent = torch.stack(cams, dim=0)                  # [V,16,Lf,h,w]
        Lf = latent.shape[2]                               # actual, post-encode
        action = align_actions_to_latent(ep.actions, Lf)   # [Lf,7] raw
        torch.save(
            {"latent": latent, "action": torch.from_numpy(action).float(),
             "text": ep.text, "num_latent_frames": int(Lf)},
            os.path.join(out, split, f"{ep_id}.pt"),
        )
        sample_list.append({"ep_id": ep_id, "num_latent_frames": int(Lf)})
        action_pool.append(action)
        if (i + 1) % 50 == 0 or i + 1 == len(ep_ids):
            print(f"[{split}] {i + 1}/{len(ep_ids)}  (kept {len(sample_list)})", flush=True)
    with open(os.path.join(out, f"{split}_sample.json"), "w") as f:
        json.dump(sample_list, f)
    return action_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="droid_ctrl_world")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per split (0=all); for testing")
    ap.add_argument("--min-latent-frames", type=int, default=8)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "preprocess needs a GPU for the VAE"
    fmt = build_format(args.format, args.root, num_views=args.num_views)
    enc = VaeLatentEncoder(_load_vae(args.vae_dir, torch.float32), device="cuda")
    print(f"VAE temporal_factor={enc.temporal_factor}, z_dim={enc.z_dim}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    train_actions = []
    for split in args.splits:
        pool = process_split(fmt, enc, split, args.out,
                             limit=args.limit, min_latent_frames=args.min_latent_frames)
        if split == "train":
            train_actions = pool

    # Action normalisation stats from train (1st/99th percentile per dim).
    pool = list(train_actions)
    if not pool:  # only val processed -> sample a few train episodes for stats
        split = "train" if "train" in args.splits else args.splits[0]
        for ep_id in fmt.list_episodes(split)[:200]:
            pool.append(fmt.load_episode(ep_id, split).actions)
    alla = np.concatenate(pool, axis=0)
    stats = {"state_01": np.percentile(alla, 1, axis=0).tolist(),
             "state_99": np.percentile(alla, 99, axis=0).tolist()}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote stats.json:", stats, flush=True)
    print("PREPROCESS DONE")


if __name__ == "__main__":
    main()
