"""Precompute 16-ch Wan-VAE latents for the AR world model.

Reads a raw dataset through a format adapter (RGB videos + video-rate 7-d
actions + text) and writes a standard latent layout the shared ARLatentDataset
trains on:

    <out>/<split>/<episode_id>.pt   # {latent: f16[V,16,Lf,h,w], action: f32[Lf,7], text, num_latent_frames}
    <out>/<split>_sample.json       # [{ep_id, num_latent_frames}, ...]
    <out>/stats.json                # {state_01, state_99} action percentiles (from train)

Single GPU (sequential over all episodes):

    sbatch bash_scripts/data_process/preprocess_latents.sh

Parallel (embarrassingly parallel — every episode is independent). Shard the
episodes across N GPUs with a SLURM array, then run a final CPU-only merge:

    # each array task processes episodes [shard_id :: num_shards] on its own GPU,
    # writing per-shard <split>_sample.part<id>of<n>.json:
    python scripts/preprocess_ar_latents.py ... --num-shards 4 --shard-id $SLURM_ARRAY_TASK_ID
    # then once (no GPU): concatenate the part lists -> <split>_sample.json and
    # compute stats.json from the train action percentiles:
    python scripts/preprocess_ar_latents.py ... --num-shards 4 --merge

See bash_scripts/data_process/preprocess_latents_array.sh + preprocess_merge.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np
import torch

from openworld.autoregressive.data.formats import build_format
from openworld.autoregressive.data.encode import VaeLatentEncoder, align_actions_to_latent


def _load_vae(vae_dir: str, dtype):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan.from_pretrained(vae_dir, subfolder="vae", torch_dtype=dtype)


def _sample_path(out: str, split: str, num_shards: int, shard_id: int) -> str:
    """Where a (possibly sharded) run writes its sample list. Unsharded runs write
    the final ``<split>_sample.json`` directly; shards write disjoint part files
    that ``--merge`` later concatenates."""
    if num_shards <= 1:
        return os.path.join(out, f"{split}_sample.json")
    return os.path.join(out, f"{split}_sample.part{shard_id:04d}of{num_shards:04d}.json")


def process_split(fmt, enc, split, out, *, limit, min_latent_frames, num_shards=1, shard_id=0):
    ep_ids = fmt.list_episodes(split)
    if limit:
        ep_ids = ep_ids[:limit]
    if num_shards > 1:
        ep_ids = ep_ids[shard_id::num_shards]   # disjoint, deterministic stripe per shard
    os.makedirs(os.path.join(out, split), exist_ok=True)
    tag = f"{split} shard {shard_id}/{num_shards}" if num_shards > 1 else split
    sample_list = []
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
        if (i + 1) % 50 == 0 or i + 1 == len(ep_ids):
            print(f"[{tag}] {i + 1}/{len(ep_ids)}  (kept {len(sample_list)})", flush=True)
    with open(_sample_path(out, split, num_shards, shard_id), "w") as f:
        json.dump(sample_list, f)
    return sample_list


def merge_sample_lists(out: str, split: str) -> int:
    """Concatenate per-shard ``<split>_sample.part*.json`` -> ``<split>_sample.json``."""
    parts = sorted(glob.glob(os.path.join(out, f"{split}_sample.part*of*.json")))
    if not parts:
        print(f"[merge] {split}: no shard part files found, skipping", flush=True)
        return 0
    seen, merged = set(), []
    for p in parts:
        for s in json.load(open(p)):
            if s["ep_id"] not in seen:                     # shards are disjoint; dedup defensively
                seen.add(s["ep_id"])
                merged.append(s)
    merged.sort(key=lambda s: s["ep_id"])
    with open(os.path.join(out, f"{split}_sample.json"), "w") as f:
        json.dump(merged, f)
    print(f"[merge] {split}: {len(parts)} parts -> {len(merged)} episodes", flush=True)
    return len(merged)


def compute_action_stats(fmt, split, *, sample_size, limit=0, seed=0) -> dict:
    """Action 1st/99th percentile per dim, read from annotation ``states`` (no GPU/
    video decode). Sampled by default — percentiles are stable with a few thousand
    episodes; pass ``sample_size=0`` to use every episode."""
    ep_ids = fmt.list_episodes(split)
    if limit:
        ep_ids = ep_ids[:limit]
    if sample_size and len(ep_ids) > sample_size:
        ep_ids = random.Random(seed).sample(ep_ids, sample_size)
    pool = []
    for j, ep_id in enumerate(ep_ids):
        try:
            pool.append(fmt.load_episode(ep_id, split).actions)
        except Exception:
            continue
        if (j + 1) % 2000 == 0 or j + 1 == len(ep_ids):
            print(f"[stats] {j + 1}/{len(ep_ids)} episodes read", flush=True)
    if not pool:
        raise RuntimeError(f"no usable {split} episodes for action stats")
    alla = np.concatenate(pool, axis=0)
    return {"state_01": np.percentile(alla, 1, axis=0).tolist(),
            "state_99": np.percentile(alla, 99, axis=0).tolist()}


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
    # --- parallel sharding ---
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (SLURM array size)")
    ap.add_argument("--shard-id", type=int, default=0, help="this task's shard in [0, num_shards)")
    ap.add_argument("--merge", action="store_true",
                    help="CPU-only: merge shard sample lists + write stats.json (no VAE/GPU)")
    ap.add_argument("--stats-split", default="train", help="split used for action percentiles")
    ap.add_argument("--stats-sample", type=int, default=8000,
                    help="episodes sampled for action percentiles (0=all)")
    args = ap.parse_args()

    if not (0 <= args.shard_id < max(1, args.num_shards)):
        raise SystemExit(f"--shard-id {args.shard_id} out of range for --num-shards {args.num_shards}")

    os.makedirs(args.out, exist_ok=True)
    fmt = build_format(args.format, args.root, num_views=args.num_views)

    # ---- merge mode: no GPU, finalize the sharded outputs ----
    if args.merge:
        for split in args.splits:
            merge_sample_lists(args.out, split)
        stats = compute_action_stats(fmt, args.stats_split, sample_size=args.stats_sample, limit=args.limit)
        with open(os.path.join(args.out, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print("wrote stats.json:", stats, flush=True)
        print("MERGE DONE")
        return

    # ---- encode mode: needs a GPU for the VAE ----
    assert torch.cuda.is_available(), "preprocess needs a GPU for the VAE"
    enc = VaeLatentEncoder(_load_vae(args.vae_dir, torch.float32), device="cuda")
    print(f"VAE temporal_factor={enc.temporal_factor}, z_dim={enc.z_dim}", flush=True)
    if args.num_shards > 1:
        print(f"SHARD {args.shard_id}/{args.num_shards}", flush=True)

    for split in args.splits:
        process_split(fmt, enc, split, args.out, limit=args.limit,
                      min_latent_frames=args.min_latent_frames,
                      num_shards=args.num_shards, shard_id=args.shard_id)

    # Unsharded run finalizes here; sharded runs defer to a single --merge pass
    # (so stats aren't computed N times / from partial data).
    if args.num_shards <= 1:
        stats = compute_action_stats(fmt, args.stats_split, sample_size=args.stats_sample, limit=args.limit)
        with open(os.path.join(args.out, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print("wrote stats.json:", stats, flush=True)
        print("PREPROCESS DONE")
    else:
        print(f"SHARD {args.shard_id}/{args.num_shards} DONE "
              f"(run --merge after all shards finish)")


if __name__ == "__main__":
    main()
