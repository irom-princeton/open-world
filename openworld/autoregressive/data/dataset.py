"""Shared latent dataset for the AR world model (format-agnostic).

Trains on the standard layout written by ``scripts/preprocess_ar_latents.py``:

    <latent_root>/<split>/<ep>.pt   # {latent f16[V,16,Lf,h,w], action f32[Lf,7], text, num_latent_frames}
    <latent_root>/<split>_sample.json
    <latent_root>/stats.json        # action {state_01, state_99}

Returns the trainer's contract per sample (keys match the legacy SVD loader, so
the trainer is reused): a contiguous window of ``L`` latent frames, cameras
height-stacked, actions normalised to [-1, 1].

    {"latent": f32[L, 16, num_cams*h, w], "action": f32[L, 7], "text": str}
"""

from __future__ import annotations

import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_sample_list(root: str, split: str) -> list[dict]:
    """Read the episode index for ``split``.

    Prefers the consolidated ``<split>_sample.json``; otherwise concatenates the
    per-shard ``<split>_sample.part*of*.json`` files written by a sharded
    preprocess run (deduped by ``ep_id``) — so a parallel run needs no merge step.
    """
    single = os.path.join(root, f"{split}_sample.json")
    if os.path.exists(single):
        with open(single) as f:
            return json.load(f)
    seen, samples = set(), []
    for p in sorted(glob.glob(os.path.join(root, f"{split}_sample.part*of*.json"))):
        with open(p) as f:
            for s in json.load(f):
                if s["ep_id"] not in seen:
                    seen.add(s["ep_id"])
                    samples.append(s)
    if not samples:
        raise FileNotFoundError(
            f"no {split}_sample.json (or {split}_sample.part*of*.json) under {root}"
        )
    return samples


class ARLatentDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.root = cfg.latent_root
        self.num_cams = cfg.num_cams
        # window length in latent frames = (history + rollout) blocks * frames/block
        self.clip = (cfg.num_history_blocks + cfg.rollout_blocks) * cfg.frames_per_block

        samples = _load_sample_list(self.root, split)
        self.samples = [s for s in samples if s["num_latent_frames"] >= self.clip]
        if not self.samples:
            raise RuntimeError(
                f"no {split} episodes with >= {self.clip} latent frames under {self.root}"
            )
        dropped = len(samples) - len(self.samples)
        print(f"[ARLatentDataset {split}] {len(self.samples)} clips "
              f"(dropped {dropped} shorter than {self.clip} latent frames)")

        with open(os.path.join(self.root, "stats.json")) as f:
            stat = json.load(f)
        self.p01 = np.asarray(stat["state_01"], dtype=np.float32)
        self.p99 = np.asarray(stat["state_99"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _norm(a: np.ndarray, lo, hi, eps=1e-8) -> np.ndarray:
        return np.clip(2 * (a - lo) / (hi - lo + eps) - 1, -1, 1)

    def __getitem__(self, index: int) -> dict:
        s = self.samples[index]
        rec = torch.load(os.path.join(self.root, self.split, f"{s['ep_id']}.pt"), weights_only=False)
        latent = rec["latent"].float()           # [V,16,Lf,h,w]
        action = rec["action"].numpy()           # [Lf,7]
        V, C, Lf, h, w = latent.shape
        V = min(V, self.num_cams)

        start = random.randint(0, Lf - self.clip)
        end = start + self.clip
        lat = latent[:V, :, start:end]           # [V,16,L,h,w]
        # height-stack cameras: [V,16,L,h,w] -> [L,16,V*h,w]
        lat = lat.permute(2, 1, 0, 3, 4).reshape(self.clip, C, V * h, w).contiguous()

        act = self._norm(action[start:end], self.p01, self.p99)
        return {
            "latent": lat.float(),
            "action": torch.from_numpy(act).float(),
            "text": rec.get("text", ""),
        }
