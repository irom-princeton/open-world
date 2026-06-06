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

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ARLatentDataset(Dataset):
    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.root = cfg.latent_root
        self.num_cams = cfg.num_cams
        # window length in latent frames = (history + rollout) blocks * frames/block
        self.clip = (cfg.num_history_blocks + cfg.rollout_blocks) * cfg.frames_per_block

        with open(os.path.join(self.root, f"{split}_sample.json")) as f:
            samples = json.load(f)
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
