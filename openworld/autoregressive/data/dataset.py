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

from .views import select_view_indices


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

        # Auxiliary state-prediction target (off by default): per-frame ABSOLUTE state
        # + its own normalization stats. The aux head predicts this while the action
        # conditions on the commanded motion. Purely additive -- unused unless state_pred.
        self.state_pred = getattr(cfg, "state_pred", False)
        self.state_actions = None
        if self.state_pred:
            sfile = f"{split}_{getattr(cfg, 'state_pred_sidecar', 'joint_actions.npy')}"
            spath = os.path.join(self.root, sfile)
            if not os.path.exists(spath):
                raise FileNotFoundError(f"state_pred=True needs {spath} (abs-state sidecar)")
            self.state_actions = np.load(spath, allow_pickle=True).item()
            with open(os.path.join(self.root, getattr(cfg, "state_pred_stats", "stats_joint.json"))) as f:
                st = json.load(f)
            self.state_p01 = np.asarray(st["state_01"], dtype=np.float32)
            self.state_p99 = np.asarray(st["state_99"], dtype=np.float32)

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
        V_stored, C, Lf, h, w = latent.shape
        # Subset the stored cameras. An explicit ``view_indices`` pins an exact
        # heterogeneous subset (fixed for train+val); otherwise fall back to the
        # num_cams side-sampling (keep wrist + sample sides, random per clip while
        # training, fixed for val so previews are stable).
        vi = getattr(self.cfg, "view_indices", None)
        if vi:
            if max(vi) >= V_stored:
                raise ValueError(
                    f"ep {s['ep_id']}: view_indices {tuple(vi)} exceed stored views {V_stored}"
                )
            sel = list(vi)
        else:
            sel = select_view_indices(
                V_stored, self.num_cams, self.cfg.wrist_view_idx,
                deterministic=(self.split != "train"),
            )
        V = len(sel)

        start = random.randint(0, Lf - self.clip)
        end = start + self.clip
        lat = latent[sel][:, :, start:end]       # [V,16,L,h,w]
        # height-stack cameras: [V,16,L,h,w] -> [L,16,V*h,w]
        lat = lat.permute(2, 1, 0, 3, 4).reshape(self.clip, C, V * h, w).contiguous()

        act = self._norm(action[start:end], self.p01, self.p99)
        out = {
            "latent": lat.float(),
            "action": torch.from_numpy(act).float(),
            "text": rec.get("text", ""),
        }
        if self.state_pred:
            sa = np.asarray(self.state_actions[str(s["ep_id"])], dtype=np.float32)[start:end]
            out["state"] = torch.from_numpy(self._norm(sa, self.state_p01, self.state_p99)).float()
        return out
