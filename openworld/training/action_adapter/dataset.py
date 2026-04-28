"""Dataset of (current_pose, delta_chunk, future_pose) triples for training
the LIBERO action adapter.

Reads the same on-disk LIBERO format as the WM dataset
(``scripts/preprocess_libero_for_wm.py`` output), but uses only the
action / state arrays. No latent videos are loaded.

Conceptually:
    For each starting frame ``t``:
        current_pose = state[t]                           # (7,)
        future_pose  = state[t+1 : t+1+action_num]        # (action_num, 7)
        delta_chunk  = future_pose - current_pose         # (action_num, 7)

The ``delta_chunk`` is used as the *input* to the adapter (it stands in for
the policy's predicted delta-action chunk from pi0) and ``future_pose`` is
the supervision target. At inference time the policy provides the
``delta_chunk`` and the adapter predicts ``future_pose``.

This matches the role the DROID adapter plays: it learns the dynamics of
the "delta-action -> absolute-state" map empirically from data, so the WM
sees consistent action conditioning regardless of how the policy was
trained.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_episodes(split_dir: str) -> list[str]:
    if not os.path.isdir(split_dir):
        return []
    return sorted(p[: -len(".json")] for p in os.listdir(split_dir) if p.endswith(".json"))


class LiberoAdapterDataset(Dataset):
    """Iterates over (current_pose, delta_chunk, future_pose) triples."""

    def __init__(
        self,
        dataset_root: str,
        suites: list[str],
        mode: str = "train",
        annotation_name: str = "annotation",
        action_num: int = 15,
        action_dim: int = 7,
        # Native LIBERO control rate is 20 Hz. policy_skip_step parallels
        # the inference-time setting (each "policy step" advances the env
        # by `policy_skip_step` env steps so the policy's action chunk
        # spans (action_num * policy_skip_step) env steps).
        policy_skip_step: int = 2,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.suites = suites
        self.mode = mode
        self.annotation_name = annotation_name
        self.action_num = action_num
        self.action_dim = action_dim
        self.policy_skip_step = policy_skip_step

        self.entries: list[tuple[str, str]] = []  # (suite, episode_id)
        for suite in suites:
            split_dir = os.path.join(dataset_root, suite, annotation_name, mode)
            for ep_id in _list_episodes(split_dir):
                self.entries.append((suite, ep_id))
        if not self.entries:
            raise RuntimeError(
                f"No LIBERO annotations under {dataset_root}/<suite>/{annotation_name}/{mode}"
            )
        print(f"[LiberoAdapterDataset {mode}] {len(self.entries)} episodes across {len(suites)} suites")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_episode(self, suite: str, ep_id: str) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(self.dataset_root, suite, self.annotation_name, self.mode, f"{ep_id}.json")
        with open(path) as f:
            ann = json.load(f)
        cart = np.asarray(ann["observation.state.cartesian_position"], dtype=np.float32)  # (T, 6)
        grip = np.asarray(ann["observation.state.gripper_position"], dtype=np.float32)
        if grip.ndim == 1:
            grip = grip[:, None]
        pose = np.concatenate([cart, grip], axis=-1)  # (T, 7)
        return pose, np.array([1.0])  # second value unused; placeholder

    def fetch(self, index: int) -> dict[str, torch.Tensor]:
        suite, ep_id = self.entries[index]
        pose, _ = self._load_episode(suite, ep_id)
        T = pose.shape[0]
        span = self.action_num * self.policy_skip_step
        if T < span + 2:
            raise ValueError(f"episode {ep_id} too short: T={T}")

        start = random.randint(0, T - span - 1)
        idx = np.arange(start, start + span + 1, self.policy_skip_step)  # (action_num+1,)
        idx = np.clip(idx, 0, T - 1)
        sub = pose[idx]  # (action_num+1, 7)

        current_pose = sub[0:1]                                # (1, 7)
        future_pose = sub[1:]                                  # (action_num, 7)
        delta_chunk = future_pose - current_pose               # (action_num, 7)

        return {
            "current_pose": torch.from_numpy(current_pose).float(),
            "delta_chunk": torch.from_numpy(delta_chunk).float(),
            "future_pose": torch.from_numpy(future_pose).float(),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Same retry pattern as the DROID adapter dataset.
        try:
            return self.fetch(index)
        except Exception as e:
            print(f"[LiberoAdapterDataset] retry idx={index}: {e}")
            return self.fetch(random.randint(0, len(self.entries) - 1))


__all__ = ["LiberoAdapterDataset"]
