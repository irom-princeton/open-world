"""LIBERO world-model dataset loader.

Mirrors ``Fast-Control-World/dataset/dataset_droid_exp33.py`` for LIBERO.

Key differences from DROID:
* Two cameras (agentview + wrist) stacked vertically into latent shape
  ``(4, num_cams * 24, 40)`` -- that's ``(4, 48, 40)`` by default vs
  DROID's ``(4, 72, 40)``.
* Action conditioning is the absolute end-effector pose (xyz + axis-angle)
  plus the absolute gripper command, normalized to [-1, 1] using percentile
  stats from ``dataset_meta_info/<suite>/stat.json`` (or, as a fallback,
  ``dataset_meta_info/libero/stat.json``).
* ``down_sample`` defaults to 4 (20 Hz -> 5 Hz) instead of 3.

The on-disk format is what ``scripts/preprocess_libero_for_wm.py`` writes:

    <dataset_root_path>/<suite>/annotation/<split>/<episode_id>.json
    <dataset_root_path>/<suite>/latent_videos/<cam>/<episode_id>.pt
    <dataset_root_path>/<suite>/{train,val}_sample.json
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_stat(meta_root: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (state_p01, state_p99) for normalization. Accepts either a
    per-suite file (``<meta_root>/<suite>/stat.json``) or a single shared
    file at ``<meta_root>/stat.json`` (or the legacy pooled
    ``<meta_root>/libero/stat.json``)."""
    candidates = [
        os.path.join(meta_root, dataset_name, "stat.json"),
        os.path.join(meta_root, "stat.json"),
        os.path.join(meta_root, "libero", "stat.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                stat = json.load(f)
            p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]
            p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]
            return p01, p99
    raise FileNotFoundError(f"No stat.json found under {candidates}")


class LiberoLatentDataset(Dataset):
    """Loads pre-encoded LIBERO VAE latents + per-frame EEF/gripper actions.

    The interface (``__getitem__`` returns ``{'latent','action','text'}``)
    is identical to ``Fast-Control-World/dataset/dataset_droid_exp33.py``
    so the trainer can be a near-verbatim port.
    """

    def __init__(self, args, mode: str = "train"):
        super().__init__()
        self.args = args
        self.mode = mode

        self.dataset_path_all: list[list[str]] = []
        self.samples_all: list[list[dict[str, Any]]] = []
        self.samples_len: list[int] = []
        self.norm_all: list[tuple[np.ndarray, np.ndarray]] = []

        dataset_root = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        meta_root = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = list(args.prob)
        if len(self.prob) != len(dataset_names):
            raise ValueError(
                f"len(prob)={len(self.prob)} != len(dataset_names)={len(dataset_names)}"
            )

        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            sample_path = os.path.join(dataset_root, dataset_cfg, f"{mode}_sample.json")
            with open(sample_path) as f:
                samples = json.load(f)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
            self.dataset_path_all.append(
                [os.path.join(dataset_root, dataset_name) for _ in samples]
            )
            self.norm_all.append(_load_stat(meta_root, dataset_name))
            print(f"[LiberoLatentDataset {mode}] {dataset_name}: {len(samples)} samples")

        if not self.samples_all:
            raise RuntimeError("No LIBERO datasets found.")
        self.max_id = max(self.samples_len)

    def __len__(self) -> int:
        return self.max_id

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_latent_video(video_path: str, frame_ids: list[int]) -> torch.Tensor:
        with open(video_path, "rb") as f:
            video_tensor = torch.load(f)
        video_tensor.requires_grad = False
        max_frames = video_tensor.shape[0]
        clamped = [min(int(i), max_frames - 1) for i in frame_ids]
        return video_tensor[clamped]

    @staticmethod
    def normalize_bound(
        data: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        return np.clip(2 * (data - lo) / (hi - lo + eps) - 1, -1, 1)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _build_frame_ids(self, frame_now: int, frame_len: int) -> tuple[list[int], np.ndarray]:
        """Same temporal layout as the DROID loader:
        ``num_history`` frames going back, then ``num_frames`` future frames.
        Random skip with occasional zeroing for history augmentation."""
        skip = random.randint(1, 2)
        skip_his = int(skip * 4)
        if random.random() < 0.15:
            skip_his = 0

        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.clip(np.asarray(rgb_id), 0, frame_len).tolist()
        rgb_id = [int(x) for x in rgb_id]
        state_id = np.asarray(rgb_id) * self.args.down_sample
        return rgb_id, state_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Pick a sub-dataset weighted by prob.
        dataset_id = int(np.random.choice(len(self.samples_all), p=self.prob))
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        frame_ids = sample["frame_ids"]
        ann_file = os.path.join(
            dataset_dir, self.args.annotation_name, self.mode, f"{sample['episode_id']}.json"
        )
        with open(ann_file) as f:
            label = json.load(f)

        # Frame indices live in WM-rate units (e.g. 5 Hz for default config).
        # State arrays in the annotation live at native (20 Hz) rate, so we
        # multiply by down_sample to index them.
        joint_len = len(label["observation.state.cartesian_position"]) - 1
        frame_len = int(np.floor(joint_len / self.args.down_sample))
        frame_now = int(frame_ids[0])
        rgb_id, state_id = self._build_frame_ids(frame_now, frame_len)

        # Stack two camera latents vertically along H.
        per_cam_h = self.args.height // 8  # 24 at height=192
        total_h = self.args.num_cams * per_cam_h
        latent_w = self.args.width // 8
        latent = torch.zeros(
            (self.args.num_frames + self.args.num_history, 4, total_h, latent_w),
            dtype=torch.float32,
        )
        cam_specs = label.get("latent_videos", [])
        if len(cam_specs) < self.args.num_cams:
            raise ValueError(
                f"{ann_file} declares {len(cam_specs)} cameras but config asks for {self.args.num_cams}"
            )
        for cam_idx in range(self.args.num_cams):
            video_path = os.path.join(dataset_dir, cam_specs[cam_idx]["latent_video_path"])
            cam_latent = self._load_latent_video(video_path, rgb_id)
            latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = cam_latent

        # Action conditioning: cartesian + gripper.
        cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.cartesian_position"]) - 1)
        ]
        grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)[
            np.clip(state_id, 0, len(label["observation.state.gripper_position"]) - 1)
        ]
        if grip.ndim == 1:
            grip = grip[:, None]
        action = np.concatenate([cart, grip], axis=-1)
        action = self.normalize_bound(action, state_p01, state_p99)

        return {
            "text": label["texts"][0] if label.get("texts") else label.get("language_instruction", ""),
            "latent": latent.float(),
            "action": torch.tensor(action, dtype=torch.float32),
        }


__all__ = ["LiberoLatentDataset"]
