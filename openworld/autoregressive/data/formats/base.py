"""Raw-dataset format adapter interface.

A format knows how to enumerate episodes for a split and, per episode, hand back
the per-camera RGB video paths, the video-rate action sequence (7-d: EEF
cartesian xyz + axis-angle + gripper), and the language text. Everything
downstream (VAE encoding, latent layout, dataset, trainer) is format-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    ep_id: str
    split: str
    num_frames: int                 # number of (video-rate) RGB frames
    video_paths: list[str]          # absolute mp4 path per camera, len == num_views
    actions: np.ndarray             # [num_frames, action_dim], video-rate, UNnormalized
    text: str


class WorldModelFormat(ABC):
    action_dim: int = 7
    num_views: int = 1

    @abstractmethod
    def list_episodes(self, split: str) -> list[str]:
        """Episode ids available for ``split`` ('train' | 'val')."""

    @abstractmethod
    def load_episode(self, ep_id: str, split: str) -> Episode:
        """Metadata + action sequence + per-camera video paths for one episode."""

    # Shared RGB reader (decord). Returns uint8 [T, H, W, 3].
    @staticmethod
    def read_frames(video_path: str, frame_ids: list[int] | None = None) -> np.ndarray:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        if frame_ids is None:
            frame_ids = list(range(len(vr)))
        frame_ids = [min(int(i), len(vr) - 1) for i in frame_ids]
        return vr.get_batch(frame_ids).asnumpy()
