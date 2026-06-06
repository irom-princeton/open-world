"""Format adapter for the DROID ctrl-world dataset.

Layout (only ``annotation/`` + ``videos/`` are used)::

    <root>/annotation/<split>/<episode_id>.json
    <root>/videos/<split>/<episode_id>/{0,1,2}.mp4      # 3 cameras, 192x320

Annotation fields used:
    texts                -> [language instruction]
    video_length         -> number of video-rate frames (matches the mp4s)
    videos               -> [{video_path: "videos/<split>/<id>/<cam>.mp4"}, ...]
    states               -> [video_length, 7]  EEF cartesian(6) + gripper(1),
                            already at video rate (1:1 with frames), physical units.

The raw-rate ``action.*`` / ``observation.*`` arrays (length ``raw_length``) are
ignored in favour of the video-rate ``states`` field.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import Episode, WorldModelFormat


class DroidCtrlWorldFormat(WorldModelFormat):
    action_dim = 7
    num_views = 3

    def __init__(self, root: str, *, num_views: int | None = None):
        self.root = root
        if num_views is not None:
            self.num_views = num_views

    def _ann_dir(self, split: str) -> str:
        return os.path.join(self.root, "annotation", split)

    def list_episodes(self, split: str) -> list[str]:
        d = self._ann_dir(split)
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))

    def load_episode(self, ep_id: str, split: str) -> Episode:
        with open(os.path.join(self._ann_dir(split), f"{ep_id}.json")) as f:
            ann = json.load(f)
        n = int(ann["video_length"])
        actions = np.asarray(ann["states"], dtype=np.float32)[:n]   # [n, 7]
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"{ep_id}: states dim {actions.shape[1]} != {self.action_dim}")
        vids = ann["videos"]
        if len(vids) < self.num_views:
            raise ValueError(f"{ep_id}: {len(vids)} cameras < num_views {self.num_views}")
        video_paths = [os.path.join(self.root, vids[i]["video_path"]) for i in range(self.num_views)]
        text = ann["texts"][0] if ann.get("texts") else ann.get("language_instruction", "")
        return Episode(ep_id=ep_id, split=split, num_frames=n,
                       video_paths=video_paths, actions=actions, text=text)
