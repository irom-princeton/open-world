"""Encode LIBERO HDF5 demonstrations into the on-disk format consumed by
the world-model trainer.

Output layout (mirrors what
``Fast-Control-World/dataset/dataset_droid_exp33.py`` expects):

    <output_root>/<suite>/annotation/<split>/<episode_id>.json
    <output_root>/<suite>/latent_videos/agentview/<episode_id>.pt
    <output_root>/<suite>/latent_videos/wrist/<episode_id>.pt

Each annotation JSON looks like::

    {
        "texts": ["pick up the alphabet soup ..."],
        "observation.state.cartesian_position": [[x,y,z,ax,ay,az], ...],   # (T, 6)
        "observation.state.gripper_position":  [g, g, ...],                # (T,)
        "latent_videos": [
            {"latent_video_path": "latent_videos/agentview/00000.pt", "cam": "agentview"},
            {"latent_video_path": "latent_videos/wrist/00000.pt",     "cam": "wrist"}
        ],
        "fps": 20,
        "down_sample": 1,
        "task_suite": "libero_spatial",
        "language_instruction": "..."
    }

LIBERO's published demos are HDF5 files (one per BDDL task) with groups
``data/demo_<n>`` containing ``actions``, ``obs/agentview_rgb``,
``obs/eye_in_hand_rgb``, ``obs/ee_pos``, ``obs/ee_ori`` (quat),
``obs/gripper_states``, etc. Field names vary slightly across LIBERO
versions; this loader tolerates the common variants.

Example:

    python scripts/preprocess_libero_for_wm.py \\
        --libero_root data/raw_libero \\
        --task_suites libero_spatial libero_object libero_goal libero_10 libero_90 \\
        --output_root data/libero_processed \\
        --svd_path external/stable-video-diffusion-img2vid \\
        --val_fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-name compatibility with multiple LIBERO releases.
# ---------------------------------------------------------------------------

AGENT_RGB_KEYS = ["agentview_rgb", "agentview_image", "rgb"]
WRIST_RGB_KEYS = ["eye_in_hand_rgb", "robot0_eye_in_hand_image", "wrist_rgb"]
EEF_POS_KEYS = ["ee_pos", "eef_pos", "robot0_eef_pos"]
EEF_ORI_KEYS = ["ee_ori", "ee_quat", "robot0_eef_quat"]
GRIPPER_KEYS = ["gripper_states", "robot0_gripper_qpos", "gripper_position"]


def _first_existing(group: h5py.Group, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in group:
            return name
    raise KeyError(f"None of {list(candidates)} found in {group.name}")


def _quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat_xyzw)
    return rot.as_rotvec().astype(np.float32)


def _read_attr(grp: h5py.Group, key: str, default: str | None = None) -> str | None:
    val = grp.attrs.get(key, default)
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return val


# ---------------------------------------------------------------------------
# SVD-VAE latent encoding
# ---------------------------------------------------------------------------


@dataclass
class LatentEncoder:
    svd_path: str
    device: str = "cuda"
    target_h: int = 320
    target_w: int = 320
    chunk: int = 8

    def __post_init__(self) -> None:
        from diffusers import AutoencoderKLTemporalDecoder

        logger.info("Loading SVD VAE from %s", self.svd_path)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.svd_path, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        self.vae.eval()
        self.scale = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, frames_uint8: np.ndarray) -> torch.Tensor:
        """frames_uint8: (T, H, W, 3) uint8 -> (T, 4, 24, 40) float16 latents."""
        T = frames_uint8.shape[0]
        out = torch.empty((T, 4, self.target_h // 8, self.target_w // 8), dtype=torch.float16)
        for start in range(0, T, self.chunk):
            end = min(T, start + self.chunk)
            tile = []
            for i in range(start, end):
                img = Image.fromarray(frames_uint8[i])
                img = img.resize((self.target_w, self.target_h), Image.BICUBIC)
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                tile.append(arr)
            tensor = torch.tensor(np.stack(tile), dtype=torch.float16, device=self.device)
            tensor = tensor.permute(0, 3, 1, 2)  # (b,3,H,W)
            latent = self.vae.encode(tensor).latent_dist.mean * self.scale  # (b,4,h,w)
            out[start:end] = latent.cpu()
        return out


# ---------------------------------------------------------------------------
# HDF5 -> annotation + latent pipeline
# ---------------------------------------------------------------------------


def iter_episodes(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        problem_info = _read_attr(data_group, "problem_info", "")
        bddl_file_name = _read_attr(data_group, "bddl_file_name", str(hdf5_path.stem))
        for demo_name in sorted(data_group.keys(), key=lambda s: int(s.split("_")[-1])):
            demo = data_group[demo_name]
            obs = demo["obs"]
            language = _read_attr(demo, "language_instruction") or _read_attr(
                data_group, "language_instruction"
            )
            if language is None and problem_info:
                try:
                    info = json.loads(problem_info)
                    language = info.get("language_instruction", "")
                except json.JSONDecodeError:
                    language = ""

            agent_key = _first_existing(obs, AGENT_RGB_KEYS)
            wrist_key = _first_existing(obs, WRIST_RGB_KEYS)
            pos_key = _first_existing(obs, EEF_POS_KEYS)
            ori_key = _first_existing(obs, EEF_ORI_KEYS)
            grip_key = _first_existing(obs, GRIPPER_KEYS)

            # robosuite/MuJoCo offscreen renders are stored bottom-up; flip H.
            agent = np.ascontiguousarray(np.asarray(obs[agent_key])[:, ::-1])
            wrist = np.ascontiguousarray(np.asarray(obs[wrist_key])[:, ::-1])
            pos = np.asarray(obs[pos_key], dtype=np.float32)
            ori = np.asarray(obs[ori_key], dtype=np.float32)
            grip = np.asarray(obs[grip_key], dtype=np.float32)
            if grip.ndim == 2:
                grip = grip.mean(axis=-1)  # qpos has 2 finger joints -> aggregate

            if ori.shape[-1] == 4:
                axang = _quat_to_axisangle(ori)  # robosuite quat (x,y,z,w) -> rotvec
            else:
                axang = ori  # modified_libero stores ee_ori as axis-angle already
            cart = np.concatenate([pos, axang], axis=-1)  # (T, 6)

            yield {
                "demo_name": demo_name,
                "language": language or "",
                "agent_rgb": agent,
                "wrist_rgb": wrist,
                "cart": cart,
                "grip": grip,
                "bddl": bddl_file_name,
            }


def write_episode(
    *,
    suite: str,
    split: str,
    episode_id: str,
    output_root: Path,
    encoder: LatentEncoder | None,
    payload: dict,
) -> None:
    suite_root = output_root / suite
    ann_dir = suite_root / "annotation" / split
    agent_dir = suite_root / "latent_videos" / "agentview"
    wrist_dir = suite_root / "latent_videos" / "wrist"
    ann_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)

    agent_path = f"latent_videos/agentview/{episode_id}.pt"
    wrist_path = f"latent_videos/wrist/{episode_id}.pt"

    if encoder is not None:
        torch.save(encoder.encode(payload["agent_rgb"]), suite_root / agent_path)
        torch.save(encoder.encode(payload["wrist_rgb"]), suite_root / wrist_path)

    annotation = {
        "texts": [payload["language"]],
        "language_instruction": payload["language"],
        "task_suite": suite,
        "bddl": payload["bddl"],
        "fps": 20,
        "down_sample": 1,
        "observation.state.cartesian_position": payload["cart"].tolist(),
        "observation.state.gripper_position": payload["grip"].tolist(),
        "latent_videos": [
            {"latent_video_path": agent_path, "cam": "agentview"},
            {"latent_video_path": wrist_path, "cam": "wrist"},
        ],
    }
    with open(ann_dir / f"{episode_id}.json", "w") as f:
        json.dump(annotation, f)


# ---------------------------------------------------------------------------
# Sample-list generation
# ---------------------------------------------------------------------------


def write_sample_list(
    suite_root: Path,
    split: str,
    episode_ids: list[str],
    num_history: int,
    num_frames: int,
    down_sample: int,
) -> None:
    """Build the {split}_sample.json index used by the WM dataset loader.

    Each sample fixes a starting frame index. Following the DROID convention
    in ``dataset_droid_exp33.py`` the frame index is in *downsampled* units.
    """
    samples = []
    for episode_id in episode_ids:
        ann_path = suite_root / "annotation" / split / f"{episode_id}.json"
        with open(ann_path) as f:
            ann = json.load(f)
        T = len(ann["observation.state.cartesian_position"])
        # number of downsampled-rate frames available with margin for history+future
        max_start = max(1, (T // down_sample) - num_frames - 1)
        for start in range(num_history, max_start, max(1, num_frames // 2)):
            samples.append({"episode_id": episode_id, "frame_ids": [start]})
    out = suite_root / f"{split}_sample.json"
    with open(out, "w") as f:
        json.dump(samples, f)
    print(f"  wrote {len(samples)} samples to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SUITE_TO_DIR = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", type=Path, required=True,
                    help="Directory holding raw LIBERO HDF5 demos, organized as "
                         "<libero_root>/<suite_dir>/<*.hdf5>.")
    ap.add_argument("--task_suites", nargs="+", required=True,
                    choices=list(SUITE_TO_DIR.keys()))
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--svd_path", type=str, default=None,
                    help="Path to SVD model dir (with subfolder=vae). If unset, "
                         "annotation JSONs are still written but no latents are encoded.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--max_episodes_per_suite", type=int, default=None,
                    help="Useful for smoke tests.")
    ap.add_argument("--num_history", type=int, default=6)
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--down_sample", type=int, default=4,
                    help="20Hz LIBERO downsampled to (20/down_sample) Hz for WM training. "
                         "Default 4 -> 5Hz (matches DROID 15Hz/3).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    encoder = LatentEncoder(args.svd_path, device=args.device) if args.svd_path else None
    if encoder is None:
        logger.warning("No --svd_path provided: writing annotations only, no latents.")

    rng = np.random.default_rng(0)

    for suite in args.task_suites:
        suite_in = args.libero_root / SUITE_TO_DIR[suite]
        suite_out = args.output_root / suite
        if not suite_in.exists():
            logger.warning("missing suite dir %s", suite_in)
            continue

        train_ids: list[str] = []
        val_ids: list[str] = []

        episode_counter = 0
        hdf5_files = sorted(suite_in.glob("*.hdf5")) + sorted(suite_in.glob("*.h5"))
        if not hdf5_files:
            logger.warning("no HDF5 files in %s", suite_in)
            continue

        for hdf5_path in hdf5_files:
            logger.info("[%s] %s", suite, hdf5_path.name)
            for ep in iter_episodes(hdf5_path):
                if args.max_episodes_per_suite is not None and episode_counter >= args.max_episodes_per_suite:
                    break
                episode_id = f"{episode_counter:06d}"
                split = "val" if rng.random() < args.val_fraction else "train"
                payload = ep | {"hdf5_source": hdf5_path.name}
                write_episode(
                    suite=suite,
                    split=split,
                    episode_id=episode_id,
                    output_root=args.output_root,
                    encoder=encoder,
                    payload=payload,
                )
                (train_ids if split == "train" else val_ids).append(episode_id)
                episode_counter += 1
            else:
                continue
            break

        write_sample_list(suite_out, "train", train_ids,
                          num_history=args.num_history, num_frames=args.num_frames,
                          down_sample=args.down_sample)
        write_sample_list(suite_out, "val", val_ids,
                          num_history=args.num_history, num_frames=args.num_frames,
                          down_sample=args.down_sample)


if __name__ == "__main__":
    main()
