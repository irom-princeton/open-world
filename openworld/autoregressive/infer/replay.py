"""Open-loop trajectory replay for the autoregressive world model.

Given a ground-truth trajectory (preprocessed latents + recorded actions, e.g.
from the ``val`` split), we:

1. prime the KV-cache with the first ``history_blocks`` of *ground-truth* latents
   (the "first frame(s)");
2. feed the **full recorded action sequence** as conditioning (open-loop: actions
   come from the dataset, not a live policy);
3. let the student autoregressively generate every remaining latent block, each
   block conditioning on its *own* predictions via the cache.

The result is a predicted latent clip of the same length as the ground truth,
which the caller decodes with the Wan VAE and compares side-by-side.

This module is decode-free (latent-space only) so it can be unit-tested on CPU
with the dummy backbone; pixel decoding + video IO live in ``scripts/replay_ar.py``.
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data loading (matches scripts/preprocess_ar_latents.py + ARLatentDataset)
# ---------------------------------------------------------------------------
def load_action_stats(
    latent_root: str, stats_file: str = "stats.json",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the train-set action percentiles used to normalize actions.

    ``stats_file`` selects the action space's stats (``stats.json`` for cartesian,
    ``stats_joint.json`` for joint_pos) -- pass ``cfg.stats_file``.
    """
    with open(os.path.join(latent_root, stats_file)) as f:
        stat = json.load(f)
    return (np.asarray(stat["state_01"], dtype=np.float32),
            np.asarray(stat["state_99"], dtype=np.float32))


def load_joint_actions(latent_root: str, split: str) -> dict:
    """Load the joint-position sidecar for ``split`` (dict ep_id -> [Lf, 8])."""
    path = os.path.join(latent_root, f"{split}_joint_actions.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {split}_joint_actions.npy under {latent_root}")
    return np.load(path, allow_pickle=True).item()


def normalize_actions(action: np.ndarray, p01: np.ndarray, p99: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize raw actions to [-1, 1] exactly as ``ARLatentDataset`` does."""
    return np.clip(2 * (action - p01) / (p99 - p01 + eps) - 1, -1, 1).astype(np.float32)


def load_full_episode(
    latent_root: str, split: str, ep_id: str, num_cams: int, wrist_view_idx: int = 2,
    joint_actions: dict | None = None, view_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, np.ndarray, str]:
    """Load one preprocessed episode as the full (un-windowed) clip.

    Returns ``(latent [L, C, V*h, w] fp32, action [L, action_dim] raw, text)`` —
    cameras height-stacked, matching ``ARLatentDataset.__getitem__`` but over the
    whole episode rather than a random window. View subsetting (``num_cams`` <
    stored views -> wrist + side cameras) is deterministic here so previews/eval
    are reproducible.

    When ``joint_actions`` (a preloaded ``{split}_joint_actions.npy`` dict) is
    given, the 8-dim joint action for ``ep_id`` is returned instead of the .pt's
    cartesian ``action`` -- matching ``action_space='joint_pos'`` training.
    """
    from ..data.views import select_view_indices

    rec = torch.load(os.path.join(latent_root, split, f"{ep_id}.pt"), weights_only=False)
    latent = rec["latent"].float()                 # [V, C, Lf, h, w]
    if joint_actions is not None:
        if str(ep_id) not in joint_actions:
            raise KeyError(f"ep_id {ep_id!r} missing from {split}_joint_actions.npy")
        action = np.asarray(joint_actions[str(ep_id)], dtype=np.float32)  # [Lf, 8]
    else:
        action = rec["action"].numpy()             # [Lf, action_dim]
    V_stored, C, Lf, h, w = latent.shape
    # An explicit view_indices pins the exact subset (must match training); else
    # fall back to the num_cams wrist+side selection. Single source of truth with
    # ARLatentDataset so previews/eval use the SAME cameras the model trained on.
    if view_indices:
        if max(view_indices) >= V_stored:
            raise ValueError(f"view_indices {tuple(view_indices)} exceed stored views {V_stored}")
        sel = list(view_indices)
    else:
        sel = select_view_indices(V_stored, num_cams, wrist_view_idx, deterministic=True)
    V = len(sel)
    lat = latent[sel]                              # [V, C, Lf, h, w]
    # height-stack cameras: [V, C, Lf, h, w] -> [Lf, C, V*h, w]
    lat = lat.permute(2, 1, 0, 3, 4).reshape(Lf, C, V * h, w).contiguous()
    return lat.float(), action, rec.get("text", "")


# Scenegen "initialization suite" view order. The benchmark stores one PNG per
# view; we lay them out in DROID stored order ``[side, side, wrist]`` (wrist last)
# so ``select_view_indices`` / the height-stack match ``load_full_episode`` exactly.
_INIT_VIEW_FILES = ("exterior_left.png", "exterior_right.png", "wrist.png")
_INIT_RESIZE = (320, 192)                          # (W, H) -> 24x40 latents (VAE /8)


def load_init_frame(
    init_dir: str, num_cams: int, wrist_view_idx: int, encoder, action_space: str = "cartesian",
    view_indices: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, np.ndarray, str]:
    """Load a scenegen ``initialization`` directory as ONE latent frame + one action.

    ``init_dir`` holds per-view PNGs (``exterior_left.png``, ``exterior_right.png``,
    ``wrist.png``) and an ``initialization.yaml`` carrying the robot's initial_state
    + instruction. Each view image is VAE-encoded to a single latent frame, the
    views are subset (``num_cams``) and height-stacked exactly like
    :func:`load_full_episode`, and the robot's initial pose becomes one raw action.

    Returns ``(latent [1, C, V*h, w] fp32, action [action_dim] raw fp32, text)``.
    The caller repeats the single latent frame to fill the model's history/init
    block (there is no recorded clip to prime from -- only a still initialization).
    This is the latent-free seed path: no preprocessed ``.pt`` episode required, only
    the still PNGs (a couple of these ship in ``assets/teleop_inits/``).
    """
    import yaml
    from PIL import Image

    from ..data.views import select_view_indices

    cams = []
    for fn in _INIT_VIEW_FILES:
        img = Image.open(os.path.join(init_dir, fn)).convert("RGB").resize(_INIT_RESIZE)
        rgb = np.asarray(img, dtype=np.uint8)[None]       # [1, H, W, 3] (T=1)
        cams.append(encoder.encode_video(rgb).float())    # [C, 1, h, w]
    latent = torch.stack(cams, dim=0)                      # [V_stored, C, 1, h, w]
    V_stored, C, Lf, h, w = latent.shape
    if view_indices:
        if max(view_indices) >= V_stored:
            raise ValueError(f"view_indices {tuple(view_indices)} exceed stored views {V_stored}")
        sel = list(view_indices)
    else:
        sel = select_view_indices(V_stored, num_cams, wrist_view_idx, deterministic=True)
    V = len(sel)
    lat = latent[sel].permute(2, 1, 0, 3, 4).reshape(Lf, C, V * h, w).contiguous()  # [1, C, V*h, w]

    with open(os.path.join(init_dir, "initialization.yaml")) as f:
        meta = yaml.safe_load(f)
    robot = meta["initial_state"]["robot"]
    if action_space == "joint_pos":
        jp = np.asarray(robot["joint_position"], dtype=np.float32).reshape(-1)[:7]
        grip = np.asarray(robot["gripper_position"], dtype=np.float32).reshape(-1)[:1]
        action = np.concatenate([jp, grip]).astype(np.float32)        # [8]
    else:
        action = np.asarray(robot["state"], dtype=np.float32).reshape(-1)[:7]  # [7]
    return lat.float(), action, str(meta.get("instruction", ""))


# ---------------------------------------------------------------------------
# Replay (latent space)
# ---------------------------------------------------------------------------
@torch.no_grad()
def replay_episode_latents(
    model,
    latent_gt: torch.Tensor,        # [L, C, V*h, w] fp32, latent-frame rate
    action_norm: np.ndarray,        # [L, action_dim] normalized, 1:1 with latent frames
    *,
    frames_per_block: int,
    num_history_blocks: int,
    in_channels: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_blocks: int | None = None,
    scheduler=None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Open-loop AR replay in latent space.

    Returns ``(gt [N, C, V*h, w], pred [N, C, V*h, w], n_history_frames)`` where
    ``N = n_history_frames + generated`` and ``pred`` is the history (ground truth)
    concatenated with the generated blocks, aligned 1:1 with ``gt`` for comparison.
    """
    fpb = frames_per_block
    L, C, Hs, W = latent_gt.shape
    assert C == in_channels, f"latent channels {C} != backbone in_channels {in_channels}"

    hist_frames = num_history_blocks * fpb
    num_blocks = (L - hist_frames) // fpb
    if max_blocks is not None:
        num_blocks = min(num_blocks, max_blocks)
    if num_blocks < 1:
        raise RuntimeError(
            f"episode too short: {L} latent frames, need > {hist_frames} "
            f"(history {num_history_blocks} blocks x {fpb})"
        )
    N = hist_frames + num_blocks * fpb

    pdt = dtype or next(model.parameters()).dtype
    gt = latent_gt[:N].to(device)
    hist = gt[:hist_frames].to(pdt)                                   # [hist_frames, C, Hs, W]
    history_blocks = [hist[i:i + fpb].unsqueeze(0)                    # each [1, fpb, C, Hs, W]
                      for i in range(0, hist_frames, fpb)]

    actions = torch.from_numpy(action_norm[:N]).float().unsqueeze(0).to(device)   # [1, N, A]

    ac = (torch.autocast(device.type, dtype=pdt)
          if device.type == "cuda" else contextlib.nullcontext())
    with ac:
        cond = model.encode_cond(actions, cfg_drop=False)            # [1, N, cross]
        latent_block_shape = (1, fpb, in_channels, Hs, W)
        gen = model.rollout(
            history_blocks, cond,
            num_blocks=num_blocks, latent_block_shape=latent_block_shape,
            scheduler=scheduler,
        )                                                            # [1, num_blocks*fpb, C, Hs, W]

    gen = gen[0].float().cpu()
    pred = torch.cat([hist.float().cpu(), gen], dim=0)               # [N, C, Hs, W]
    return gt.float().cpu(), pred, hist_frames
