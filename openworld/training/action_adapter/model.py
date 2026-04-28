"""LIBERO action-adapter model.

Mirrors the ``Dynamics`` MLP from
``Fast-Control-World/models/action_adapter/train2.py:38`` but with
LIBERO-appropriate I/O semantics:

* DROID input: (current_joint, joint_velocity_chunk) -> future joint deltas
  -> FK -> Cartesian poses (done outside the MLP).
* LIBERO input: (current_eef_pose, future_delta_eef_chunk) -> future
  *absolute* EEF poses, no FK.

Pose convention: 6-D EEF (xyz + axis-angle). Gripper is concatenated as the
7th dim and treated as an absolute command (the model just learns to copy
it through, but training it lets the same module handle gripper smoothing /
quantization later if needed).
"""

from __future__ import annotations

from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reasonable LIBERO defaults for the (delta) action chunk and the
# (delta) absolute pose deltas. Override in the config if your dataset has
# a different distribution.
DEFAULT_LIBERO_DELTA_P01 = (-0.3, -0.3, -0.3, -0.5, -0.5, -0.5, -1.0)
DEFAULT_LIBERO_DELTA_P99 = (0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 1.0)
DEFAULT_LIBERO_POSE_DELTA_P01 = (-0.4, -0.4, -0.4, -1.0, -1.0, -1.0, -1.0)
DEFAULT_LIBERO_POSE_DELTA_P99 = (0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0)


class LiberoDynamics(nn.Module):
    """MLP that maps (current pose, delta-action chunk) -> future poses.

    Same shape signature as ``Dynamics`` in Fast-Control-World so the
    inference wrapper in ``openworld/policies/libero_action_adapter.py`` can
    use the same call pattern.

    Parameters
    ----------
    action_dim:
        7 for the LIBERO default (6 EEF + 1 gripper).
    action_num:
        Length of the action chunk per WM call. Default 15 (matches DROID).
    hidden_size:
        MLP hidden width. Default 512.
    delta_p01, delta_p99:
        Percentile normalization for the *input* delta chunk.
    pose_delta_p01, pose_delta_p99:
        Percentile normalization for the *target* (future_pose - current_pose).
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_num: int = 15,
        hidden_size: int = 512,
        delta_p01: Sequence[float] = DEFAULT_LIBERO_DELTA_P01,
        delta_p99: Sequence[float] = DEFAULT_LIBERO_DELTA_P99,
        pose_delta_p01: Sequence[float] = DEFAULT_LIBERO_POSE_DELTA_P01,
        pose_delta_p99: Sequence[float] = DEFAULT_LIBERO_POSE_DELTA_P99,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size

        self.register_buffer(
            "delta_p01", torch.tensor(delta_p01, dtype=torch.float32).view(1, 1, action_dim)
        )
        self.register_buffer(
            "delta_p99", torch.tensor(delta_p99, dtype=torch.float32).view(1, 1, action_dim)
        )
        self.register_buffer(
            "pose_delta_p01", torch.tensor(pose_delta_p01, dtype=torch.float32).view(1, 1, action_dim)
        )
        self.register_buffer(
            "pose_delta_p99", torch.tensor(pose_delta_p99, dtype=torch.float32).view(1, 1, action_dim)
        )

        input_dim = action_dim * (action_num + 1)
        output_dim = action_dim * action_num
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_dim),
        )

    # ------------------------------------------------------------------
    # Normalization helpers (kept identical to DROID for symmetry).
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return 2 * (x - lo) / (hi - lo + eps) - 1

    @staticmethod
    def _denormalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return (x + 1) / 2 * (hi - lo) + lo

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(
        self,
        current_pose: torch.Tensor,   # (B, 1, action_dim)
        delta_chunk: torch.Tensor,    # (B, action_num, action_dim)
        future_pose: torch.Tensor | None = None,  # (B, action_num, action_dim) for training
    ) -> torch.Tensor:
        if current_pose.dim() == 2:
            current_pose = current_pose.unsqueeze(1)
        if delta_chunk.dim() == 2:
            delta_chunk = delta_chunk.unsqueeze(0)

        if current_pose.shape[1:] != (1, self.action_dim):
            raise ValueError(
                f"current_pose: expected (B, 1, {self.action_dim}), got {tuple(current_pose.shape)}"
            )
        if delta_chunk.shape[1:] != (self.action_num, self.action_dim):
            raise ValueError(
                f"delta_chunk: expected (B, {self.action_num}, {self.action_dim}), got {tuple(delta_chunk.shape)}"
            )

        delta_norm = self._normalize(delta_chunk, self.delta_p01, self.delta_p99)
        b = current_pose.shape[0]
        x = torch.cat(
            [current_pose.reshape(b, -1), delta_norm.reshape(b, -1)], dim=1
        )  # (B, action_dim*(action_num+1))
        pred_norm = self.net(x)
        pred_norm = einops.rearrange(
            pred_norm, "b (t d) -> b t d", t=self.action_num, d=self.action_dim
        )
        pose_delta = self._denormalize(pred_norm, self.pose_delta_p01, self.pose_delta_p99)
        future_abs = current_pose + pose_delta

        if future_pose is None:
            return future_abs

        target_pose_delta = future_pose - current_pose
        target_norm = self._normalize(target_pose_delta, self.pose_delta_p01, self.pose_delta_p99)
        loss = F.mse_loss(pred_norm, target_norm)
        return loss

    # ------------------------------------------------------------------
    # Pure-numpy entry point for inference-time use (mirrors the DROID
    # adapter's call signature so callers can swap implementations).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_numpy(
        self,
        current_pose_np: np.ndarray,  # (action_dim,) or (1, action_dim) or (1, 1, action_dim)
        delta_chunk_np: np.ndarray,   # (action_num, action_dim) or (1, action_num, action_dim)
    ) -> np.ndarray:
        device = next(self.parameters()).device
        cur = torch.as_tensor(current_pose_np, dtype=torch.float32, device=device).reshape(
            1, 1, self.action_dim
        )
        chunk = torch.as_tensor(delta_chunk_np, dtype=torch.float32, device=device).reshape(
            1, self.action_num, self.action_dim
        )
        future_abs = self.forward(cur, chunk, future_pose=None)
        return future_abs[0].detach().cpu().numpy()


__all__ = ["LiberoDynamics"]
