"""Inference-time LIBERO action adapter.

This is the LIBERO analog of
``openworld/policies/openpi_action_adapter.py``. It loads the
:class:`LiberoDynamics` MLP trained by
``openworld/training/action_adapter/train.py`` and exposes the same
``adapt(...)`` interface used by the world-model env so policies can be
swapped (DROID vs LIBERO) without changing the rollout code.

Compared to the DROID adapter:
* No FK pass: LIBERO actions are already in EEF space, so the WM
  conditioning pose is just what the adapter outputs (xyz + axis-angle +
  gripper).
* Input from the policy: per-step delta-EEF actions (LIBERO env format),
  shape ``(T, 7)``. The first 6 dims are the xyz/axis-angle delta; the 7th
  is the absolute gripper command (LIBERO convention: -1 open, +1 close).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from openworld.training.action_adapter.model import LiberoDynamics


@dataclass
class AdaptedLiberoChunk:
    """Mirror of ``openpi_action_adapter.AdaptedActionChunk`` for LIBERO."""

    env_actions: np.ndarray        # (T, 7) -- exactly what env.step expects (delta + gripper)
    eef_poses: np.ndarray          # (T, 7) -- absolute EEF pose conditioning for the WM
    gripper_positions: np.ndarray  # (T, 1) -- absolute gripper command


class OpenPILiberoActionAdapter:
    """Loads a trained LiberoDynamics adapter and converts pi0 outputs."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        action_num: int = 15,
        action_dim: int = 7,
        hidden_size: int = 512,
        device: Optional[str] = None,
        # Tighten if you want to clip the gripper magnitude (e.g. 0.95 to
        # avoid saturation). Default 1.0 = pass-through.
        gripper_max: float = 1.0,
    ):
        self.checkpoint_path = checkpoint_path
        self.action_num = action_num
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gripper_max = gripper_max
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.dynamics_model = LiberoDynamics(
            action_dim=action_dim, action_num=action_num, hidden_size=hidden_size
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.dynamics_model.load_state_dict(state_dict)
        self.dynamics_model.to(self.device)
        self.dynamics_model.eval()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def adapt(
        self,
        current_eef_pose: Any,    # (6,) or (7,) -- LIBERO state[:6] (xyz + axis-angle), gripper auto-padded
        current_gripper: Any,     # scalar or (1,)
        raw_action_chunk: np.ndarray,  # (T, 7) from pi0 (delta-EEF + abs gripper)
    ) -> AdaptedLiberoChunk:
        raw = np.asarray(raw_action_chunk, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[-1] < self.action_dim:
            raise ValueError(
                f"Expected raw chunk shape (T, {self.action_dim}), got {raw.shape}"
            )

        cur_pose = np.asarray(current_eef_pose, dtype=np.float32).reshape(-1)
        cur_grip = np.asarray(current_gripper, dtype=np.float32).reshape(-1)
        if cur_pose.shape[0] == self.action_dim - 1:
            # Pose given as 6-D; concatenate gripper.
            cur_pose = np.concatenate([cur_pose, cur_grip[:1]], axis=0)
        if cur_pose.shape[0] != self.action_dim:
            raise ValueError(f"current_eef_pose must be {self.action_dim - 1} or {self.action_dim} dim")
        cur_pose = cur_pose.reshape(1, self.action_dim)

        delta_chunk = raw[:, : self.action_dim].copy()
        # The LIBERO env consumes the policy's raw output directly; we
        # forward it as-is, only clamping the gripper if requested.
        env_actions = raw.copy()
        env_actions[:, -1] = np.clip(env_actions[:, -1], -self.gripper_max, self.gripper_max)
        delta_chunk[:, -1] = env_actions[:, -1]

        # Pad to action_num if the policy emitted a shorter chunk.
        T = delta_chunk.shape[0]
        if T < self.action_num:
            pad = self.action_num - T
            delta_chunk = np.concatenate([delta_chunk, np.repeat(delta_chunk[-1:], pad, axis=0)], axis=0)
            env_actions = np.concatenate([env_actions, np.repeat(env_actions[-1:], pad, axis=0)], axis=0)

        with torch.no_grad():
            future_pose = self.dynamics_model.predict_numpy(cur_pose, delta_chunk[: self.action_num])

        # Prepend the current pose so the conditioning has length action_num.
        eef_poses = np.concatenate([cur_pose, future_pose], axis=0)[: self.action_num]
        gripper_positions = eef_poses[:, -1:].copy()

        return AdaptedLiberoChunk(
            env_actions=np.asarray(env_actions[: self.action_num], dtype=np.float32),
            eef_poses=np.asarray(eef_poses, dtype=np.float32),
            gripper_positions=np.asarray(gripper_positions, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Initial-state resolver, parallel to resolve_initial_joint_state in the
# DROID adapter. LIBERO state is (eef_pos[3] + eef_quat[4] + gripper_qpos[2])
# in the env, but the openpi LIBERO policy/wrappers already convert quat
# -> axis-angle so callers will typically pass the 6-D EEF pose directly.
# ---------------------------------------------------------------------------


def resolve_initial_eef_state(
    state: Any,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Try several common state shapes and return (eef_pose_6d, gripper_1d).

    Returns (None, None) if nothing matches; callers should then prompt the
    env for a fresh observation.
    """
    if isinstance(state, dict):
        # openpi LIBERO state dict (as built by examples/libero/main.py)
        if "robot0_eef_pos" in state and "robot0_eef_quat" in state:
            from scipy.spatial.transform import Rotation as R
            pos = np.asarray(state["robot0_eef_pos"], dtype=np.float32).reshape(3)
            quat = np.asarray(state["robot0_eef_quat"], dtype=np.float32).reshape(4)
            ax = R.from_quat(quat).as_rotvec().astype(np.float32)
            grip = np.asarray(state.get("robot0_gripper_qpos", [0.0]), dtype=np.float32).reshape(-1)
            if grip.shape[0] == 2:
                grip = grip.mean(keepdims=True)
            return np.concatenate([pos, ax]), grip[:1]

        # openworld convention: state["robot"] sub-dict
        robot = state.get("robot")
        if isinstance(robot, dict):
            if "eef_pose" in robot and "gripper_position" in robot:
                return (
                    np.asarray(robot["eef_pose"], dtype=np.float32).reshape(-1),
                    np.asarray(robot["gripper_position"], dtype=np.float32).reshape(-1),
                )

    return None, None


__all__ = [
    "AdaptedLiberoChunk",
    "OpenPILiberoActionAdapter",
    "resolve_initial_eef_state",
]
