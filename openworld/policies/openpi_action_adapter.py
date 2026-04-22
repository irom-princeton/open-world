from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """Ctrl-World adapter from 15x7 joint velocities to future 7D joints."""

    def __init__(self, action_dim: int, action_num: int, hidden_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size

        self.joint_vel_01 = np.array(
            [-0.4077107, -0.79047304, -0.47850373, -0.8666644, -0.6729502, -0.5602032, -0.692411],
            dtype=np.float32,
        )[None, :]
        self.joint_vel_99 = np.array(
            [0.4900636, 0.7259861, 0.45910007, 0.79220384, 0.69864315, 0.648198, 0.810115],
            dtype=np.float32,
        )[None, :]
        self.joint_delta_01 = np.array(
            [-0.2801219, -0.397792, -0.22935797, -0.3351759, -0.42025003, -0.36825255, -0.450706],
            dtype=np.float32,
        )[None, :]
        self.joint_delta_99 = np.array(
            [0.2827909, 0.42184818, 0.33529875, 0.35958457, 0.375613, 0.44463825, 0.4697690],
            dtype=np.float32,
        )[None, :]

        input_dim = int(action_dim * (action_num + 1))
        output_dim = int(action_num * action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, joint: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        if joint.ndim == 2:
            joint = joint[None, :]
        if joint_vel.ndim == 2:
            joint_vel = joint_vel[None, :]
        if joint.shape[1:] != (1, self.action_dim):
            raise ValueError(f"Expected joint shape (B, 1, {self.action_dim}), got {joint.shape}")
        if joint_vel.shape[1:] != (self.action_num, self.action_dim):
            raise ValueError(
                f"Expected joint_vel shape (B, {self.action_num}, {self.action_dim}), got {joint_vel.shape}"
            )

        device = next(self.parameters()).device
        joint_tensor = torch.as_tensor(joint, dtype=torch.float32, device=device)
        joint_vel = self.normalize_bound(joint_vel, self.joint_vel_01, self.joint_vel_99)
        joint_vel_tensor = torch.as_tensor(joint_vel, dtype=torch.float32, device=device)

        batch = joint_tensor.shape[0]
        joint_tensor = joint_tensor.reshape(batch, -1)
        joint_vel_tensor = joint_vel_tensor.reshape(batch, -1)
        pred = self.net(torch.cat((joint_tensor, joint_vel_tensor), dim=1))
        pred = einops.rearrange(pred, "b (t d) -> b t d", t=self.action_num, d=self.action_dim)

        pred = pred.detach().cpu().numpy()
        pred = self.denormalize_bound(pred, self.joint_delta_01, self.joint_delta_99)
        joint_np = joint_tensor.detach().cpu().numpy()
        joint_future = joint_np + pred
        return joint_future[0]

    @staticmethod
    def normalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        return 2 * (data - data_min) / (data_max - data_min + eps) - 1

    @staticmethod
    def denormalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        return (data - clip_min) / clip_range * (data_max - data_min) + data_min


def get_fk_solution(joint_angles: np.ndarray) -> np.ndarray:
    """Franka Panda forward kinematics copied locally to avoid external repo coupling."""

    def get_tf_mat(index: int, dh_params: list[list[float]]) -> np.ndarray:
        a = dh_params[index][0]
        d = dh_params[index][1]
        alpha = dh_params[index][2]
        theta = dh_params[index][3]
        q = theta

        return np.array(
            [
                [np.cos(q), -np.sin(q), 0, a],
                [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    dh_params = [
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        [0, 0.107, 0, 0],
        [0, 0, 0, -np.pi / 4],
        [0.0, 0.1034, 0, 0],
    ]

    transform = np.eye(4, dtype=np.float64)
    for index in range(8):
        transform = transform @ get_tf_mat(index, dh_params)
    return transform


@dataclass
class AdaptedActionChunk:
    env_actions: np.ndarray
    joint_positions: np.ndarray
    gripper_positions: np.ndarray


class OpenPIActionAdapter:
    """Port of Ctrl-World's OpenPI DROID joint-velocity adapter."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        action_num: int = 15,
        action_dim: int = 7,
        hidden_size: int = 512,
        gripper_max: float = 1.0,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.action_num = action_num
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gripper_max = gripper_max
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.dynamics_model = Dynamics(action_dim=action_dim, action_num=action_num, hidden_size=hidden_size)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.dynamics_model.load_state_dict(state_dict)
        self.dynamics_model.to(self.device)
        self.dynamics_model.eval()

    def adapt(self, current_joint: Any, current_gripper: Any, raw_action_chunk: np.ndarray) -> AdaptedActionChunk:
        raw_action_chunk = np.asarray(raw_action_chunk, dtype=np.float32)
        if raw_action_chunk.ndim != 2 or raw_action_chunk.shape[-1] < 8:
            raise ValueError(f"Expected raw OpenPI action chunk with shape (T, 8), got {raw_action_chunk.shape}")

        current_joint = np.asarray(current_joint, dtype=np.float32).reshape(1, self.action_dim)
        current_gripper = np.asarray(current_gripper, dtype=np.float32).reshape(1, 1)

        joint_vel = raw_action_chunk[:, : self.action_dim]
        gripper_pos = np.clip(raw_action_chunk[:, self.action_dim : self.action_dim + 1], 0.0, self.gripper_max)

        # Pad to action_num (15) by repeating last row if the policy outputs fewer steps
        T = joint_vel.shape[0]
        if T < self.action_num:
            joint_vel = np.concatenate(
                [joint_vel, np.repeat(joint_vel[-1:], self.action_num - T, axis=0)], axis=0,
            )
            gripper_pos = np.concatenate(
                [gripper_pos, np.repeat(gripper_pos[-1:], self.action_num - T, axis=0)], axis=0,
            )

        with torch.no_grad():
            future_joint = self.dynamics_model(current_joint, joint_vel)

        joint_pos = np.concatenate([current_joint, future_joint], axis=0)[: self.action_num]
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[: self.action_num]

        env_actions = []
        for index in range(joint_pos.shape[0]):
            joints = joint_pos[index, :7]
            if not np.all(np.isfinite(joints)):
                import logging
                logging.getLogger(__name__).warning(
                    "Non-finite joint values at step %d (likely policy divergence): %s",
                    index, joints,
                )
                joints = np.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)

            fk = get_fk_solution(joints)
            xyz = fk[:3, 3]
            rotation_matrix = fk[:3, :3]

            if not np.all(np.isfinite(rotation_matrix)):
                import logging
                logging.getLogger(__name__).warning(
                    "Non-finite FK rotation at step %d; substituting identity. joints=%s",
                    index, joints,
                )
                rotation_matrix = np.eye(3)
                xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

            euler = R.from_matrix(rotation_matrix).as_euler("xyz")
            env_actions.append(np.concatenate([xyz, euler, gripper_pos[index]], axis=0))

        return AdaptedActionChunk(
            env_actions=np.asarray(env_actions, dtype=np.float32),
            joint_positions=np.asarray(joint_pos, dtype=np.float32),
            gripper_positions=np.asarray(gripper_pos, dtype=np.float32),
        )


def resolve_initial_joint_state(state: Any, metadata: Optional[dict[str, Any]]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(state, dict):
        robot = state.get("robot")
        if isinstance(robot, dict):
            joint = robot.get("joint_position")
            if joint is None:
                joint = robot.get("joint_positions")
            gripper = robot.get("gripper_position")
            if joint is not None and gripper is not None:
                return np.asarray(joint, dtype=np.float32).reshape(-1), np.asarray(gripper, dtype=np.float32).reshape(-1)

    if not metadata:
        return None, None

    source_dataset = metadata.get("source_dataset")
    source_annotation = metadata.get("source_annotation")
    if not source_dataset or not source_annotation:
        return None, None

    annotation_path = Path(source_dataset) / source_annotation
    if not annotation_path.exists():
        return None, None

    import json

    with open(annotation_path) as handle:
        payload = json.load(handle)
    state_index = _parse_state_index(metadata.get("state_source"))
    raw_index = min(state_index * 3, len(payload.get("observation.state.joint_position", [])) - 1)
    if raw_index < 0:
        return None, None

    joint_series = payload.get("observation.state.joint_position")
    gripper_series = payload.get("observation.state.gripper_position")
    if not joint_series or not gripper_series:
        return None, None

    joint = np.asarray(joint_series[raw_index], dtype=np.float32).reshape(-1)
    gripper = np.asarray([gripper_series[raw_index]], dtype=np.float32)
    return joint, gripper


def _parse_state_index(state_source: Any) -> int:
    if not isinstance(state_source, str):
        return 0
    match = re.search(r"\[(\d+)\]", state_source)
    if match is None:
        return 0
    return int(match.group(1))
