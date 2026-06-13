"""Wrapper for Diffusion Policy (irom-princeton/dppo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from openworld.policies.base_policy import Policy
from openworld.policies.openpi_action_adapter import AdaptedActionChunk, OpenPIActionAdapter
from openworld.policies.dppo_loader import (
    DEFAULT_POLICY_JSON,
    DEFAULT_DPPO_REPO,
    ensure_dppo_repo_on_path,
    load_policy_from_checkpoint,
    load_policy_paths_from_alias,
)


class DPPolicy(Policy):
    """Adapter around a Diffusion Policy checkpoint.

    DPPO policies trained on DROID emit joint *velocities* + gripper position
    per timestep. The world model expects absolute cartesian end-effector
    actions (xyz + euler + gripper), so we integrate joint velocities into
    joint positions with a learned dynamics model (same checkpoint OpenPI
    uses) and then apply forward kinematics.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        normalization_path: Optional[str] = None,
        policy_alias: Optional[str] = None,
        policy_json: str = str(DEFAULT_POLICY_JSON),
        repo_path: Optional[str] = str(DEFAULT_DPPO_REPO),
        ordered_obs_keys: Optional[list[str]] = None,
        camera_indices: Optional[list[int]] = None,
        view_names: Optional[list[str]] = None,
        stacked_view_order: Optional[list[str]] = None,
        act_steps: Optional[int] = None,
        action_adapter_checkpoint_path: Optional[str] = None,
        action_adapter_gripper_max: float = 1.0,
        policy_skip_step: int = 1,
        num_action_steps: Optional[int] = None,
        joint_position_dim: int = 7,
        device: str = "cuda",
        **_: Any,
    ):
        self.config_path = config_path
        self.normalization_path = normalization_path
        self.policy_alias = policy_alias
        self.policy_json = policy_json
        self.repo_path = repo_path
        self.ordered_obs_keys = ordered_obs_keys
        self.camera_indices = camera_indices
        self.view_names = list(view_names or ["exterior_right", "wrist"])
        self.stacked_view_order = list(
            stacked_view_order or ["exterior_left", "exterior_right", "wrist"]
        )
        self.act_steps = act_steps
        self.action_adapter_checkpoint_path = action_adapter_checkpoint_path
        self.action_adapter_gripper_max = action_adapter_gripper_max
        self.policy_skip_step = max(int(policy_skip_step), 1)
        self.num_action_steps = num_action_steps
        self.joint_position_dim = joint_position_dim
        self.device = device

        self._policy = None
        self._action_adapter: Optional[OpenPIActionAdapter] = None
        self._instruction: Optional[str] = None
        self._pending_actions: list[Any] = []

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = instruction
        self._pending_actions = []
        if self._policy is not None:
            self._policy.reset()

    def act(
        self,
        observation: Any,
        state: Any,
        instruction: Optional[str] = None,
    ) -> Any:
        if self._policy is None:
            raise RuntimeError(
                "DPPolicy.act() requires a loaded DPPO model. "
                "Set `checkpoint_path` in the policy config and provide the "
                "matching DPPO config and normalization stats from your "
                "`external/dsrl/dppo` fork."
            )

        if not self._pending_actions:
            dp_observation = self._build_dp_observation(observation=observation, state=state)
            result = self._policy.infer(dp_observation)
            predicted = np.asarray(result["actions"], dtype=np.float32)
            if predicted.ndim == 1:
                predicted = predicted[np.newaxis, :]
            self._pending_actions = self._adapt_action_chunk(predicted, state)

        if not self._pending_actions:
            raise RuntimeError("DPPolicy produced an empty action sequence.")

        return self._pending_actions.pop(0)

    def _adapt_action_chunk(self, predicted: np.ndarray, state: Any) -> list[Any]:
        """Integrate joint-velocity chunk to cartesian env actions via dynamics + FK.

        Falls back to returning raw 1D actions when no adapter checkpoint is
        configured (e.g. in tests). Real evaluation must set
        ``action_adapter_checkpoint_path``; without integration, FK on joint
        velocities collapses to the home pose and the WM renders a static robot.
        """
        if self.action_adapter_checkpoint_path is None:
            return [np.asarray(action, dtype=np.float32) for action in predicted]

        adapter = self._get_action_adapter()
        joint_position, gripper_position = self._extract_joint_state(state)
        adapted = adapter.adapt(joint_position, gripper_position, predicted)

        if self.policy_skip_step > 1 or self.num_action_steps is not None:
            indices = list(range(0, adapted.env_actions.shape[0], self.policy_skip_step))
            if self.num_action_steps is not None:
                indices = indices[: self.num_action_steps]
            adapted = AdaptedActionChunk(
                env_actions=adapted.env_actions[indices],
                joint_positions=adapted.joint_positions[indices],
                gripper_positions=adapted.gripper_positions[indices],
            )

        step_actions: list[dict[str, Any]] = []
        for index in range(adapted.env_actions.shape[0]):
            cartesian_action = adapted.env_actions[index]
            next_joint = adapted.joint_positions[index]
            next_gripper = adapted.gripper_positions[index]
            step_actions.append(
                {
                    "env_action": cartesian_action,
                    "state_update": {
                        "robot": {
                            "state_representation": "cartesian_position_with_gripper",
                            "state": cartesian_action,
                            "cartesian_position": cartesian_action[:6],
                            "joint_position": next_joint,
                            "joint_positions": next_joint,
                            "gripper_position": next_gripper,
                        }
                    },
                }
            )
        return step_actions

    def _get_action_adapter(self) -> OpenPIActionAdapter:
        if self._action_adapter is None:
            self._action_adapter = OpenPIActionAdapter(
                checkpoint_path=self.action_adapter_checkpoint_path,
                gripper_max=self.action_adapter_gripper_max,
                action_dim=self.joint_position_dim,
                device=self.device,
            )
        return self._action_adapter

    def _extract_joint_state(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(state, dict):
            robot = state.get("robot")
            if isinstance(robot, dict):
                joint = robot.get("joint_position")
                if joint is None:
                    joint = robot.get("joint_positions")
                gripper = robot.get("gripper_position")
                if joint is not None and gripper is not None:
                    return (
                        np.asarray(joint, dtype=np.float32).reshape(-1)[: self.joint_position_dim],
                        np.asarray(gripper, dtype=np.float32).reshape(-1)[:1],
                    )
        raise ValueError(
            "DPPolicy action adapter requires `state.robot.joint_position` (or "
            "`joint_positions`) and `state.robot.gripper_position`."
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ensure_dppo_repo_on_path(self.repo_path)

        resolved_checkpoint = checkpoint_path
        resolved_config = self.config_path
        resolved_norm = self.normalization_path

        if self.policy_alias is not None:
            alias_paths = load_policy_paths_from_alias(
                self.policy_alias,
                self.policy_json,
            )
            resolved_config = resolved_config or alias_paths["config_path"]
            resolved_checkpoint = alias_paths["checkpoint_path"]
            resolved_norm = resolved_norm or alias_paths["norm_stats"]

        if resolved_config is None:
            raise ValueError(
                "DPPolicy requires `config_path` or `policy_alias` to load a checkpoint."
            )
        if resolved_norm is None:
            raise ValueError(
                "DPPolicy requires `normalization_path` or `policy_alias` to load a checkpoint."
            )

        self._policy = load_policy_from_checkpoint(
            config_path=resolved_config,
            checkpoint_path=resolved_checkpoint,
            normalization_stats_path=resolved_norm,
            device=self.device,
            repo_path=self.repo_path,
            ordered_obs_keys=self.ordered_obs_keys,
            camera_indices=self.camera_indices,
            act_steps=self.act_steps,
        )
        self._pending_actions = []

    def _build_dp_observation(self, observation: Any, state: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"robot_state": self._build_robot_state_dict(state)}

        if self._policy is not None and getattr(self._policy, "camera_indices", []):
            payload["image"] = self._build_image_dict(observation)

        return payload

    def _build_robot_state_dict(self, state: Any) -> dict[str, np.ndarray]:
        if isinstance(state, dict):
            if "robot_state" in state and isinstance(state["robot_state"], dict):
                return {
                    key: self._as_row_vector(value)
                    for key, value in state["robot_state"].items()
                }

            if "robot" in state and isinstance(state["robot"], dict):
                robot_state = state["robot"]
                # Prefer explicit joint_positions/gripper_position if available
                if self.ordered_obs_keys:
                    extracted = {}
                    for key in self.ordered_obs_keys:
                        if key in robot_state:
                            extracted[key] = self._as_row_vector(robot_state[key])
                    if extracted:
                        return extracted
                if "state" in robot_state:
                    return self._vector_to_robot_state(robot_state["state"])

            if "state" in state:
                return self._vector_to_robot_state(state["state"])

            if self.ordered_obs_keys:
                extracted = {}
                for key in self.ordered_obs_keys:
                    if key in state:
                        extracted[key] = self._as_row_vector(state[key])
                if extracted:
                    return extracted

        return self._vector_to_robot_state(state)

    def _vector_to_robot_state(self, value: Any) -> dict[str, np.ndarray]:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if self.ordered_obs_keys == ["state"]:
            return {"state": vector.reshape(1, -1)}

        if vector.size >= 2:
            return {
                "joint_positions": vector[:-1].reshape(1, -1),
                "gripper_position": vector[-1:].reshape(1, -1),
                "state": vector.reshape(1, -1),
            }

        return {"state": vector.reshape(1, -1)}

    def _build_image_dict(self, observation: Any) -> dict[int, np.ndarray]:
        resolved_views = self._resolve_views(observation)
        return {
            camera_idx: self._to_bgr_float32(
                self._resize_to_square(resolved_views[view_name])
            )
            for camera_idx, view_name in zip(
                getattr(self._policy, "camera_indices", self.camera_indices or []),
                self.view_names,
            )
        }

    @staticmethod
    def _resize_to_square(image: np.ndarray, size: int = 192) -> np.ndarray:
        """Resize image to square if it isn't already (policy expects square input)."""
        h, w = image.shape[:2]
        if h == w:
            return image
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    def _resolve_views(self, observation: Any) -> dict[str, np.ndarray]:
        if isinstance(observation, dict):
            if "views" in observation and isinstance(observation["views"], dict):
                return {
                    name: self._load_image(value)
                    for name, value in observation["views"].items()
                }
            direct_views = {
                name: self._load_image(value)
                for name, value in observation.items()
                if isinstance(value, (str, np.ndarray, list, tuple))
            }
            if direct_views:
                return direct_views

        image = self._load_image(observation)
        if image.ndim != 3:
            raise ValueError(f"Unsupported observation shape for DPPolicy: {image.shape}")

        if image.shape[0] % len(self.stacked_view_order) != 0:
            return {self.view_names[0]: image}

        split_height = image.shape[0] // len(self.stacked_view_order)
        return {
            view_name: image[index * split_height : (index + 1) * split_height]
            for index, view_name in enumerate(self.stacked_view_order)
        }

    def _load_image(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, str):
            path = Path(value)
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        raise ValueError(f"Unsupported image value for DPPolicy: {type(value)!r}")

    @staticmethod
    def _to_bgr_float32(image: np.ndarray) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
        return rgb[..., ::-1].copy()

    @staticmethod
    def _as_row_vector(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(1, -1)
        return array
