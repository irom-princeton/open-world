"""Wrapper for the OpenPI policy (Physical-Intelligence/openpi)."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from openworld.policies.base_policy import Policy
from openworld.policies.openpi_action_adapter import AdaptedActionChunk, OpenPIActionAdapter
from openworld.policies.openpi_loader import (
    DEFAULT_OPENPI_REPO,
    ensure_openpi_repo_on_path,
    load_policy_from_checkpoint,
)

logger = logging.getLogger(__name__)


class OpenPIPolicy(Policy):
    """Adapter around an in-process or websocket-backed OpenPI policy."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        *,
        config_name: str = "pi05_droid",
        repo_path: Optional[str] = str(DEFAULT_OPENPI_REPO),
        default_prompt: Optional[str] = None,
        pytorch_device: Optional[str] = None,
        exterior_view_name: str = "exterior_left",
        wrist_view_name: str = "wrist",
        stacked_view_order: Optional[list[str]] = None,
        intermediate_resize_height: Optional[int] = None,
        policy_skip_step: int = 1,
        num_action_steps: Optional[int] = None,
        resize_height: int = 224,
        resize_width: int = 224,
        joint_position_dim: int = 7,
        action_dim: Optional[int] = None,
        action_indices: Optional[list[int]] = None,
        action_adapter_checkpoint_path: Optional[str] = None,
        action_adapter_gripper_max: float = 1.0,
        debug: bool = False,
        debug_log_limit: int = 3,
        **_: Any,
    ):
        self.server_url = server_url
        self.config_name = config_name
        self.repo_path = repo_path
        self.default_prompt = default_prompt
        self.pytorch_device = pytorch_device
        self.exterior_view_name = exterior_view_name
        self.wrist_view_name = wrist_view_name
        self.stacked_view_order = list(
            stacked_view_order or ["exterior_right", "exterior_left", "wrist"]
        )
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.joint_position_dim = joint_position_dim
        self.action_dim = action_dim
        self.action_indices = list(action_indices) if action_indices is not None else None
        self.action_adapter_checkpoint_path = action_adapter_checkpoint_path
        self.action_adapter_gripper_max = action_adapter_gripper_max
        self.intermediate_resize_height = intermediate_resize_height
        self.policy_skip_step = policy_skip_step
        self.num_action_steps = num_action_steps
        self.debug = debug
        self.debug_log_limit = debug_log_limit

        self._instruction: Optional[str] = None
        self._policy = None
        self._action_adapter: Optional[OpenPIActionAdapter] = None
        self._pending_actions: list[Any] = []
        self._debug_logs_emitted = 0

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = instruction
        self._pending_actions = []
        self._debug_logs_emitted = 0

    def act(
        self,
        observation: Any,
        state: Any,
        instruction: Optional[str] = None,
    ) -> Any:
        if self._policy is None:
            if self.server_url is None:
                raise RuntimeError(
                    "OpenPIPolicy.act() requires either `checkpoint_path` for local "
                    "in-process loading or `params.server_url` for websocket mode."
                )
            self._policy = self._build_websocket_policy(self.server_url)

        if not self._pending_actions:
            openpi_observation = self._build_openpi_observation(
                observation=observation,
                state=state,
                instruction=instruction,
            )
            result = self._policy.infer(openpi_observation)
            predicted = np.asarray(result["actions"], dtype=np.float32)
            if predicted.ndim == 1:
                predicted = predicted[np.newaxis, :]
            adapted_actions = self._adapt_action_chunk(predicted, state)
            self._debug_log_inference(openpi_observation, predicted, adapted_actions, state)
            self._pending_actions = adapted_actions

        if not self._pending_actions:
            raise RuntimeError("OpenPI policy produced an empty action sequence.")

        return self._pending_actions.pop(0)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._policy = load_policy_from_checkpoint(
            config_name=self.config_name,
            checkpoint_path=checkpoint_path,
            repo_path=self.repo_path,
            default_prompt=self.default_prompt,
            pytorch_device=self.pytorch_device,
        )
        self._pending_actions = []
        if self.debug:
            logger.info(
                "OpenPI local policy loaded: config=%s checkpoint=%s repo=%s device=%s",
                self.config_name,
                checkpoint_path,
                self.repo_path,
                self.pytorch_device,
            )

    def _build_websocket_policy(self, server_url: str) -> Any:
        ensure_openpi_repo_on_path(self.repo_path)

        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        parsed = urlparse(server_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Unsupported OpenPI server URL: {server_url}")
        if parsed.hostname is None:
            raise ValueError(f"OpenPI server URL is missing a host: {server_url}")

        return WebsocketClientPolicy(
            host=parsed.hostname,
            port=parsed.port or 8000,
        )

    def _build_openpi_observation(
        self,
        *,
        observation: Any,
        state: Any,
        instruction: Optional[str],
    ) -> dict[str, Any]:
        views = self._resolve_views(observation)
        joint_position, gripper_position = self._build_state_inputs(state)
        prompt = instruction or self._instruction or self.default_prompt

        payload: dict[str, Any] = {
            "observation/exterior_image_1_left": self._prepare_image(
                views[self.exterior_view_name]
            ),
            "observation/wrist_image_left": self._prepare_image(
                views[self.wrist_view_name]
            ),
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        return payload

    def _build_state_inputs(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(state, dict):
            if "robot" in state and isinstance(state["robot"], dict):
                robot_state = state["robot"]
                joint_position = self._first_present(
                    robot_state,
                    "joint_position",
                    "joint_positions",
                )
                gripper_position = robot_state.get("gripper_position")
                if joint_position is not None:
                    return (
                        self._fit_joint_position(joint_position),
                        self._coerce_gripper_position(gripper_position, joint_position),
                    )
                if "state" in robot_state:
                    return self._vector_to_openpi_state(robot_state["state"])

            joint_position = self._first_present(
                state,
                "joint_position",
                "joint_positions",
            )
            gripper_position = state.get("gripper_position")
            if joint_position is not None:
                return (
                    self._fit_joint_position(joint_position),
                    self._coerce_gripper_position(gripper_position, joint_position),
                )

            if "state" in state:
                return self._vector_to_openpi_state(state["state"])

        return self._vector_to_openpi_state(state)

    def _vector_to_openpi_state(self, value: Any) -> tuple[np.ndarray, np.ndarray]:
        vector = self._as_vector(value)
        if vector.size == 0:
            raise ValueError("OpenPI state vector cannot be empty.")
        joint_position = self._fit_joint_position(vector)
        gripper_position = vector[-1:].astype(np.float32, copy=False)
        return joint_position, gripper_position

    def _fit_joint_position(self, value: Any) -> np.ndarray:
        vector = self._as_vector(value)
        if vector.size == self.joint_position_dim:
            return vector
        if vector.size > self.joint_position_dim:
            return vector[: self.joint_position_dim]
        if vector.size == 0:
            raise ValueError("Joint position vector cannot be empty.")
        pad_value = vector[-1]
        padding = np.full(
            (self.joint_position_dim - vector.size,),
            pad_value,
            dtype=np.float32,
        )
        return np.concatenate([vector, padding], axis=0)

    def _adapt_action(self, action: np.ndarray) -> np.ndarray:
        adapted = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_indices is not None:
            adapted = adapted[self.action_indices]
        elif self.action_dim is not None and adapted.size != self.action_dim:
            adapted = adapted[: self.action_dim]
        return adapted

    def _adapt_action_chunk(self, predicted: np.ndarray, state: Any) -> list[Any]:
        if self.action_adapter_checkpoint_path is None:
            return [self._adapt_action(action) for action in predicted]

        adapter = self._get_action_adapter()
        joint_position, gripper_position = self._extract_joint_state(state)
        adapted = adapter.adapt(joint_position, gripper_position, predicted)

        # Full control-rate absolute joint+gripper trajectory from the dynamics
        # model (model2_15_9), BEFORE the policy_skip_step subsample. Joint-space
        # world models (WEAVER) use this to subsample at their own video rate
        # (RGB_SKIP) and compute relabel-style position deltas. Cartesian models
        # (ctrl-world/AR) ignore it.
        pi_joint_traj = np.concatenate(
            [adapted.joint_positions, adapted.gripper_positions], axis=-1
        ).astype(np.float32)                                     # [action_num, 8]

        # Subsample to match world model temporal resolution
        if self.policy_skip_step > 1:
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
                            "_pi_joint_traj": pi_joint_traj,
                        }
                    },
                }
            )
        return step_actions

    def _extract_joint_state(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(state, dict):
            robot = state.get("robot")
            if isinstance(robot, dict):
                joint_position = self._first_present(robot, "joint_position", "joint_positions")
                gripper_position = robot.get("gripper_position")
                if joint_position is not None and gripper_position is not None:
                    return self._fit_joint_position(joint_position), self._coerce_gripper_position(
                        gripper_position,
                        joint_position,
                    )
        joint_position, gripper_position = self._build_state_inputs(state)
        return joint_position, gripper_position

    def _get_action_adapter(self) -> OpenPIActionAdapter:
        if self._action_adapter is None:
            self._action_adapter = OpenPIActionAdapter(
                checkpoint_path=self.action_adapter_checkpoint_path,
                gripper_max=self.action_adapter_gripper_max,
                device=self.pytorch_device,
            )
        return self._action_adapter

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
        if image.ndim == 3 and image.shape[0] % len(self.stacked_view_order) == 0:
            split_height = image.shape[0] // len(self.stacked_view_order)
            return {
                view_name: image[index * split_height : (index + 1) * split_height]
                for index, view_name in enumerate(self.stacked_view_order)
            }
        return {
            self.exterior_view_name: image,
            self.wrist_view_name: image,
        }

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Intermediate resize (e.g. 192x320 → 180x320) to match DROID policy preprocessing before pad-to-square
        if self.intermediate_resize_height is not None:
            h, w = rgb.shape[:2]
            target_h = self.intermediate_resize_height
            if h != target_h:
                t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
                t = F.interpolate(t, size=(target_h, w), mode="bilinear", align_corners=False)
                rgb = t.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

        pil_image = Image.fromarray(rgb)
        if pil_image.size != (self.resize_width, self.resize_height):
            pil_image = self._resize_with_pad(
                pil_image,
                height=self.resize_height,
                width=self.resize_width,
            )
        return np.asarray(pil_image)

    def _load_image(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, str):
            path = Path(value)
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        raise ValueError(f"Unsupported image value for OpenPIPolicy: {type(value)!r}")

    @staticmethod
    def _as_vector(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        return array

    def _coerce_gripper_position(
        self,
        gripper_position: Any,
        fallback_value: Any,
    ) -> np.ndarray:
        if gripper_position is not None:
            return self._as_vector(gripper_position)
        fallback = self._as_vector(fallback_value)
        return fallback[-1:].astype(np.float32, copy=False)

    @staticmethod
    def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _resize_with_pad(image: Image.Image, *, height: int, width: int) -> Image.Image:
        current_width, current_height = image.size
        if (current_width, current_height) == (width, height):
            return image

        ratio = max(current_width / width, current_height / height)
        resized_width = int(current_width / ratio)
        resized_height = int(current_height / ratio)
        resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

        canvas = Image.new(resized.mode, (width, height), 0)
        pad_width = max(0, int((width - resized_width) / 2))
        pad_height = max(0, int((height - resized_height) / 2))
        canvas.paste(resized, (pad_width, pad_height))
        return canvas

    def _debug_log_inference(
        self,
        payload: dict[str, Any],
        predicted: np.ndarray,
        adapted_actions: list[Any],
        state: Any,
    ) -> None:
        if not self.debug or self._debug_logs_emitted >= self.debug_log_limit:
            return

        def _shape(value: Any) -> Any:
            try:
                return tuple(np.asarray(value).shape)
            except Exception:
                return type(value).__name__

        first_raw = np.asarray(predicted[0], dtype=np.float32).reshape(-1)
        if adapted_actions and isinstance(adapted_actions[0], dict):
            first_adapted = np.asarray(adapted_actions[0]["env_action"], dtype=np.float32).reshape(-1)
        else:
            first_adapted = (
                np.asarray(adapted_actions[0], dtype=np.float32).reshape(-1)
                if adapted_actions
                else np.asarray([], dtype=np.float32)
            )
        logger.info(
            "OpenPI debug[%d]: config=%s prompt=%r joint_shape=%s gripper_shape=%s ext_shape=%s wrist_shape=%s raw_action_chunk_shape=%s raw_action_min=%.4f raw_action_max=%.4f adapted_action_shape=%s action_indices=%s adapter=%s first_raw_action=%s first_adapted_action=%s state_summary=%s",
            self._debug_logs_emitted,
            self.config_name,
            payload.get("prompt"),
            _shape(payload["observation/joint_position"]),
            _shape(payload["observation/gripper_position"]),
            _shape(payload["observation/exterior_image_1_left"]),
            _shape(payload["observation/wrist_image_left"]),
            tuple(predicted.shape),
            float(predicted.min()),
            float(predicted.max()),
            tuple(first_adapted.shape),
            self.action_indices,
            bool(self.action_adapter_checkpoint_path),
            np.array2string(first_raw, precision=4, suppress_small=True),
            np.array2string(first_adapted, precision=4, suppress_small=True),
            self._summarize_state(state),
        )
        self._debug_logs_emitted += 1

    @staticmethod
    def _summarize_state(state: Any) -> str:
        if isinstance(state, dict):
            parts = [f"keys={sorted(state.keys())}"]
            robot = state.get("robot")
            if isinstance(robot, dict):
                parts.append(f"robot_keys={sorted(robot.keys())}")
                if "state_representation" in robot:
                    parts.append(f"state_rep={robot['state_representation']}")
                if "state" in robot:
                    parts.append(f"robot_state_shape={tuple(np.asarray(robot['state']).shape)}")
            if "state" in state:
                parts.append(f"state_shape={tuple(np.asarray(state['state']).shape)}")
            if "current_latent" in state:
                parts.append(f"latent_shape={tuple(state['current_latent'].shape)}")
            return " ".join(parts)
        return type(state).__name__
