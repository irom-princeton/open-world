"""Wrapper for the MolmoAct2 policy (allenai/molmoact2)."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from openworld.policies.base_policy import Policy
from openworld.policies.openpi_action_adapter import get_fk_solution
from openworld.policies.molmoact2_loader import (
    DEFAULT_ACTION_MODE,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MOLMOACT2_REPO,
    DEFAULT_NORM_TAG,
    DEFAULT_NUM_STEPS,
    MolmoAct2HttpClient,
    load_policy_from_checkpoint,
    parse_server_url,
)

logger = logging.getLogger(__name__)


class MolmoAct2Policy(Policy):
    """Adapter around an in-process or HTTP-backed MolmoAct2 policy.

    Mirrors :class:`OpenPIPolicy`: choose between
      * HTTP client mode (``server_url``) for talking to
        ``host_server_droid.py``, or
      * in-process mode (``checkpoint_path`` / ``hf_repo_id``) for loading
        the Hugging Face model directly.

    The DROID checkpoint outputs absolute joint positions + gripper (8-D);
    we convert to a 7-D cartesian + gripper action via forward kinematics
    before handing it to the world model (matching :class:`DPPolicy`).
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        *,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        repo_path: Optional[str] = str(DEFAULT_MOLMOACT2_REPO),
        dtype: str = "bfloat16",
        device: str = "cuda",
        default_prompt: Optional[str] = None,
        exterior_view_name: str = "exterior_left",
        wrist_view_name: str = "wrist",
        stacked_view_order: Optional[list[str]] = None,
        norm_tag: str = DEFAULT_NORM_TAG,
        action_mode: str = DEFAULT_ACTION_MODE,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_depth_reasoning: bool = False,
        normalize_language: bool = True,
        joint_position_dim: int = 7,
        http_timeout: float = 60.0,
        enable_cuda_graph: Optional[bool] = None,
        debug: bool = False,
        debug_log_limit: int = 3,
        **_: Any,
    ):
        self.server_url = server_url
        self.hf_repo_id = hf_repo_id
        self.repo_path = repo_path
        self.dtype = dtype
        self.device = device
        self.default_prompt = default_prompt
        self.exterior_view_name = exterior_view_name
        self.wrist_view_name = wrist_view_name
        self.stacked_view_order = list(
            stacked_view_order or ["exterior_left", "exterior_right", "wrist"]
        )
        self.norm_tag = norm_tag
        self.action_mode = action_mode
        self.num_steps = num_steps
        self.enable_depth_reasoning = enable_depth_reasoning
        self.normalize_language = normalize_language
        self.joint_position_dim = joint_position_dim
        self.http_timeout = http_timeout
        self.enable_cuda_graph = enable_cuda_graph
        self.debug = debug
        self.debug_log_limit = debug_log_limit

        self._instruction: Optional[str] = None
        self._policy: Any = None
        self._pending_actions: list[dict[str, Any]] = []
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
                    "MolmoAct2Policy.act() requires either `checkpoint_path` for local "
                    "in-process loading or `params.server_url` for HTTP-client mode."
                )
            self._policy = self._build_http_client(self.server_url)

        if not self._pending_actions:
            payload = self._build_payload(
                observation=observation,
                state=state,
                instruction=instruction,
            )
            result = self._policy.infer(payload)
            predicted = np.asarray(result["actions"], dtype=np.float32)
            if predicted.ndim == 1:
                predicted = predicted[np.newaxis, :]
            self._pending_actions = [
                self._action_to_env_format(action) for action in predicted
            ]
            self._debug_log_inference(payload, predicted, state)

        if not self._pending_actions:
            raise RuntimeError("MolmoAct2 policy produced an empty action sequence.")

        return self._pending_actions.pop(0)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        # ``checkpoint_path`` is accepted for parity with the other policies;
        # if it looks like a HF repo id we honor it, otherwise fall back to
        # ``hf_repo_id``.
        resolved_id = checkpoint_path or self.hf_repo_id
        self._policy = load_policy_from_checkpoint(
            hf_repo_id=resolved_id,
            dtype=self.dtype,
            device=self.device,
            norm_tag=self.norm_tag,
            action_mode=self.action_mode,
            num_steps=self.num_steps,
            enable_depth_reasoning=self.enable_depth_reasoning,
            normalize_language=self.normalize_language,
            enable_cuda_graph=bool(self.enable_cuda_graph),
            repo_path=self.repo_path,
        )
        self._pending_actions = []
        if self.debug:
            logger.info(
                "MolmoAct2 local policy loaded: id=%s dtype=%s device=%s",
                resolved_id,
                self.dtype,
                self.device,
            )

    def _build_http_client(self, server_url: str) -> MolmoAct2HttpClient:
        host, port = parse_server_url(server_url)
        return MolmoAct2HttpClient(host=host, port=port, timeout=self.http_timeout)

    def _build_payload(
        self,
        *,
        observation: Any,
        state: Any,
        instruction: Optional[str],
    ) -> dict[str, Any]:
        views = self._resolve_views(observation)
        external_cam = self._prepare_image(views[self.exterior_view_name])
        wrist_cam = self._prepare_image(views[self.wrist_view_name])
        state_vector = self._build_state_vector(state)

        prompt = instruction or self._instruction or self.default_prompt or ""

        payload: dict[str, Any] = {
            "external_cam": external_cam,
            "wrist_cam": wrist_cam,
            "instruction": prompt,
            "state": state_vector,
            "num_steps": self.num_steps,
        }
        if self.enable_cuda_graph is not None:
            payload["enable_cuda_graph"] = self.enable_cuda_graph
        return payload

    def _build_state_vector(self, state: Any) -> np.ndarray:
        """Return an 8-D ``[q1..q7, gripper]`` float32 vector."""
        if isinstance(state, dict):
            robot = state.get("robot") if isinstance(state.get("robot"), dict) else None
            container = robot if robot is not None else state
            joint = self._first_present(container, "joint_position", "joint_positions")
            gripper = container.get("gripper_position")
            if joint is not None:
                joint_vec = self._fit_joint_position(joint)
                gripper_vec = self._coerce_gripper(gripper, joint_vec)
                return np.concatenate([joint_vec, gripper_vec], axis=0).astype(
                    np.float32, copy=False
                )
            if "state" in container:
                return self._vector_to_state(container["state"])
        return self._vector_to_state(state)

    def _vector_to_state(self, value: Any) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("MolmoAct2 state vector cannot be empty.")
        target_size = self.joint_position_dim + 1
        if vector.size == target_size:
            return vector
        joint = self._fit_joint_position(vector)
        gripper = vector[-1:].astype(np.float32, copy=False)
        return np.concatenate([joint, gripper], axis=0).astype(np.float32, copy=False)

    def _fit_joint_position(self, value: Any) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
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

    def _coerce_gripper(self, gripper: Any, fallback: np.ndarray) -> np.ndarray:
        if gripper is not None:
            return np.asarray(gripper, dtype=np.float32).reshape(-1)[:1]
        return fallback[-1:].astype(np.float32, copy=False)

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
        # Single image fallback: route it to every view we need.
        return {
            self.exterior_view_name: image,
            self.wrist_view_name: image,
        }

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Return a contiguous uint8 ``(H, W, 3)`` RGB array.

        MolmoAct2's processor handles resizing internally, so we only enforce
        dtype/shape here.
        """
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _load_image(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        if isinstance(value, str):
            path = Path(value)
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
        raise ValueError(f"Unsupported image value for MolmoAct2Policy: {type(value)!r}")

    @staticmethod
    def _action_to_env_format(action: np.ndarray) -> dict[str, Any]:
        """Convert MolmoAct2 action (joint positions + gripper) to env format.

        DROID checkpoint outputs absolute joint positions (7) + gripper (1).
        The world model expects 7-D cartesian (xyz + euler + gripper).
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        joint_pos = action[:7]
        gripper_pos = action[7:8]

        fk = get_fk_solution(joint_pos)
        xyz = fk[:3, 3].astype(np.float32)
        euler = R.from_matrix(fk[:3, :3]).as_euler("xyz").astype(np.float32)
        cartesian_action = np.concatenate([xyz, euler, gripper_pos], axis=0)

        return {
            "env_action": cartesian_action,
            "state_update": {
                "robot": {
                    "state_representation": "cartesian_position_with_gripper",
                    "state": cartesian_action,
                    "cartesian_position": cartesian_action[:6],
                    "joint_position": joint_pos,
                    "joint_positions": joint_pos,
                    "gripper_position": gripper_pos,
                }
            },
        }

    def _debug_log_inference(
        self,
        payload: dict[str, Any],
        predicted: np.ndarray,
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
        logger.info(
            "MolmoAct2 debug[%d]: id=%s prompt=%r external_shape=%s wrist_shape=%s state_shape=%s "
            "raw_action_chunk_shape=%s raw_action_min=%.4f raw_action_max=%.4f first_raw_action=%s state_summary=%s",
            self._debug_logs_emitted,
            self.hf_repo_id if self.server_url is None else self.server_url,
            payload.get("instruction"),
            _shape(payload["external_cam"]),
            _shape(payload["wrist_cam"]),
            _shape(payload["state"]),
            tuple(predicted.shape),
            float(predicted.min()),
            float(predicted.max()),
            np.array2string(first_raw, precision=4, suppress_small=True),
            self._summarize_state(state),
        )
        self._debug_logs_emitted += 1

    @staticmethod
    def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _summarize_state(state: Any) -> str:
        if isinstance(state, dict):
            parts = [f"keys={sorted(state.keys())}"]
            robot = state.get("robot")
            if isinstance(robot, dict):
                parts.append(f"robot_keys={sorted(robot.keys())}")
                if "state_representation" in robot:
                    parts.append(f"state_rep={robot['state_representation']}")
            return " ".join(parts)
        return type(state).__name__
