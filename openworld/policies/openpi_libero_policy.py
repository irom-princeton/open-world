"""OpenPI policy wrapper specialized for LIBERO.

Differences from :class:`openworld.policies.openpi_policy.OpenPIPolicy`:

* The openpi LIBERO inference key set is different from DROID
  (``observation/image``, ``observation/wrist_image``, ``observation/state``
  vs DROID's ``observation/exterior_image_1_left`` etc.) -- see
  ``external/openpi/examples/libero/main.py:130-141``.
* Images are rotated 180 degrees before being sent to the policy to match
  training (also from ``examples/libero/main.py:114-122``).
* The action adapter is :class:`OpenPILiberoActionAdapter` instead of the
  DROID one. Its outputs are ``env_actions`` (raw delta-EEF for the LIBERO
  env), ``eef_poses`` (absolute conditioning for the WM), and
  ``gripper_positions``.
* State is treated as a 6-D EEF pose (xyz + axis-angle) plus gripper, not
  as a 7-D joint vector.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from openworld.policies.libero_action_adapter import (
    AdaptedLiberoChunk,
    OpenPILiberoActionAdapter,
    resolve_initial_eef_state,
)
from openworld.policies.openpi_policy import OpenPIPolicy

logger = logging.getLogger(__name__)


class OpenPILiberoPolicy(OpenPIPolicy):
    """LIBERO-flavored OpenPI policy."""

    def __init__(
        self,
        *args: Any,
        # LIBERO adapter knobs (mirror parent's adapter knobs).
        libero_adapter_action_num: int = 15,
        libero_adapter_action_dim: int = 7,
        libero_adapter_hidden_size: int = 512,
        libero_agent_view_name: str = "agentview",
        libero_wrist_view_name: str = "wrist",
        rotate_180: bool = True,
        **kwargs: Any,
    ):
        # LIBERO uses just two cameras; tell the parent so split-image
        # fallbacks work.
        kwargs.setdefault("stacked_view_order", [libero_agent_view_name, libero_wrist_view_name])
        kwargs.setdefault("exterior_view_name", libero_agent_view_name)
        kwargs.setdefault("wrist_view_name", libero_wrist_view_name)
        # 6-D EEF + 1-D gripper; the parent uses joint_position_dim only for
        # padding, so set it to 6 here.
        kwargs.setdefault("joint_position_dim", 6)
        super().__init__(*args, **kwargs)

        self.libero_adapter_action_num = libero_adapter_action_num
        self.libero_adapter_action_dim = libero_adapter_action_dim
        self.libero_adapter_hidden_size = libero_adapter_hidden_size
        self.libero_agent_view_name = libero_agent_view_name
        self.libero_wrist_view_name = libero_wrist_view_name
        self.rotate_180 = rotate_180

    # ------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------

    def _get_action_adapter(self) -> OpenPILiberoActionAdapter:  # type: ignore[override]
        if self._action_adapter is None:
            if self.action_adapter_checkpoint_path is None:
                raise ValueError(
                    "OpenPILiberoPolicy requires action_adapter_checkpoint_path "
                    "(path to a LiberoDynamics .pth)."
                )
            self._action_adapter = OpenPILiberoActionAdapter(
                checkpoint_path=self.action_adapter_checkpoint_path,
                action_num=self.libero_adapter_action_num,
                action_dim=self.libero_adapter_action_dim,
                hidden_size=self.libero_adapter_hidden_size,
                gripper_max=self.action_adapter_gripper_max,
                device=self.pytorch_device,
            )
        return self._action_adapter

    def _extract_joint_state(self, state: Any) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        eef, grip = resolve_initial_eef_state(state)
        if eef is not None and grip is not None:
            return eef, grip
        # Fallback: parent path (numerical state vector). Joint padding
        # will be wrong but we won't use it -- the LIBERO adapter only
        # cares about EEF pose.
        return super()._extract_joint_state(state)

    def _adapt_action_chunk(self, predicted: np.ndarray, state: Any) -> list[Any]:  # type: ignore[override]
        if self.action_adapter_checkpoint_path is None:
            return [self._adapt_action(action) for action in predicted]

        adapter = self._get_action_adapter()
        eef_pose, gripper = self._extract_joint_state(state)
        adapted: AdaptedLiberoChunk = adapter.adapt(eef_pose, gripper, predicted)

        if self.policy_skip_step > 1:
            indices = list(range(0, adapted.env_actions.shape[0], self.policy_skip_step))
            if self.num_action_steps is not None:
                indices = indices[: self.num_action_steps]
            adapted = AdaptedLiberoChunk(
                env_actions=adapted.env_actions[indices],
                eef_poses=adapted.eef_poses[indices],
                gripper_positions=adapted.gripper_positions[indices],
            )

        step_actions: list[dict[str, Any]] = []
        for index in range(adapted.env_actions.shape[0]):
            env_action = adapted.env_actions[index]      # (7,) raw delta-EEF + gripper
            next_pose = adapted.eef_poses[index]         # (7,) absolute EEF + gripper
            next_gripper = adapted.gripper_positions[index]
            step_actions.append(
                {
                    "env_action": env_action,
                    "state_update": {
                        "robot": {
                            "state_representation": "eef_pose_with_gripper",
                            "state": next_pose,
                            "eef_pose": next_pose[:6],
                            "cartesian_position": next_pose[:6],
                            "gripper_position": next_gripper,
                        }
                    },
                }
            )
        return step_actions

    # ------------------------------------------------------------------
    # OpenPI observation packing -- LIBERO key names + 180-deg rotation.
    # ------------------------------------------------------------------

    def _build_openpi_observation(  # type: ignore[override]
        self,
        *,
        observation: Any,
        state: Any,
        instruction: Optional[str],
    ) -> dict[str, Any]:
        views = self._resolve_views(observation)
        agent = views.get(self.libero_agent_view_name)
        wrist = views.get(self.libero_wrist_view_name)
        if agent is None or wrist is None:
            raise KeyError(
                "LIBERO observations must expose both "
                f"'{self.libero_agent_view_name}' and '{self.libero_wrist_view_name}' views; "
                f"got {sorted(views)}"
            )

        if self.rotate_180:
            agent = np.ascontiguousarray(agent[::-1, ::-1])
            wrist = np.ascontiguousarray(wrist[::-1, ::-1])

        eef, grip = self._extract_joint_state(state)
        # The pi05_libero policy expects 8-D state: 6-D EEF (axisangle) +
        # 2-D gripper qpos. We provide eef[:6] + repeat(grip, 2) as a
        # safe approximation of the 2-finger qpos.
        if eef.size != 6:
            eef = eef[:6]
        if grip.size == 1:
            grip2 = np.array([float(grip[0]), float(grip[0])], dtype=np.float32)
        else:
            grip2 = grip[:2].astype(np.float32)
        state_vec = np.concatenate([eef.astype(np.float32), grip2], axis=0)

        prompt = instruction or self._instruction or self.default_prompt
        payload: dict[str, Any] = {
            "observation/image": self._prepare_image(agent),
            "observation/wrist_image": self._prepare_image(wrist),
            "observation/state": state_vec,
        }
        if prompt is not None:
            payload["prompt"] = prompt
        return payload


__all__ = ["OpenPILiberoPolicy"]
