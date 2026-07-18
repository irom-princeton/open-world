"""INFERENCE config: AR Wan2.1-1.3B student, **2-view** + **cartesian** conditioning
with an auxiliary joint state-prediction head (the ``wm_student_2view`` checkpoint).

Matches the geometry the published ``wm_student_2view.pt`` was trained with, so its
weights load with no missing/unexpected keys:
  * ``num_cams=2``               -- 1 sampled side view + wrist, height-stacked
  * ``action_space="cartesian"`` -- 7-dim absolute EEF pose (xyz + Euler-XYZ + gripper),
                                    normalized with ``stats.json``
  * ``action_cond_mode="cross_attn_aligned"`` -- inherited from ar_wan_droid.py
  * ``frames_per_block=1``       -- fully-causal single-frame blocks
  * ``num_history_blocks=4`` / ``rollout_blocks=12``
  * ``state_pred=True, state_pred_dim=8`` -- the aux joint state-prediction head. It is
                                    NOT used by the forward-only rollout, but the model
                                    must be built WITH it (``backbone.state_head.*``) or
                                    the checkpoint won't load cleanly.

This is an **undistilled** student: sample with the many-step preview schedule
(do NOT pass ``--distilled``, which is for a few-step distilled checkpoint).

    python scripts/interactive_ar.py \
        --config configs/inference/ar_wan_student_2view.py \
        --checkpoint checkpoints/ar_wm/wm_student_2view.pt
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(),
        num_cams=2,
        action_space="cartesian",                  # -> stats.json, action_dim=7
        frames_per_block=1,
        num_history_blocks=4,
        rollout_blocks=12,
        state_pred=True,
        state_pred_dim=8,
        tag="ar_wan_student_2view_infer",
    )
