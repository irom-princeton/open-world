"""INFERENCE config: AR Wan2.1-1.3B student, **3-view bimanual** + **cartesian**
conditioning with an auxiliary joint state-prediction head (the
``wm_student_3view_bimanual`` checkpoint).

Matches the geometry the published ``wm_student_3view_bimanual.pt`` was trained with,
so its weights load with no missing/unexpected keys:
  * ``view_indices=(1, 2, 3)``   -- the exact 3-view subset of the stored views
                                    (a scene cam + two wrist cams). This overrides the
                                    num_cams wrist+side sampling and forces num_cams=3.
  * ``action_space="cartesian", action_dim=20`` -- bimanual absolute pose, per arm
                                    xyz(3) + rot6d(6) + gripper(1), x2, normalized with
                                    ``stats.json``.
  * ``action_cond_mode="cross_attn_aligned"`` -- inherited from ar_wan_droid.py
  * ``frames_per_block=1``       -- fully-causal single-frame blocks
  * ``num_history_blocks=4`` / ``rollout_blocks=12``
  * ``state_pred=True, state_pred_dim=16`` -- the aux joint state-prediction head; not
                                    used by the forward-only rollout, but required for
                                    a clean checkpoint load (``backbone.state_head.*``).

This is an **undistilled** student: sample with the many-step preview schedule
(do NOT pass ``--distilled``).

Replay / eval of this checkpoint needs matching preprocessed latents (all stored views,
so ``view_indices`` can select 1,2,3) and their 20-dim ``stats.json``. Teleoperation
additionally needs 3-view + 20-dim initialization stills (the bundled DROID inits in
``assets/teleop_inits/`` are 2-view / 7-dim and do not fit this checkpoint).

    python scripts/replay_ar.py \
        --config configs/inference/ar_wan_student_3view_bimanual.py \
        --checkpoint checkpoints/ar_wm/wm_student_3view_bimanual.pt \
        --latent-root <bimanual_latents> --split val
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(),
        view_indices=(1, 2, 3),                    # -> forces num_cams=3
        action_space="cartesian",
        action_dim=20,                             # bimanual xyz+rot6d+grip, x2
        frames_per_block=1,
        num_history_blocks=4,
        rollout_blocks=12,
        state_pred=True,
        state_pred_dim=16,
        tag="ar_wan_student_3view_bimanual_infer",
    )
