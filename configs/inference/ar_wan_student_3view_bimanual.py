"""INFERENCE config: AR Wan2.1-1.3B student, **3-view bimanual** + **cartesian**
conditioning with **camera_cond** geometry and an auxiliary joint state-prediction
head (the ``wm_student_3view_bimanual`` checkpoint).

This is the camera_cond ("camcond v2 + context-noise") student. Its patch-embed conv
takes **25 input channels** (16 base latent + 9 camera_cond: 3 trajectory-band + 6
ray-map), so it is NOT loadable with a plain 16-channel config, and camera_cond
geometry MUST be provided at inference. The geometry matches what the checkpoint was
trained with, so the weights load with no missing/unexpected keys:

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
  * ``camera_cond=True, camera_cond_channels=9`` -- 3 band + 6 ray-map extra input
                                    channels (widens the patch-embed conv 16 -> 25).
                                    ``camera_cond_band=True`` (trajectory band on),
                                    ``camera_cond_wrist_band=True`` (band on the wrist
                                    views too), ``camera_cond_sticks=False`` (no
                                    orientation spokes).

This is an **undistilled** student: sample with the many-step preview schedule
(do NOT pass ``--distilled``).

Replay / eval of this checkpoint needs matching preprocessed latents (all stored views,
so ``view_indices`` can select 1,2,3), their 20-dim ``stats.json``, AND a
``{split}_camera_cond.npy`` geometry sidecar (per-episode pose / c2w / K / band_valid).
For closed-loop (FK-synthesized) camera_cond, additionally provide a
``{split}_joint_actions.npy`` sidecar and pass ``--conditioning action``. Teleoperation
needs 3-view + 20-dim initialization stills plus camera_cond geometry (the bundled DROID
inits in ``assets/teleop_inits/`` are 2-view / 7-dim and do not fit this checkpoint).

    python scripts/replay_ar.py \
        --config configs/inference/ar_wan_student_3view_bimanual.py \
        --checkpoint checkpoints/ar_wm/wm_student_3view_bimanual.pt \
        --latent-root <bimanual_latents> --split val --conditioning episode
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
        # camera_cond geometry: 9 extra input channels (3 band + 6 ray-map) ->
        # patch-embed widens 16 -> 25 to match the checkpoint's patch_embedding.weight.
        camera_cond=True,
        camera_cond_channels=9,
        camera_cond_band=True,
        camera_cond_wrist_band=True,               # band drawn on the wrist views too
        camera_cond_sticks=False,                  # no orientation spokes
        tag="ar_wan_student_3view_bimanual_camcond_infer",
    )
