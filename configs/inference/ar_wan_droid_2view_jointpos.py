"""INFERENCE config: AR Wan2.1-1.3B DROID, **2-view** + **joint_pos** conditioning.

Inference-only config for a checkpoint trained with 8-dim joint-position
conditioning (e.g. the ``*_2view_*_joint`` training recipes). Use for teleop
(``scripts/interactive_ar.py``), replay, and eval.

Inference-relevant knobs:
  * ``num_cams=2``           -- 1 sampled side view + wrist, height-stacked
  * ``action_space="joint_pos"`` -- 7 joint positions + gripper = 8 dims, read from
                                ``<latent_root>/<split>_joint_actions.npy`` +
                                ``stats_joint.json``. ``action_dim`` auto-bumps
                                7 -> 8 in ARWMArgs.__post_init__.

Trained weights come from ``--checkpoint``; inherited training-only fields are
irrelevant to a forward-only rollout.

    python scripts/interactive_ar.py \
        --config configs/inference/ar_wan_droid_2view_jointpos.py \
        --checkpoint <trained_2view_jointpos_student.pt>
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(),
        num_cams=2,
        action_space="joint_pos",                  # -> stats_joint.json, action_dim=8
        tag="ar_wan_droid_infer_2view_jointpos",
    )
