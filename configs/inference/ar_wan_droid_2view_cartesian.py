"""INFERENCE config: AR Wan2.1-1.3B DROID, **2-view** + **cartesian** conditioning.

Inference-only counterpart of the training recipe in
``configs/training/ar_wan_droid.py`` (which is itself 2-view + cartesian). Use this
for teleoperation (``scripts/interactive_ar.py``), open-loop replay
(``openworld/autoregressive/infer/replay.py``), and eval -- anything that just loads
a trained student and rolls it out.

Only the knobs that change *model/data geometry at inference* are pinned here:
  * ``num_cams=2``         -- 1 sampled side view + wrist, height-stacked
  * ``action_space="cartesian"`` -- 7-dim EEF pose (xyz + axis-angle + gripper),
                              read from ``stats.json`` (action_dim=7; see
                              openworld/autoregressive/config.py ACTION_SPACES)

The trained weights come from the ``--checkpoint`` flag, NOT from this config; the
inherited training-only fields (lr, distillation schedule, student_init/teacher
ckpt paths) are irrelevant to a forward-only rollout and are left untouched.

    python scripts/interactive_ar.py \
        --config configs/inference/ar_wan_droid_2view_cartesian.py \
        --checkpoint <trained_2view_cartesian_student.pt>
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(),
        num_cams=2,
        action_space="cartesian",                  # -> stats.json, action_dim=7
        tag="ar_wan_droid_infer_2view_cartesian",
    )
