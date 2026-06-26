"""INFERENCE config: AR Wan2.1-1.3B DROID, **3-view** + **cartesian** conditioning.

Inference-only counterpart of ``configs/training/ar_wan_droid_3view.py``. Use for
teleop (``scripts/interactive_ar.py``), replay, and eval of a checkpoint trained on
the full 3-view DROID layout (2 side views + wrist) with cartesian conditioning.

Inference-relevant knobs:
  * ``num_cams=3``         -- full DROID layout, all stored views height-stacked
  * ``action_space="cartesian"`` -- 7-dim EEF pose (xyz + axis-angle + gripper),
                              read from ``stats.json`` (action_dim=7)

Note: ``num_cams`` only changes how many height-stacked views the data path feeds,
not the model's parameters -- a checkpoint can be replayed at 2 or 3 views. This
config pins the 3-view geometry explicitly so the choice is recorded rather than
relying on the ``NUM_CAMS`` env override.

Trained weights come from ``--checkpoint``; inherited training-only fields are
irrelevant to a forward-only rollout.

    python scripts/interactive_ar.py \
        --config configs/inference/ar_wan_droid_3view_cartesian.py \
        --checkpoint <trained_3view_cartesian_student.pt>
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(),
        num_cams=3,
        action_space="cartesian",                  # -> stats.json, action_dim=7
        tag="ar_wan_droid_infer_3view_cartesian",
    )
