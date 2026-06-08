"""Stage L2a student-init, action_cond_mode="cross_attn_aligned" (Fix 2).

Same as ar_wan_studentinit_droid.py but latent frame f cross-attends ONLY to its
own action token f (per-frame block-diagonal cross-attn mask in training; one
action per block in the rollout). Strongest action->frame binding. Pair with
ar_wan_teacher_droid_aligned.py.

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_studentinit_droid_aligned.py
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_studentinit_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(), action_cond_mode="cross_attn_aligned", tag="ar_wan_studentinit_aligned",
    )
