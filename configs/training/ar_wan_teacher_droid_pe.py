"""Stage L1b teacher, action_cond_mode="cross_attn_pe" (Fix 1).

Teacher counterpart of ar_wan_studentinit_droid_pe.py -- must share the action
mode (L0 self-forcing feeds the generator's cond to this teacher/critic).

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_teacher_droid_pe.py
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_teacher_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(), action_cond_mode="cross_attn_pe", tag="ar_wan_teacher_pe",
    )
