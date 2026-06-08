"""Stage L2a student-init, action_cond_mode="cross_attn_pe" (Fix 1).

Same as ar_wan_studentinit_droid.py but adds a learned temporal positional
embedding to the per-frame action tokens so cross-attention can learn the
action->frame alignment. Pair with ar_wan_teacher_droid_pe.py (the teacher must
use the same mode -- L0 self-forcing feeds the generator's cond to the teacher).

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_studentinit_droid_pe.py
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_studentinit_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(), action_cond_mode="cross_attn_pe", tag="ar_wan_studentinit_pe",
    )
