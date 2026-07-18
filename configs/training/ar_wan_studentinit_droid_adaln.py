"""Stage L2a student-init, action_cond_mode="adaln" (Fix 3).

Same as ar_wan_studentinit_droid.py but the action drops the text cross-attention
entirely and instead modulates the per-frame AdaLN time-embedding (a clean
injection-site ablation vs the cross-attn modes). Pair with
ar_wan_teacher_droid_adaln.py.

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_studentinit_droid_adaln.py
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_studentinit_droid import get_args as _base


def get_args():
    return dataclasses.replace(
        _base(), action_cond_mode="adaln", tag="ar_wan_studentinit_adaln",
    )
