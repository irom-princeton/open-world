"""Stage L1b -- bidirectional teacher mid-training, Cosmos-Predict2-2B on DROID.

Full bidirectional flow-matching fine-tuning of the base Cosmos backbone; its
final checkpoint initializes the self-forcing real-score teacher + critic. See
ar_wan_teacher_droid.py for the recipe; only the backbone differs.

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_cosmos_teacher_droid.py
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_cosmos_droid import get_args as _l0


def get_args():
    return dataclasses.replace(
        _l0(),
        stage="teacher",
        tag="ar_cosmos_teacher",
        max_train_steps=40_000,        # omni-dreams: 150_000
        log_samples=False,
    )
