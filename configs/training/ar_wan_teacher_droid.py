"""Stage L1b -- bidirectional teacher mid-training, Wan-1.3B on DROID.

Plain flow-matching fine-tuning of the base Wan backbone with **full
bidirectional** attention. Its final checkpoint initializes both the self-forcing
real-score teacher and the fake-score critic (``teacher_ckpt`` in ar_wan_droid.py).
Independent of the student-init stage; run them in parallel.

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_teacher_droid.py

Step count scaled for DROID (~23k clips); omni-dreams used 150k. Tune from the
loss curve. Sample previews are off (AR rollout is not the teacher's generation
mode).
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _l0


def get_args():
    return dataclasses.replace(
        _l0(),
        stage="teacher",               # -> full bidirectional attention
        tag="ar_wan_teacher",
        max_train_steps=40_000,        # omni-dreams: 150_000
        log_samples=False,
    )
