"""Stage L2a -- causal student-init mid-training, Wan-1.3B on DROID.

Plain flow-matching fine-tuning of the base Wan backbone with **block-causal**
(chunk2) attention -- the same masking the few-step student uses at rollout. Its
final checkpoint initializes the self-forcing generator (``student_init_ckpt`` in
ar_wan_droid.py). Independent of the teacher stage; run them in parallel.

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_midtrain \
        --config configs/training/ar_wan_studentinit_droid.py

Step count is scaled for the DROID set (~23k clips); omni-dreams used 150k. Tune
from the loss curve.
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid import get_args as _l0


def get_args():
    return dataclasses.replace(
        _l0(),
        stage="student_init",          # -> block-causal attention (ARWMArgs.stage_is_causal)
        tag="ar_wan_studentinit",
        max_train_steps=40_000,        # omni-dreams: 150_000
        log_samples=True,              # causal -> AR rollout previews are meaningful
    )
