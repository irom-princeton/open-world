"""Short, fully-instrumented DMD diagnostic run (clean restart from the healthy
inits). Identical to ``ar_wan_droid_aligned_v2`` except:

  * ``tag`` -> new output dir (``ar_wan_dmd_aligned_v2_diag``); does NOT touch the
    real v2 run's dir or resume its state.
  * ``max_train_steps=80`` and ``log_every_steps=1`` -- log every step so we can
    watch gen_x0_std / dmd_grad_norm / gen_grad_norm / critic_grad_norm and see
    exactly when/how the collapse happens.
  * ``save_resume_state=False`` -- no 34 GB training_state writes for a throwaway.
  * checkpoint/preview every 40 steps so we also get a decoded sample at 40 & 80.

This run carries the bug-A fix (DMD score cond sliced to the generated window),
so it doubles as a test of whether that fix alone keeps the student healthy.
"""
from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned_v2 import get_args as _v2


def get_args():
    return dataclasses.replace(
        _v2(),
        tag="ar_wan_dmd_aligned_v2_diag",
        max_train_steps=80,
        log_every_steps=1,
        save_resume_state=False,
        checkpointing_steps=40,
        permanent_checkpoint_steps=80,
    )
