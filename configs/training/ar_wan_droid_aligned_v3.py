"""Stage L0 DMD distillation, aligned -- v3 (clean restart with the post-v2 fixes).

Inherits ``ar_wan_droid_aligned_v2`` (same aligned student-init + teacher inits,
whole-clip scoring, mean-reduced losses) and adds the two fixes the diagnostics
established:

  * bug-A cond alignment: the DMD score path now slices the action cond to the
    generated window (``slice_cond_to_frames`` in ``distill/self_forcing.py``), so
    the teacher/critic score generated frame j on action hist+j, not action j.

  * gen learning_rate 2e-6 -> 1e-6. The inherited 2e-6 came from the 64-GPU
    Self-Forcing / omni-dreams recipe and ran too hot for this 8-GPU
    under-batched setup: the generator outpaced the critic and eroded the
    student's learned per-view motion routing -- the fixed exterior cameras
    started drifting (action-coupled), the earliest symptom of the off-manifold
    drift that black-collapsed v2 by ~step 500. Validated with the step-80
    side-camera-motion probe: at 1e-6 the static cameras return to ~GT (0.9-1.0x)
    while wrist motion is preserved; CFG was ruled out as the lever (it only
    scales global motion gain). critic_lr stays 4e-7 (the validated pairing).

Checkpoints every 200 steps (kept), full per-step instrumentation logged every 20
(gen_x0_std / gen_x0_absmean / dmd_grad_norm / gen_grad_norm / critic_grad_norm).
"""
from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned_v2 import get_args as _v2


def get_args():
    return dataclasses.replace(
        _v2(),
        tag="ar_wan_dmd_aligned_v3",
        learning_rate=1e-6,                 # gen_lr; critic_learning_rate stays 4e-7
        checkpointing_steps=200,            # save (permanent) every 200 steps
        permanent_checkpoint_steps=200,     # == cadence -> every save is kept
        log_every_steps=20,
    )
