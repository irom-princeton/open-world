"""Smaller-scale 4-GPU TEST of the v3 DMD distillation run.

Identical to ``ar_wan_droid_aligned_v3`` (same post-v2 fixes: bug-A cond
alignment + gen lr 1e-6) except for a fresh ``tag`` so this test writes to its
OWN output dir (``checkpoints/ar_wm/ar_wan_dmd_aligned_v3_4gpu``) and does NOT
collide with / get resumed by the queued 8-GPU ``ar_wan_dmd_aligned_v3`` job.

NOTE: at 4 GPUs the global batch halves again vs the already-"under-batched"
8-GPU setup (train_batch_size=1 per GPU, grad_accum unchanged), so the gen/critic
balance the v3 lr fix targets will differ -- this is a smoke/feasibility test at
smaller scale, not an apples-to-apples reproduction of the 8-GPU dynamics.
"""
from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v3_4gpu",
                               checkpointing_steps=100,
                               sample_every=50)
