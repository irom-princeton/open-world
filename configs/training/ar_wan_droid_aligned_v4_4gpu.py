"""v4 (4-GPU): gen lr halved again, 1e-6 -> 5e-7.

If v3 (gen lr 1e-6) still black-collapses in the first ~200 steps, the generator
is still outpacing the critic in this under-batched 4-GPU setup. v4 continues the
v3 lever (slow the generator) by halving gen lr once more. critic_lr stays 4e-7.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v4_4gpu",
                               learning_rate=5e-7,
                               checkpointing_steps=100,
                               sample_every=50)
