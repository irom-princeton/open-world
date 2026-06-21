"""v6 (4-GPU): gen lr 5e-7 + critic_lr 4e-7 -> 8e-7 + critic_steps 10.

If 10 critic steps (v5) still cannot keep up, let the critic also move faster
per step (2x critic lr). Strengthens critic tracking on both axes (count + step).
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v6_4gpu",
                               learning_rate=5e-7,
                               critic_learning_rate=8e-7,
                               critic_steps_per_gen_step=10,
                               checkpointing_steps=100,
                               sample_every=50)
