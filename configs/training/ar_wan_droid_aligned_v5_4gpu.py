"""v5 (4-GPU): gen lr 5e-7 + critic_steps_per_gen_step 5 -> 10.

Beyond slowing the generator (v4), give the critic twice as many updates per
generator step so the fake-score critic tracks the moving generator
distribution -- directly attacks the "generator outpaces critic" failure mode.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v5_4gpu",
                               learning_rate=5e-7,
                               critic_steps_per_gen_step=10,
                               checkpointing_steps=100,
                               sample_every=50)
