"""v8 (4-GPU): gen lr 2.5e-7 (very conservative generator).

Escalate the primary lever: quarter the v3 gen lr (1e-6 -> 2.5e-7). Slow enough
that the critic should comfortably track even at the default 5 critic steps.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v8_4gpu",
                               learning_rate=2.5e-7,
                               checkpointing_steps=100,
                               sample_every=50)
