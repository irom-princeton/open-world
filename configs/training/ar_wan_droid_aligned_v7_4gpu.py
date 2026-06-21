"""v7 (4-GPU): gen lr 5e-7 + max_grad_norm 10 -> 1.0.

Different lever: cap the per-step generator update magnitude hard. A loose
clip (10) lets an occasional large DMD-gradient step kick the generator
off-manifold; a tight clip (1.0) bounds how far any single step can drift.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v7_4gpu",
                               learning_rate=5e-7,
                               max_grad_norm=1.0,
                               checkpointing_steps=100,
                               sample_every=50)
