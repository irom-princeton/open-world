"""v10 (4-GPU): most conservative -- last attempt in the sweep.

gen lr 2.5e-7 + critic_steps 15 + max_grad_norm 0.5 + real_guidance 1.5.
Every divergence lever pushed to its conservative extreme simultaneously. If
this still collapses, the issue is not LR/critic-balance/clip/CFG and needs a
design change (inits, scoring, schedule), not another HP point.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v10_4gpu",
                               learning_rate=2.5e-7,
                               critic_steps_per_gen_step=15,
                               max_grad_norm=0.5,
                               real_guidance_scale=1.5,
                               checkpointing_steps=100,
                               sample_every=50)
