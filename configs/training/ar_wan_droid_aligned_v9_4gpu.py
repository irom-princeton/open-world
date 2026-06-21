"""v9 (4-GPU): combine the working levers.

gen lr 2.5e-7 + critic_steps 10 + max_grad_norm 1.0 + real_guidance 3.0 -> 2.0.
Slow generator, strong+tracking critic, tight clip, and lower real-CFG so the
teacher score injects less global motion gain (a suspected drift amplifier).
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v9_4gpu",
                               learning_rate=2.5e-7,
                               critic_steps_per_gen_step=10,
                               max_grad_norm=1.0,
                               real_guidance_scale=2.0,
                               checkpointing_steps=100,
                               sample_every=50)
