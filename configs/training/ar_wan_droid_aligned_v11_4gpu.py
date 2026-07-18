"""4-GPU DMD distillation, v11 -- the mechanism FIX experiment (only surviving non-default arm).

Tests whether capping the high-sigma DMD range (``dmd_max_step_ratio`` 0.98 -> 0.7)
prevents the latent-std runaway that every gen-lr arm only delayed. Pinned to v4's
gen lr 5e-7 plus the pre-v10 critic/clip/CFG values, so it is a clean A/B against
the old v4 arm and is unaffected by the v10-based default.
"""
from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned import get_args as _aligned


def get_args():
    return dataclasses.replace(_aligned(), tag="ar_wan_dmd_aligned_v11_4gpu",
                               learning_rate=5e-7,
                               dmd_max_step_ratio=0.7,
                               critic_steps_per_gen_step=5,
                               max_grad_norm=10.0,
                               real_guidance_scale=3.0,
                               checkpointing_steps=100,
                               sample_every=50)
