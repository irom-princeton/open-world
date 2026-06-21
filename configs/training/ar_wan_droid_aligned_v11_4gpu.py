"""v11 (4-GPU): the FIRST non-LR fix attempt — cap the DMD high-sigma range.

The v3-v10 sweep proved gen lr only DELAYS the collapse (onset ~150/330/660 at
1e-6/5e-7/2.5e-7, clean ~2x per halving) — never prevents it. The plateau-DMD-
gradient probe (scripts/plateau_dmd_gradient_probe.py) showed why: a net-inflating
teacher-vs-critic score mismatch that GROWS with sigma (grad std 0.02 at sigma 0.1
-> 0.29 at 0.98) ratchets the generated-latent std from ~0.85 to a stable ~1.5
fixed point. The high-sigma DMD samples dominate that inflating pressure.

v11 attacks the mechanism (not the LR): lower ``dmd_max_step_ratio`` 0.98 -> 0.7
so the DMD distillation loss stops sampling the highest-noise regime where the
two scores diverge most. Gen lr is held at v4's 5e-7 (onset ~330) for a clean A/B
against v4: if v11 sails past ~330-500 healthy, capping high sigma is the fix.
checkpoint/sample cadence matches the sweep (100/50) for early visibility.
"""
from __future__ import annotations
import dataclasses
from configs.training.ar_wan_droid_aligned_v3 import get_args as _v3


def get_args():
    return dataclasses.replace(_v3(), tag="ar_wan_dmd_aligned_v11_4gpu",
                               learning_rate=5e-7,
                               dmd_max_step_ratio=0.7,
                               # Pin the critic/clip/CFG levers to the pre-v10-default
                               # values v11 was designed with, so the new v10-based
                               # default in ar_wan_droid_aligned.py does NOT alter this
                               # active A/B (v11 = v4's 5e-7 + dmd_hi 0.7, clean vs v4).
                               critic_steps_per_gen_step=5,
                               max_grad_norm=10.0,
                               real_guidance_scale=3.0,
                               checkpointing_steps=100,
                               sample_every=50)
