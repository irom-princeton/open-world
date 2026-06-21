"""4-GPU DMD distillation -- the adopted default configuration.

v10 was the chosen point of the (now-removed) divergence sweep: its hyperparameters
were promoted into the default ``ar_wan_droid_aligned`` config, so this 4-GPU launch
config only adds the run tag and the denser preview cadence on top of that default.
"""
from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned import get_args as _aligned


def get_args():
    return dataclasses.replace(_aligned(), tag="ar_wan_dmd_aligned_v10_4gpu",
                               sample_every=50)
