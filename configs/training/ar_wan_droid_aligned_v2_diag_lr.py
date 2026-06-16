"""Generator-LR dose-response variant of the 80-step DMD diagnostic.

Reads generator ``learning_rate`` from env ``DMD_GEN_LR`` (default 2e-6) so one
config drives the A/B. Tests whether the action-coupled static-camera-motion
erosion is a "generator outpaces the critic" instability (we're 8-GPU
under-batched vs the reference 64-GPU recipe, so the inherited LRs run hot). All
else equal to the baseline diag (CFG 3.0, 80 steps, log every step).
"""
from __future__ import annotations

import dataclasses
import os

from configs.training.ar_wan_droid_aligned_v2_diag import get_args as _diag


def get_args():
    lr = float(os.environ.get("DMD_GEN_LR", "2e-6"))
    tag = "ar_wan_dmd_aligned_v2_diag_lr" + f"{lr:.0e}"          # e.g. lr1e-06
    return dataclasses.replace(_diag(), learning_rate=lr, tag=tag)
