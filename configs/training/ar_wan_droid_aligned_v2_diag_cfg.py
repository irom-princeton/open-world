"""CFG dose-response variant of the 80-step DMD diagnostic.

Reads ``real_guidance_scale`` from env ``DMD_RGS`` (default 3.0) so a single
config drives several A/B runs, each writing to its own tag/output dir. Used to
test whether the action-coupled static-camera-motion degradation is driven by the
real-score CFG over-extrapolating the action-conditional direction.
"""
from __future__ import annotations

import dataclasses
import os

from configs.training.ar_wan_droid_aligned_v2_diag import get_args as _diag


def get_args():
    rgs = float(os.environ.get("DMD_RGS", "3.0"))
    tag = "ar_wan_dmd_aligned_v2_diag_cfg" + f"{rgs:g}".replace(".", "p")  # e.g. cfg1p5
    return dataclasses.replace(_diag(), real_guidance_scale=rgs, tag=tag)
