"""Stage L0 DMD distillation (aligned) -- v2 / clean restart.

Identical to ``ar_wan_droid_aligned.py`` except for a fresh ``tag`` so this run
writes to a NEW output dir (``checkpoints/ar_wm/ar_wan_dmd_aligned_v2``) and does
NOT auto-resume from the previous (diverged) ``ar_wan_dmd_aligned`` run. It still
loads the correct mid-trained inits (40k aligned student-init + latest aligned
teacher) -- only the broken DMD ``training_state.pt`` is left behind.

This run picks up the loss-scaling / scoring fixes that landed after v1:
  * whole-clip scoring (``dmd_score_whole_clip=True``, default) -- the
    bidirectional teacher/critic now see the full clip, not 2-frame fragments.
  * mean-reduced DMD/critic losses (matches omni-dreams / Self-Forcing). v1 used
    a sum-over-blocks scale that over-drove both optimizers by ~num_blocks (8x)
    at the configs' reference LRs and diverged the student to rainbow noise.
  * random-exit stays OFF (default) -- this run isolates the correctness fixes;
    the speed levers are opt-in once this is confirmed healthy.
"""

from __future__ import annotations

import dataclasses

from configs.training.ar_wan_droid_aligned import get_args as _aligned


def get_args():
    # dataclasses.replace re-runs __post_init__, so output_dir / wandb_run_name
    # are recomputed from the new tag.
    return dataclasses.replace(_aligned(), tag="ar_wan_dmd_aligned_v2")
