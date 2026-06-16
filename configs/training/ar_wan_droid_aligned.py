"""Stage L0 self-forcing / DMD distillation, action_cond_mode="cross_attn_aligned".

Aligned counterpart of ar_wan_droid.py. The few-step generator is initialized
from the aligned student-init stage (ar_wan_studentinit_aligned) and the
real-score teacher + fake-score critic from the aligned bidirectional teacher
stage (ar_wan_teacher_aligned). All three must share ``action_cond_mode`` so the
generator's action conditioning feeds the teacher/critic correctly (Fix 2).

``teacher_ckpt`` defaults to the *latest* checkpoint-*.pt in
``checkpoints/ar_wm/ar_wan_teacher_aligned/`` -- the highest training step,
preferring a permanent checkpoint over a same-step rolling one -- resolved at
config-load time so a re-submit always picks up the freshest teacher. Override
with env ``DMD_TEACHER_CKPT`` (the launch script pins + logs the resolved path).

    torchrun --nproc_per_node=8 -m openworld.autoregressive.train_self_forcing \
        --config configs/training/ar_wan_droid_aligned.py
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

from configs.training.ar_wan_droid import get_args as _base

_TEACHER_DIR = "checkpoints/ar_wm/ar_wan_teacher_aligned"


def _latest_teacher_ckpt() -> str:
    """Latest teacher checkpoint: env override, else highest-step file in the
    aligned teacher dir (permanent preferred over a same-step ``-rolling``)."""
    override = os.environ.get("DMD_TEACHER_CKPT")
    if override:
        return override
    cands = list(Path(_TEACHER_DIR).glob("checkpoint-*.pt"))
    if not cands:
        raise FileNotFoundError(
            f"No checkpoint-*.pt found in {_TEACHER_DIR}; the aligned teacher "
            "stage must produce a checkpoint before L0 distillation can run."
        )

    def _key(p: Path) -> tuple[int, int]:
        m = re.search(r"checkpoint-(\d+)", p.name)
        step = int(m.group(1)) if m else -1
        is_permanent = "rolling" not in p.name
        return (step, 1 if is_permanent else 0)

    return str(max(cands, key=_key))


def get_args():
    return dataclasses.replace(
        _base(),
        action_cond_mode="cross_attn_aligned",
        tag="ar_wan_dmd_aligned",
        student_init_ckpt="checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-40000.pt",
        teacher_ckpt=_latest_teacher_ckpt(),
        # DMD distillation is slow per step (~23s) and converges in a few hundred
        # steps (cf. OmniDreams L0 @ 500, Self-Forcing @ 600), unlike the 150k-step
        # mid-training runs. Checkpoint much more densely than the base defaults
        # (rolling 500 / permanent 4000): keep a permanent every 200 steps and an
        # intermediate (rolling) every 100.
        checkpointing_steps=100,
        permanent_checkpoint_steps=200,
    )
