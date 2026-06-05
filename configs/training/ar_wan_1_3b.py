"""AR self-forcing world model on Wan2.1-1.3B (recommended substrate).

    accelerate launch -m openworld.autoregressive.train_self_forcing \
        --config configs/training/ar_wan_1_3b.py

Set ``teacher_ckpt`` / ``student_init_ckpt`` to your robot-finetuned bidirectional
Wan checkpoint (the E1/E2 stage) before launching a real distillation run.
"""

from __future__ import annotations

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        backbone="wan_1_3b",
        tag="ar_wan_selfforcing",
        # rollout geometry
        num_cams=3,
        multiview_layout="height_stack",   # dataset-compatible; cameras stacked along H
        width=320,
        height=320,
        frames_per_block=2,                # OmniDreams chunk2
        num_history_blocks=3,
        rollout_blocks=12,
        max_kv_blocks=None,                # None -> full memory; set e.g. 16 for local attn
        # few-step student (2 steps for the fast runner; 4 for higher quality)
        denoising_step_list=(1000, 500),
        warp_denoising_step=True,
        critic_steps_per_gen_step=5,
        real_guidance_scale=3.5,
        # init from robot-finetuned bidirectional Wan (fill these in):
        student_init_ckpt=None,
        teacher_ckpt=None,
        # infra
        learning_rate=6e-6,
        critic_learning_rate=6e-6,
        mixed_precision="bf16",
        train_batch_size=1,
        max_train_steps=200_000,
    )
