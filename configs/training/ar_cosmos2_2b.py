"""AR self-forcing world model on Cosmos-Predict2-2B (the OmniDreams substrate).

    accelerate launch -m openworld.autoregressive.train_self_forcing \
        --config configs/training/ar_cosmos2_2b.py

Note: Cosmos cached rollout (RoPE offset) is not yet wired — training uses the
exact full-clip block-causal forward, but for *cached* autoregressive rollout
prefer the Wan backbone. See docs/AUTOREGRESSIVE.md.
"""

from __future__ import annotations

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        backbone="cosmos_predict2_2b",
        tag="ar_cosmos2_selfforcing",
        num_cams=3,
        multiview_layout="height_stack",
        width=320,
        height=320,
        frames_per_block=2,
        num_history_blocks=3,
        rollout_blocks=12,
        denoising_step_list=(1000, 750, 500, 250),
        warp_denoising_step=True,
        critic_steps_per_gen_step=5,
        real_guidance_scale=3.5,
        student_init_ckpt=None,
        teacher_ckpt=None,
        learning_rate=6e-6,
        critic_learning_rate=6e-6,
        mixed_precision="bf16",
        train_batch_size=1,
        max_train_steps=200_000,
    )
