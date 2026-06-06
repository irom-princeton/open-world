"""AR self-forcing on Wan2.1-1.3B, DROID ctrl-world data (3 cams, height-stacked).

Pipeline:
    1. sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/preprocess_ar_latents.py \
           --format droid_ctrl_world --root /scratch/gpfs/AM43/yy4041/data/droid_ctrl_world \
           --out data/droid_ar_latents --splits train val --num-views 3
    2. sbatch bash_scripts/ar_gpu.slurm accelerate launch \
           -m openworld.autoregressive.train_self_forcing --config configs/training/ar_wan_droid.py

Geometry note: DROID episodes are ~109 RGB frames -> ~28 latent frames (Wan VAE
4x temporal). The clip is (num_history_blocks + rollout_blocks) * frames_per_block
= (2 + 8) * 2 = 20 latent frames (~77 RGB frames), which fits.
"""

from __future__ import annotations

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        backbone="wan_1_3b",
        backbone_ckpt="external/Wan2.1-T2V-1.3B-Diffusers",
        tag="ar_wan_droid",
        # data
        data_format="droid_ctrl_world",
        data_root="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world",
        latent_root="data/droid_ar_latents",
        # rollout geometry (DROID: 192x320, 3 cams)
        num_cams=3,
        multiview_layout="height_stack",
        height=192,
        width=320,
        frames_per_block=2,
        num_history_blocks=2,
        rollout_blocks=8,
        # few-step student + distillation
        denoising_step_list=(1000, 500),
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
