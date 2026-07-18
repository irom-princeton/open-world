"""AR self-forcing on Wan2.1-1.3B, DROID ctrl-world data (2 cams, height-stacked).

2-view is the default (1 per-clip randomly sampled side view + wrist); it trains on
the SAME 3-view preprocessed latents (ARLatentDataset subsets at load time), so no
re-preprocessing is needed. For the full 3-view layout use ar_wan_droid_3view.py, or
set env NUM_CAMS=3 for a one-off (e.g. eval/replay of a 3-view checkpoint).

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

import os
import torch

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        backbone="wan_1_3b",
        backbone_ckpt=os.environ.get("BACKBONE_CKPT") or "external/Wan2.1-T2V-1.3B-Diffusers",
        tag="ar_wan_droid",
        # data
        data_format="droid_ctrl_world",
        data_root=os.environ.get("DATA_ROOT") or "/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world",
        latent_root=os.environ.get("LATENT_ROOT") or "data/droid_ar_latents",
        # rollout geometry (DROID: 192x320). 2-view default (see module docstring);
        # NUM_CAMS env or ar_wan_droid_3view.py for the full 3-view layout.
        num_cams=2,
        multiview_layout="height_stack",
        # per-frame cross-attn (latent frame f attends only to action token f) --
        # the strongest action->frame binding; the DROID models are trained/inferred
        # with it (the inference configs in configs/inference/ inherit this).
        action_cond_mode="cross_attn_aligned",
        height=192,
        width=320,
        frames_per_block=2,
        num_history_blocks=2,
        rollout_blocks=8,
        # Stage L0: few-step DMD distillation. omni-dreams self-forcing recipe:
        # 4-step schedule, gen lr 2e-6 / critic lr 4e-7, betas (0, 0.999), wd 1e-2,
        # grad-clip 10, real-CFG 3.0, ~10k steps. Initializes from the two
        # mid-training stages -- run those first (ar_wan_{studentinit,teacher}_droid.py).
        denoising_step_list=(1000, 750, 500, 250),
        warp_denoising_step=True,
        critic_steps_per_gen_step=5,
        real_guidance_scale=3.0,
        student_init_ckpt=os.environ.get("STUDENT_INIT_CKPT") or "checkpoints/ar_wm/ar_wan_studentinit/checkpoint-40000.pt",
        teacher_ckpt=os.environ.get("TEACHER_CKPT") or "checkpoints/ar_wm/ar_wan_teacher/checkpoint-40000.pt",
        learning_rate=2e-6,
        critic_learning_rate=4e-7,
        adam_betas=(0.0, 0.999),
        weight_decay=1e-2,
        max_grad_norm=10.0,
        # fp32 master weights + bf16 autocast compute (see docs/AUTOREGRESSIVE.md "Dtype").
        dtype=torch.float32,
        mixed_precision="bf16",
        train_batch_size=1,
        max_train_steps=10_000,
        vae_dir=os.environ.get("VAE_DIR") or "external/Wan2.1-T2V-1.3B-Diffusers",
    )
