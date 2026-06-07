"""AR self-forcing on Cosmos-Predict2-2B, DROID ctrl-world data (3 cams, height-stacked).

Apples-to-apples sanity check against ``ar_wan_droid.py``: identical data, geometry,
and latents — only the backbone changes (Wan-1.3B -> Cosmos-Predict2-2B, the
OmniDreams substrate). Cosmos-Predict2 pairs with the **same Wan VAE**
(``AutoencoderKLWan``), so the latents preprocessed for the Wan run are reused
as-is — no re-encode.

Pipeline:
    1. (once) download the Cosmos transformer into external/ on the login node:
           bash bash_scripts/download_weights.sh nvidia/Cosmos-Predict2-2B-Video2World
    2. reuse the Wan-VAE latents from the ar_wan_droid run (data/droid_ar_latents);
       if not present yet:
           sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/preprocess_ar_latents.py \
               --format droid_ctrl_world --root /scratch/gpfs/AM43/yy4041/data/droid_ctrl_world \
               --out data/droid_ar_latents --splits train val --num-views 3
    3. sbatch bash_scripts/ar_gpu.slurm accelerate launch \
           -m openworld.autoregressive.train_self_forcing --config configs/training/ar_cosmos_droid.py

Recipe matches the OmniDreams Cosmos2 self-forcing defaults
(denoising_step_list=[1000,750,500,250]); real_guidance_scale is 3.0 in OmniDreams
(``self_forcing_dmd.py``) vs 3.5 here — kept at 3.5 for parity with ar_wan_droid.
"""

from __future__ import annotations

import torch

from openworld.autoregressive.config import ARWMArgs


def get_args() -> ARWMArgs:
    return ARWMArgs(
        backbone="cosmos_predict2_2b",
        backbone_ckpt="external/Cosmos-Predict2-2B-Video2World",
        tag="ar_cosmos_droid",
        # data (identical to ar_wan_droid — same Wan-VAE latents)
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
        # Stage L0: few-step DMD distillation (omni-dreams Cosmos2 self-forcing
        # recipe): 4-step schedule, gen lr 2e-6 / critic lr 4e-7, betas (0, 0.999),
        # wd 1e-2, grad-clip 10, real-CFG 3.0, ~10k steps. Initializes from the two
        # mid-training stages -- run those first (ar_cosmos_{studentinit,teacher}_droid.py).
        denoising_step_list=(1000, 750, 500, 250),
        warp_denoising_step=True,
        critic_steps_per_gen_step=5,
        real_guidance_scale=3.0,
        student_init_ckpt="checkpoints/ar_wm/ar_cosmos_studentinit/checkpoint-40000.pt",
        teacher_ckpt="checkpoints/ar_wm/ar_cosmos_teacher/checkpoint-40000.pt",
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
    )
