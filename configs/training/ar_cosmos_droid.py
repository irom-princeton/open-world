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
        # few-step student + distillation (OmniDreams Cosmos2 defaults)
        denoising_step_list=(1000, 750, 500, 250),
        warp_denoising_step=True,
        critic_steps_per_gen_step=5,
        real_guidance_scale=3.5,
        student_init_ckpt=None,
        teacher_ckpt=None,
        learning_rate=6e-6,
        critic_learning_rate=6e-6,
        # fp32 master weights + bf16 autocast compute (see docs/AUTOREGRESSIVE.md "Dtype").
        dtype=torch.float32,
        mixed_precision="bf16",
        train_batch_size=1,
        max_train_steps=200_000,
    )
