"""LIBERO world-model training config.

This is the LIBERO analog of ``Fast-Control-World/config_flow_map.py``.
Defaults mirror the DROID values from that file unless noted in
``docs/LIBERO.md``.

Override fields by writing a new file in ``configs/training/`` that
subclasses :class:`LiberoWMArgs` (or just instantiates it with overrides) and
points the training entrypoint at it via ``--config configs/training/foo.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


@dataclass
class LiberoWMArgs:
    # ---------------- training infra ----------------
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    train_batch_size: int = 1
    shuffle: bool = True
    num_train_epochs: int = 100
    max_train_steps: int = 500_000
    checkpointing_steps: int = 20_000
    validation_steps: int = 2_500
    max_grad_norm: float = 1.0
    video_num: int = 10
    debug: bool = False

    # ---------------- model paths -------------------
    svd_model_path: str = "external/stable-video-diffusion-img2vid"
    clip_model_path: str = "external/clip-vit-base-patch32"
    ckpt_path: str | None = None  # initial WM weights, e.g. checkpoint-10000.pt

    # ---------------- dataset -----------------------
    dataset_root_path: str = "data/libero_processed"
    # Mix all 5 suites by default (equal probability). Override to specialize.
    dataset_names: str = "libero_spatial+libero_object+libero_goal+libero_10+libero_90"
    dataset_meta_info_path: str = "dataset_meta_info"
    # By default the per-suite normalization stat lives at
    # ``dataset_meta_info/<suite>/stat.json``. If a per-suite file is
    # missing, the loader falls back to ``dataset_meta_info/libero/stat.json``.
    dataset_cfgs: str = "libero_spatial+libero_object+libero_goal+libero_10+libero_90"
    prob: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    annotation_name: str = "annotation"
    num_workers: int = 4
    # 20 Hz LIBERO -> 5 Hz WM rate: down_sample=4 is the natural analog of
    # DROID's 15Hz/3 = 5Hz.
    down_sample: int = 4
    skip_step: int = 1

    # ---------------- logging -----------------------
    tag: str = "libero_flow_matching"
    output_dir: str = field(init=False)
    wandb_project_name: str = "libero_world_model"
    wandb_run_name: str = field(init=False)

    # ---------------- model arch --------------------
    motion_bucket_id: int = 127
    fps: int = 7
    guidance_scale: float = 2.0
    num_inference_steps: int = 50
    test_num_inference_steps: tuple[int, ...] = ()
    decode_chunk_size: int = 7
    width: int = 320
    height: int = 320

    num_frames: int = 5  # future frames per WM rollout
    num_history: int = 6
    action_dim: int = 7  # 6 EEF (xyz + axis-angle) + 1 gripper
    num_cams: int = 2  # agentview + wrist

    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    dtype: torch.dtype = torch.bfloat16

    # ---------------- flow / shortcut ---------------
    SIGMA_MIN: float = 0.02
    SIGMA_MAX: float = 700.0
    flow_map_type: str = "flow_matching"  # one of {flow_matching, shortcut, flow_map}
    distance_conditioning: bool = False
    use_train_set_for_val: bool = False
    use_weights: bool = False

    flow_map_loss_type: str = "lsd"
    psd_sample_mode: str = "uniform"
    bias_prob: float = -1
    one_step_prob: float = 0.0
    one_step_sample: bool = False

    # ---------------- shortcut ----------------------
    bootstrap_bs: int = 1
    DENOISE_TIMESTEPS: int = 128
    single_bs_mode: bool = False

    def __post_init__(self) -> None:
        self.output_dir = f"checkpoints/wm_libero/{self.tag}"
        self.wandb_run_name = self.tag
        # Per-camera latent shape after SVD VAE 8x downsample.
        self.latent_h_per_cam = self.height // 8
        self.latent_h_total = self.latent_h_per_cam * self.num_cams
        self.latent_w = self.width // 8
