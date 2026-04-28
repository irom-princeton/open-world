"""LIBERO world-model training config (Python).

Mirrors the Fast-Control-World pattern: a Python file that defines
``get_args() -> LiberoWMArgs``. The training entry point loads this via
``--config configs/training/libero_wm.py``.
"""

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    args = LiberoWMArgs(
        # ----- Paths (set these to your installation) -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        ckpt_path=None,  # set to e.g. checkpoints/wm/checkpoint-120000.pt to warm-start

        # ----- Dataset -----
        dataset_root_path="data/libero_processed",
        dataset_meta_info_path="dataset_meta_info",
        dataset_names="libero_spatial+libero_object+libero_goal+libero_10",
        dataset_cfgs="libero_spatial+libero_object+libero_goal+libero_10",
        prob=(0.25, 0.25, 0.25, 0.25),

        # ----- Compute -----
        train_batch_size=1,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule -----
        learning_rate=1e-5,
        max_train_steps=500_000,
        checkpointing_steps=20_000,
        validation_steps=2_500,
        max_grad_norm=1.0,

        # ----- Architecture (LIBERO-specific) -----
        num_cams=2,            # agentview + wrist
        num_frames=5,
        num_history=6,
        action_dim=7,          # 6 EEF + 1 gripper
        down_sample=4,         # 20 Hz -> 5 Hz

        # ----- Loss / sampling defaults -----
        flow_map_type="flow_matching",
        distance_conditioning=False,

        tag="libero_flow_matching_v0",
    )
    return args
