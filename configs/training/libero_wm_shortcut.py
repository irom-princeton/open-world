"""LIBERO world-model training config -- shortcut (few-step) student.

Distills the flow-matching teacher trained via ``libero_wm.py`` into a
shortcut model that can sample in 1/2/4/8 steps. Initialize from the
teacher checkpoint via ``ckpt_path``.

Differences vs. ``libero_wm.py``:
* ``flow_map_type="shortcut"`` selects the shortcut training branch in
  ``flow_map_ctrl_world.py``.
* ``distance_conditioning=True`` so the UNet receives ``dt_base``.
* Lower ``learning_rate`` and ``max_train_steps`` -- distillation needs
  far fewer updates than teacher pretraining.
"""

from openworld.training.world_model.config import LiberoWMArgs


def get_args() -> LiberoWMArgs:
    args = LiberoWMArgs(
        # ----- Paths -----
        svd_model_path="external/stable-video-diffusion-img2vid",
        clip_model_path="external/clip-vit-base-patch32",
        # IMPORTANT: point this at the trained flow-matching teacher.
        ckpt_path="checkpoints/wm_libero/libero_flow_matching_v0/checkpoint-200000.pt",

        # ----- Dataset (same as teacher) -----
        dataset_root_path="data/wm_training/libero_processed",
        dataset_meta_info_path="data/wm_training/libero_processed",
        dataset_names="libero_spatial+libero_object+libero_goal+libero_10",
        dataset_cfgs="libero_spatial+libero_object+libero_goal+libero_10",
        prob=(0.25, 0.25, 0.25, 0.25),

        # ----- Compute -----
        train_batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        num_workers=4,

        # ----- Schedule (shorter than teacher) -----
        learning_rate=5e-6,
        max_train_steps=100_000,
        checkpointing_steps=10_000,
        validation_steps=2_500,
        max_grad_norm=1.0,

        # ----- Architecture (must match teacher) -----
        num_cams=2,
        num_frames=5,
        num_history=6,
        action_dim=7,
        down_sample=4,

        # ----- Shortcut training -----
        flow_map_type="shortcut",
        distance_conditioning=True,

        # ----- Few-step eval during training -----
        # The shortcut model is meant to run in very few steps; render
        # samples at 1/2/4/8 to track distillation progress.
        test_num_inference_steps=(1, 2, 4, 8),

        tag="libero_shortcut_v0",
    )
    return args
