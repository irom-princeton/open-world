"""Config for the autoregressive / self-forcing world model.

Mirrors the conventions of ``openworld.training.world_model.config.LiberoWMArgs``
(plain dataclass, ``get_args()`` factory in ``configs/training/*.py``,
``--config`` points the trainer at one) but adds the autoregressive-specific
knobs: backbone choice, block-causal chunking, multi-view packing, and the
self-forcing / DMD distillation schedule.

The dataset layout (preprocessed LIBERO latents) is shared with the SVD trainer,
so the data-side fields match ``LiberoWMArgs`` field-for-field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Backbone presets. dim/layers/heads mirror the public Wan2.1-1.3B and
# Cosmos-Predict2-2B configs so we can build a random-init model for tests and
# `from_pretrained` the real weights for training. `cross_attn_dim` is the
# text/condition width the backbone's cross-attention expects (T5/UMT5 width);
# the action encoder projects 1024 -> cross_attn_dim.
# ---------------------------------------------------------------------------
BACKBONE_PRESETS: dict[str, dict] = {
    "wan_1_3b": {
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "in_channels": 16,
        "cross_attn_dim": 4096,   # UMT5-XXL
        "vae_spatial_factor": 8,
        "vae_temporal_factor": 4,
        "patch_temporal": 1,      # Wan patch_size = (1, 2, 2)
    },
    "cosmos_predict2_2b": {
        "hf_repo": "nvidia/Cosmos-Predict2-2B-Video2World",
        "in_channels": 16,
        "cross_attn_dim": 1024,   # Cosmos uses a 1024-d cross-attn projection
        "vae_spatial_factor": 8,
        "vae_temporal_factor": 4,
        "patch_temporal": 1,
    },
    # Wraps the existing SVD CrtlWorld UNet so the AR rollout/eval harness can
    # run against the legacy backbone for apples-to-apples comparison.
    "svd": {
        "hf_repo": "external/stable-video-diffusion-img2vid",
        "in_channels": 4,
        "cross_attn_dim": 1024,
        "vae_spatial_factor": 8,
        "vae_temporal_factor": 1,
        "patch_temporal": 1,
    },
    # Tiny random DiT for unit tests — no weights, CPU-friendly.
    "dummy": {
        "hf_repo": None,
        "in_channels": 16,
        "cross_attn_dim": 64,
        "vae_spatial_factor": 8,
        "vae_temporal_factor": 4,
        "patch_temporal": 1,
    },
}


@dataclass
class ARWMArgs:
    # ---------------- backbone -----------------------
    backbone: str = "wan_1_3b"          # key into BACKBONE_PRESETS
    backbone_ckpt: str | None = None    # local path / HF repo override; None -> preset hf_repo
    random_init_backbone: bool = False  # True -> build from config w/o downloading weights (tests/CI)

    # ---------------- multistage pipeline ------------
    # Mirrors omni-dreams' 3-stage recipe. The two mid-training stages are
    # independent (both start from the base backbone) and run as parallel jobs;
    # self-forcing then loads both. ``stage`` selects the trainer/entry behavior:
    #   "student_init" -- L2a: causal (block-causal) flow-matching mid-training
    #                     -> initializes the self-forcing generator.
    #   "teacher"      -- L1b: bidirectional flow-matching mid-training
    #                     -> initializes the self-forcing real-score teacher + critic.
    #   "self_forcing" -- L0: few-step DMD distillation (loads the two above).
    stage: str = "self_forcing"
    # Mid-training (L2a/L1b) AdamW knobs -- omni-dreams causal/teacher configs.
    midtrain_lr: float = 3e-5
    midtrain_weight_decay: float = 1e-3
    midtrain_grad_clip: float = 0.1

    # ---------------- training infra -----------------
    learning_rate: float = 6e-6         # generator (student) lr
    critic_learning_rate: float = 6e-6  # DMD fake-score / critic lr
    # AdamW betas + weight decay for the self-forcing (L0) optimizers. omni-dreams
    # uses betas=(0.0, 0.999) (no first-moment momentum) and wd 1e-2 for both the
    # generator and critic; max_grad_norm=10 there too.
    adam_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    # Shard the generator/critic/teacher + optimizer states across GPUs with FSDP2
    # (torch ``fully_shard``) when launched on >1 process. Required to fit fp32
    # master weights for the 3-model self-forcing stack -- plain DDP replicates the
    # full models per GPU and OOMs. No effect on a single process. Compute precision
    # is still driven by ``mixed_precision`` (bf16 autocast over fp32 master).
    use_fsdp: bool = True
    train_batch_size: int = 1
    shuffle: bool = True
    max_train_steps: int = 200_000
    # Step-based checkpoint retention. A *rolling* checkpoint is rewritten every
    # ``checkpointing_steps`` (~1h of wall-clock) and the previous rolling file is
    # deleted once the new one lands -- a crash-safety net that never grows. A
    # *permanent* checkpoint is kept every ``permanent_checkpoint_steps`` (~8h) and
    # never deleted. Between two permanents only the single latest rolling file
    # occupies disk.
    #
    # The step counts assume ~500 steps/h, an *estimate* for Wan-1.3B self-forcing
    # (6 rollouts/step x 8 blocks x 2 denoise steps) on an H200; data-parallel DDP
    # keeps per-step wall time ~constant regardless of GPU count, so this holds for
    # multi-GPU runs too. The trainer logs measured "steps/h" after warmup -- snap
    # these to your real rate once you see it (rolling ~= steps/h, permanent = 8x).
    checkpointing_steps: int = 500
    permanent_checkpoint_steps: int = 4_000
    # Crash-safe resume. Alongside the (generator-only) inference checkpoints above,
    # write ONE full training-state bundle (generator + online critic + both AdamW
    # optimizer states + step) every ``checkpointing_steps``, overwritten in place
    # so only a single ~30 GB file ever exists. On startup the trainer auto-resumes
    # from ``<output_dir>/training_state.pt`` if present (set ``resume_from`` to load
    # a specific path instead), continuing *exactly* from the cut point. Disable to
    # save disk if you don't need mid-run resume.
    save_resume_state: bool = True
    resume_from: str | None = None
    validation_steps: int = 1_000
    max_grad_norm: float = 1.0
    video_num: int = 4
    num_workers: int = 4
    seed: int = 0

    # ---------------- qualitative sample previews -------------------
    # At every checkpoint save we run open-loop AR replay on a few random ``val``
    # episodes, VAE-decode GT|PRED side-by-side, and log them to wandb as videos.
    # The rollout reuses the same KV-cached path as training, so it is FSDP-safe
    # (the forward all-gathers fire on every rank); episode selection is seeded by
    # step so all ranks pick the same episodes and only rank 0 decodes/logs.
    log_samples: bool = True
    num_sample_videos: int = 2        # random val episodes previewed per checkpoint
    sample_history_blocks: int = 1    # ground-truth blocks used to prime ("first frame")
    sample_max_blocks: int = 8        # cap generated blocks for a fast preview (0 = to ep end)
    sample_video_fps: int = 8
    # Wan VAE used to decode 16-ch latents -> RGB (Cosmos latents use it too).
    vae_dir: str = "external/Wan2.1-T2V-1.3B-Diffusers"

    # ---------------- data (precomputed latents) ----------------
    # The framework ingests any raw format via a format adapter
    # (openworld.autoregressive.data.formats) at preprocess time, then trains on
    # a standard precomputed latent layout (scripts/preprocess_ar_latents.py).
    data_format: str = "droid_ctrl_world"    # format adapter for the raw dataset
    data_root: str | None = None             # raw dataset root (preprocess input)
    latent_root: str = "data/droid_ar_latents"  # precomputed latent layout (training input)

    # ---------------- logging ------------------------
    tag: str = "ar_wan_selfforcing"
    output_dir: str = field(init=False)
    wandb_project_name: str = "ar_world_model"
    wandb_run_name: str = field(init=False)

    # ---------------- rollout geometry ---------------
    # Latent video shape per view is (C, H/8, W/8). Cameras can be packed two
    # ways (see conditioning/multiview.py):
    #   "height_stack"  -> stack views along H (the existing CrtlWorld layout)
    #   "sequence_pack" -> concat views along the token sequence w/ view-id embeds
    #                      (OmniDreams layout; full cross-view attn, causal in time)
    # "height_stack" matches the existing LiberoLatentDataset (cameras already
    # stacked along H in the preprocessed latents) and needs no backbone change.
    # "sequence_pack" is the OmniDreams layout (per-view latents + view-id embed)
    # and requires re-preprocessing the dataset to emit per-view latents.
    multiview_layout: str = "height_stack"
    num_cams: int = 3                 # robot views to predict jointly (3 or 4)
    width: int = 320
    height: int = 320

    # latent-frame chunking for block-causal attention.
    frames_per_block: int = 2         # latent frames generated per AR step (OmniDreams chunk2)
    num_history_blocks: int = 3       # clean context blocks kept warm in the KV-cache at train time
    rollout_blocks: int = 12          # blocks generated per self-forcing rollout (train)
    max_kv_blocks: int | None = None  # None -> unbounded; else sliding-window local attention

    action_dim: int = 7              # 6 EEF (xyz + axis-angle) + 1 gripper
    text_cond: bool = True
    frame_level_cond: bool = True

    # ---------------- self-forcing / DMD -------------
    # Few-step student denoising schedule (flow-matching timesteps in [0,1000]).
    # Matches OmniDreams' `denoising_step_list = [1000, 750, 500, 250]`; the
    # shipped fast runner uses 2 steps -> set to e.g. (1000, 500).
    denoising_step_list: tuple[int, ...] = (1000, 750, 500, 250)
    warp_denoising_step: bool = True  # warp the integer steps onto the scheduler's sigmas
    num_train_timestep: int = 1000
    dmd_min_step_ratio: float = 0.02  # DMD samples the score-distillation noise level in
    dmd_max_step_ratio: float = 0.98  # [min, max] * num_train_timestep
    critic_steps_per_gen_step: int = 5  # critic (fake score) updates per generator update
    real_guidance_scale: float = 3.5   # CFG scale used inside the real (teacher) score
    fake_guidance_scale: float = 1.0
    teacher_ckpt: str | None = None    # bidirectional teacher (real score); None -> backbone_ckpt
    # Optional: warm-start the causal student from a bidirectional checkpoint
    # finetuned on robot data (the "student-init" / E1 stage).
    student_init_ckpt: str | None = None

    # ``dtype`` is the *parameter/optimizer* dtype. For a real run keep it
    # float32 (stable AdamW master weights) and let ``mixed_precision`` drive the
    # bf16 autocast compute; for a quick smoke bf16 everywhere is fine.
    dtype: torch.dtype = torch.bfloat16

    @property
    def stage_is_causal(self) -> bool:
        """Mid-training attention pattern: student-init is block-causal, teacher
        is bidirectional. (Self-forcing handles its own per-model masking.)"""
        return self.stage != "teacher"

    @property
    def autocast_dtype(self) -> "torch.dtype | None":
        """Compute dtype for the transformer forward, derived from
        ``mixed_precision``. ``None`` when params already carry the compute dtype
        (so we don't autocast bf16 weights to bf16, or fp32 to fp32)."""
        want = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.mixed_precision)
        # only autocast when it would actually change precision vs the param dtype
        return want if want is not None and want != self.dtype else None

    def __post_init__(self) -> None:
        self.output_dir = f"checkpoints/ar_wm/{self.tag}"
        self.wandb_run_name = self.tag
        if self.backbone not in BACKBONE_PRESETS:
            raise ValueError(
                f"Unknown backbone {self.backbone!r}; choose from {list(BACKBONE_PRESETS)}"
            )
        preset = BACKBONE_PRESETS[self.backbone]
        self.preset = preset
        # Per-camera latent shape after the backbone VAE spatial downsample.
        f = preset["vae_spatial_factor"]
        self.latent_h_per_cam = self.height // f
        self.latent_w = self.width // f
        # height_stack glues cameras along H into one tall latent image.
        self.latent_h_total = (
            self.latent_h_per_cam * self.num_cams
            if self.multiview_layout == "height_stack"
            else self.latent_h_per_cam
        )
        self.in_channels = preset["in_channels"]
        self.cross_attn_dim = preset["cross_attn_dim"]

    @property
    def resolved_backbone_ckpt(self) -> str | None:
        return self.backbone_ckpt or self.preset["hf_repo"]
