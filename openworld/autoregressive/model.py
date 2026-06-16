"""ARWorldModel: backbone + action conditioning + multi-view, acting as the
self-forcing *generator*.

It delegates the two backbone forward modes (so the trainer can treat it as a
:class:`DiTBackbone` whose ``parameters()`` also include the trainable action
conditioner), assembles the action(+text) condition, and provides an
autoregressive ``rollout`` for inference.

Latent-channel note: Wan/Cosmos operate on 16-channel latents from their own
VAEs, whereas the existing LIBERO dataset stores 4-channel SVD-VAE latents. To
train these backbones on robot data the dataset must be **re-encoded with the
backbone's VAE** (see ``docs/AUTOREGRESSIVE.md`` and
``scripts/preprocess_ar_latents.py`` TODO). ``cfg.in_channels`` reflects the
backbone's expectation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import build_backbone
from .conditioning.action import ActionConditioner
from .conditioning.multiview import ViewPacker, ViewEmbedding
from .distill.scheduler import FlowMatchScheduler
from .distill.self_forcing import generate_rollout


class ARWorldModel(nn.Module):
    def __init__(self, cfg, *, build_backbone_fn=build_backbone):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone_fn(cfg)
        self.conditioner = ActionConditioner(
            action_dim=cfg.action_dim,
            cross_attn_dim=cfg.cross_attn_dim,
            text_cond=cfg.text_cond,
            mode=cfg.action_cond_mode,
        )
        self.packer = ViewPacker(cfg.multiview_layout, cfg.num_cams)
        self.view_embed = (
            ViewEmbedding(cfg.num_cams, cfg.in_channels)
            if cfg.multiview_layout == "sequence_pack"
            else None
        )

    # -- act as a DiTBackbone (generator) -------------------------------
    def forward_train(self, latents, timestep, cond, **kw):
        return self.backbone.forward_train(latents, timestep, cond, **kw)

    def forward_cached(self, latent_block, timestep, cond, **kw):
        return self.backbone.forward_cached(latent_block, timestep, cond, **kw)

    def make_kv_cache(self, **kw):
        return self.backbone.make_kv_cache(**kw)

    def slice_cond_to_frames(self, cond, start_frame, num_frames):
        return self.backbone.slice_cond_to_frames(cond, start_frame, num_frames)

    @property
    def num_self_layers(self):
        return self.backbone.num_self_layers

    # -- conditioning ----------------------------------------------------
    def encode_cond(self, actions, texts=None, tokenizer=None, text_encoder=None, *, cfg_drop=None):
        return self.conditioner(
            actions, texts=texts, tokenizer=tokenizer, text_encoder=text_encoder,
            frame_level_cond=self.cfg.frame_level_cond, cfg_drop=cfg_drop,
        )

    def null_cond_like(self, cond):
        return torch.zeros_like(cond)

    # -- inference -------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        history_latents: list[torch.Tensor] | None,
        cond: torch.Tensor,
        *,
        num_blocks: int,
        latent_block_shape: tuple,
        scheduler: FlowMatchScheduler | None = None,
        max_kv_blocks: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate ``num_blocks`` latent blocks. Returns
        ``[B, num_blocks*fpb, C, H, W]`` (caller decodes with the backbone VAE)."""
        scheduler = scheduler or FlowMatchScheduler(
            self.cfg.denoising_step_list, num_train_timestep=self.cfg.num_train_timestep,
            warp=self.cfg.warp_denoising_step,
        )
        kv = self.make_kv_cache(max_blocks=max_kv_blocks or self.cfg.max_kv_blocks)
        blocks, _ = generate_rollout(
            self.backbone, cond, scheduler,
            frames_per_block=self.cfg.frames_per_block,
            num_blocks=num_blocks, latent_block_shape=latent_block_shape,
            history_blocks=history_latents, kv_cache=kv, last_step_grad=False,
        )
        return torch.cat(blocks, dim=1)


def build_training_stack(cfg, *, build_backbone_fn=build_backbone):
    """Construct (generator ARWorldModel, critic backbone, teacher backbone).

    * generator — the trainable causal student (+ action conditioner).
    * critic    — "fake score"; init from a fresh backbone (in practice: copy the
      student or the teacher and let it track the student online).
    * teacher   — frozen bidirectional "real score" (loads ``cfg.teacher_ckpt``).

    For real runs, point ``cfg.student_init_ckpt`` / ``cfg.teacher_ckpt`` at the
    robot-finetuned bidirectional checkpoint (the E1/E2 stage); the critic is
    typically initialised from the teacher.
    """
    generator = ARWorldModel(cfg, build_backbone_fn=build_backbone_fn)
    critic = build_backbone_fn(cfg)
    teacher = build_backbone_fn(cfg)
    return generator, critic, teacher
