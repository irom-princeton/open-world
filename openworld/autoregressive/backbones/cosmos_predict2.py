"""Cosmos-Predict2-2B backbone adapter (diffusers ``CosmosTransformer3DModel``).

Cosmos-Predict2 is the DiT that NVIDIA OmniDreams itself post-trains from, so it
is the closest analog to the driving model. Structurally it matches Wan (unified
3D self-attention ``attn1`` + cross-attention, RoPE), so we reuse the same
block-causal/KV-cache machinery via ``attach_block_causal_cosmos``.

``forward_train`` (full clip, block-causal mask) is exact and is what the
self-forcing generator / DMD critic / teacher use. ``forward_cached`` needs the
Cosmos RoPE sliced to absolute frame positions; that offset is not yet wired
here (Cosmos's ``CosmosRotaryPosEmbed`` differs from Wan's and needs its own
slice), so cached rollout currently raises with guidance — use the Wan backbone
for validated cached rollout, or train Cosmos with ``forward_train`` and run the
masked rollout. See ``docs/AUTOREGRESSIVE.md``.
"""

from __future__ import annotations

import torch

from .base import DiTBackbone
from ..causal.convert import CausalContext, attach_block_causal_cosmos
from ..causal.kv_cache import KVCache
from ..causal.mask import block_ids_for_video, dense_block_causal_mask


class CosmosBackbone(DiTBackbone):
    def __init__(self, transformer, *, cross_attn_dim: int):
        super().__init__()
        self.transformer = transformer
        self.in_channels = transformer.config.in_channels
        self.cross_attn_dim = cross_attn_dim
        self.patch_spatial = transformer.config.patch_size[1]
        self.patch_temporal = transformer.config.patch_size[0]
        self.context = attach_block_causal_cosmos(transformer, CausalContext())
        self.num_self_layers = self.context._num_self_layers

    @classmethod
    def from_pretrained(cls, repo_or_path: str, *, cross_attn_dim: int, torch_dtype=torch.bfloat16):
        from diffusers import CosmosTransformer3DModel
        tf = CosmosTransformer3DModel.from_pretrained(
            repo_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )
        return cls(tf, cross_attn_dim=cross_attn_dim)

    @classmethod
    def random_init(cls, *, cross_attn_dim: int = 1024, small: bool = True):
        from diffusers import CosmosTransformer3DModel
        if small:
            tf = CosmosTransformer3DModel(
                in_channels=16, out_channels=16, num_attention_heads=4, attention_head_dim=32,
                num_layers=2, text_embed_dim=cross_attn_dim, adaln_lora_dim=64,
                max_size=(16, 48, 48), patch_size=(1, 2, 2),
            )
        else:  # Cosmos-Predict2-2B shape
            tf = CosmosTransformer3DModel(
                in_channels=16, out_channels=16, num_attention_heads=16, attention_head_dim=128,
                num_layers=28, text_embed_dim=cross_attn_dim, adaln_lora_dim=256,
                max_size=(128, 240, 240), patch_size=(1, 2, 2),
            )
        return cls(tf, cross_attn_dim=cross_attn_dim)

    @staticmethod
    def _to_cfhw(x):
        return x.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _to_fchw(x):
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def _call(self, x_cfhw, timestep, cond):
        B, C, F, H, W = x_cfhw.shape
        # Cosmos is built with concat_padding_mask=True and expects a [B,1,H,W]
        # padding mask (all-zero = no padding for our square robot latents).
        padding_mask = torch.zeros(B, 1, H, W, device=x_cfhw.device, dtype=x_cfhw.dtype)
        out = self.transformer(
            hidden_states=x_cfhw, timestep=timestep, encoder_hidden_states=cond,
            padding_mask=padding_mask, return_dict=False,
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward_train(self, latents, timestep, cond, *, frames_per_block, window=None):
        B, Fr, C, H, W = latents.shape
        tpf = (H // self.patch_spatial) * (W // self.patch_spatial)
        bids = block_ids_for_video(Fr, tpf, frames_per_block, device=latents.device)
        ctx = self.context
        ctx.mode = "train"
        ctx.dense_mask = dense_block_causal_mask(bids, bids, window=window)
        ctx.begin()
        x = self._call(self._to_cfhw(latents), timestep, cond)
        ctx.mode = "off"
        return self._to_fchw(x)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True):
        raise NotImplementedError(
            "Cosmos cached rollout needs CosmosRotaryPosEmbed sliced to absolute "
            "frame positions (start_frame), which is not wired yet. Use the Wan "
            "backbone for validated cached rollout, or run a masked rollout via "
            "forward_train. See docs/AUTOREGRESSIVE.md."
        )
