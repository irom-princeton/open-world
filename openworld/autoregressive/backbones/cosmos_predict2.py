"""Cosmos-Predict2-2B backbone adapter (diffusers ``CosmosTransformer3DModel``).

Cosmos-Predict2 is the DiT that NVIDIA OmniDreams itself post-trains from, so it
is the closest analog to the driving model. Structurally it matches Wan (unified
3D self-attention ``attn1`` + cross-attention, RoPE), so we reuse the same
block-causal/KV-cache machinery via ``attach_block_causal_cosmos``.

``forward_train`` (full clip, block-causal mask) is exact and is what the
self-forcing generator / DMD critic / teacher use. ``forward_cached`` offsets
the Cosmos RoPE to the block's absolute frame positions (mirroring the Wan
backbone and OmniDreams' ``start_frame_for_rope`` slice) and lets the patched
``BlockCausalCosmosAttnProcessor`` read/grow the KV-cache, so cached
autoregressive rollout reproduces the masked forward exactly. See
``docs/AUTOREGRESSIVE.md``.
"""

from __future__ import annotations

import torch

from .base import DiTBackbone
from ._attn import attach_block_causal_cosmos
from ..causal.context import CausalContext
from ..causal.kv_cache import KVCache
from ..causal.mask import block_ids_for_video, dense_block_causal_mask


def _offset_rope_cosmos(rope, hidden_states: torch.Tensor, fps, frame_offset: int):
    """Reproduce ``CosmosRotaryPosEmbed.forward`` but with the temporal positions
    starting at absolute ``frame_offset`` instead of always at 0.

    Spatial (h/w) frequencies are unchanged; only the per-frame temporal index is
    shifted, so a block generated from a KV-cache lands at the same RoPE phase it
    had inside the full masked forward (-> cached rollout == masked forward)."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope.patch_size
    pe_size = [num_frames // p_t, height // p_h, width // p_w]
    device = hidden_states.device

    h_theta = 10000.0 * rope.h_ntk_factor
    w_theta = 10000.0 * rope.w_ntk_factor
    t_theta = 10000.0 * rope.t_ntk_factor

    # seq must reach the block's absolute end; spatial only indexes [:pe_size].
    seq_len = max(max(rope.max_size), frame_offset + pe_size[0])
    seq = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim_h_range = torch.arange(0, rope.dim_h, 2, device=device, dtype=torch.float32)[: rope.dim_h // 2] / rope.dim_h
    dim_w_range = torch.arange(0, rope.dim_w, 2, device=device, dtype=torch.float32)[: rope.dim_w // 2] / rope.dim_w
    dim_t_range = torch.arange(0, rope.dim_t, 2, device=device, dtype=torch.float32)[: rope.dim_t // 2] / rope.dim_t
    h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
    w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
    temporal_freqs = 1.0 / (t_theta**dim_t_range)

    emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
    emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

    t_seq = seq[frame_offset : frame_offset + pe_size[0]]
    if fps is None:
        emb_t = torch.outer(t_seq, temporal_freqs)
    else:
        emb_t = torch.outer(t_seq / fps * rope.base_fps, temporal_freqs)
    emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)

    freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
    return torch.cos(freqs), torch.sin(freqs)


class CosmosBackbone(DiTBackbone):
    def __init__(self, transformer, *, cross_attn_dim: int):
        super().__init__()
        self.transformer = transformer
        self.in_channels = transformer.config.in_channels
        self.cross_attn_dim = cross_attn_dim
        self.patch_spatial = transformer.config.patch_size[1]
        self.patch_temporal = transformer.config.patch_size[0]
        self.context = attach_block_causal_cosmos(transformer, CausalContext())
        self.num_self_layers = self.context.num_self_layers

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
        # Match the transformer's weight dtype (bf16 weights vs fp32 rollout
        # noise / conditioner) so the backbone is the single dtype boundary.
        dt = next(self.transformer.parameters()).dtype
        x_cfhw = x_cfhw.to(dt)
        padding_mask = torch.zeros(B, 1, H, W, device=x_cfhw.device, dtype=dt)
        out = self.transformer(
            hidden_states=x_cfhw, timestep=timestep,
            encoder_hidden_states=(cond.to(dt) if cond is not None else cond),
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
        x = self._call(self._to_cfhw(latents), timestep, cond)
        ctx.mode = "off"
        return self._to_fchw(x)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True):
        B, Fr, C, H, W = latent_block.shape
        ctx = self.context
        ctx.mode = "cache"
        ctx.kv_cache = kv_cache
        ctx.commit = commit
        # offset RoPE to this block's absolute frame positions (Cosmos's stock
        # rope always starts at frame 0); the patched attn proc reads/grows the cache.
        rope = self.transformer.rope
        orig_forward = rope.forward
        rope.forward = lambda hs, fps=None: _offset_rope_cosmos(rope, hs, fps, start_frame)  # type: ignore[assignment]
        try:
            x = self._call(self._to_cfhw(latent_block), timestep, cond)
        finally:
            rope.forward = orig_forward  # type: ignore[assignment]
            ctx.mode = "off"
        return self._to_fchw(x)
