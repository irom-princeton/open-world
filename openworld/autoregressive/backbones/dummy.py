"""A tiny, real DiT that implements the backbone contract on CPU with no
downloaded weights. Used by the unit tests to prove the block-causal training
mask and the KV-cache rollout produce identical outputs.

It is a faithful miniature: patch-embed -> N x (block-causal self-attn +
cross-attn + MLP, AdaLN-timestep) -> unpatch to velocity. Everything except the
self-attention is per-token, so masked-full and cached-rollout forwards agree
exactly when the timestep is shared across frames (which the equivalence test
enforces). Absolute sin-cos position embeddings are indexed by absolute frame so
the cached path (offset by ``start_frame``) matches the full path token-for-token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import DiTBackbone
from ..causal.attention import causal_sdpa, full_attention
from ..causal.context import CausalContext
from ..causal.kv_cache import KVCache
from ..causal.mask import block_ids_for_video, dense_block_causal_mask


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    omega = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / half)
    args = positions.float()[:, None] * omega[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, cross_dim: int):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qc = nn.Linear(dim, dim)
        self.kc = nn.Linear(cross_dim, dim)
        self.vc = nn.Linear(cross_dim, dim)
        self.oc = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        # AdaLN: timestep -> per-layer (shift, scale) for the self-attn norm.
        self.ada = nn.Linear(dim, 2 * dim)

    def _heads(self, x):  # [B,S,D] -> [B,H,S,hd]
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.hd).transpose(1, 2)

    def _merge(self, x):  # [B,H,S,hd] -> [B,S,D]
        B, H, S, hd = x.shape
        return x.transpose(1, 2).reshape(B, S, H * hd)

    def forward(self, x, temb, cond, *, layer_idx, ctx: CausalContext):
        # temb is [B, D] (global -> broadcast over tokens) or [B, S, D] (per-frame,
        # already expanded to one embedding per token).
        shift, scale = self.ada(temb).chunk(2, dim=-1)
        if shift.ndim == 2:                                   # [B, D] -> [B, 1, D]
            shift, scale = shift[:, None], scale[:, None]
        h = self.norm1(x) * (1 + scale) + shift
        q, k, v = self._heads(self.q(h)), self._heads(self.k(h)), self._heads(self.v(h))
        # same dispatch as the real Wan/Cosmos processors
        attn = causal_sdpa(ctx, q, k, v, layer_idx)
        x = x + self.o(self._merge(attn))
        # cross-attention to the (constant) condition
        hc = self.norm2(x)
        qc = self._heads(self.qc(hc))
        kc, vc = self._heads(self.kc(cond)), self._heads(self.vc(cond))
        x = x + self.oc(self._merge(full_attention(qc, kc, vc)))
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


class DummyDiT(DiTBackbone):
    def __init__(
        self,
        *,
        in_channels: int = 16,
        dim: int = 64,
        heads: int = 4,
        layers: int = 3,
        cross_attn_dim: int = 64,
        patch_spatial: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attn_dim = cross_attn_dim
        self.num_self_layers = layers
        self.patch_spatial = patch_spatial
        self.patch_temporal = 1
        self.dim = dim
        patch = in_channels * patch_spatial * patch_spatial
        self.patch_embed = nn.Linear(patch, dim)
        self.unpatch = nn.Linear(dim, patch)
        self.t_embed = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_Layer(dim, heads, cross_attn_dim) for _ in range(layers)])

    # -- (un)patchify ----------------------------------------------------
    def _patchify(self, x):  # [B,F,C,H,W] -> tokens [B, F*hp*wp, D], (hp, wp)
        B, Fr, C, H, W = x.shape
        ps = self.patch_spatial
        hp, wp = H // ps, W // ps
        x = x.reshape(B, Fr, C, hp, ps, wp, ps)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, Fr * hp * wp, C * ps * ps)
        return self.patch_embed(x), (Fr, hp, wp)

    def _unpatchify(self, tok, shape, C):
        B = tok.shape[0]
        Fr, hp, wp = shape
        ps = self.patch_spatial
        x = self.unpatch(tok).reshape(B, Fr, hp, wp, C, ps, ps)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Fr, C, hp * ps, wp * ps)
        return x

    def _t_emb(self, timestep, B, Fr, tpf, device):
        t = timestep.float()
        if t.ndim == 2:                                  # [B, Fr] per-frame timestep
            emb = self.t_embed(_sincos_1d(t.reshape(-1), self.dim))   # [B*Fr, dim]
            return emb.view(B, Fr, self.dim).repeat_interleave(tpf, dim=1)  # [B, Fr*tpf, dim]
        t = t.reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)
        return self.t_embed(_sincos_1d(t, self.dim))     # [B, dim]

    def _pos(self, num_frames, tokens_per_frame, start_frame, device):
        # absolute frame index per token (offset by start_frame in *frame* units)
        frame_idx = (torch.arange(num_frames, device=device) + start_frame).repeat_interleave(tokens_per_frame)
        # intra-frame token index so spatial positions differ within a frame
        spatial = torch.arange(tokens_per_frame, device=device).repeat(num_frames)
        return _sincos_1d(frame_idx, self.dim) + _sincos_1d(spatial + 1000, self.dim)

    # -- forward modes ---------------------------------------------------
    def forward_train(self, latents, timestep, cond, *, frames_per_block, window=None, causal=True):
        B, Fr, C, H, W = latents.shape
        tok, shp = self._patchify(latents)
        tpf = shp[1] * shp[2]
        tok = tok + self._pos(Fr, tpf, 0, latents.device)[None]
        temb = self._t_emb(timestep, B, Fr, tpf, latents.device)
        if causal:
            bids = block_ids_for_video(Fr, tpf, frames_per_block, device=latents.device)
            ctx = CausalContext(mode="train", dense_mask=dense_block_causal_mask(bids, bids, window=window))
        else:
            ctx = CausalContext(mode="off")     # full bidirectional attention (teacher)
        for i, blk in enumerate(self.blocks):
            tok = blk(tok, temb, cond, layer_idx=i, ctx=ctx)
        return self._unpatchify(tok, shp, C)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True):
        B, Fr, C, H, W = latent_block.shape
        tok, shp = self._patchify(latent_block)
        tpf = shp[1] * shp[2]
        tok = tok + self._pos(Fr, tpf, start_frame, latent_block.device)[None]
        temb = self._t_emb(timestep, B, Fr, tpf, latent_block.device)
        ctx = CausalContext(mode="cache", kv_cache=kv_cache, commit=commit)
        for i, blk in enumerate(self.blocks):
            tok = blk(tok, temb, cond, layer_idx=i, ctx=ctx)
        return self._unpatchify(tok, shp, C)
