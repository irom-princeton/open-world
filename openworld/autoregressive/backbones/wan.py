"""Wan2.1 backbone adapter (diffusers ``WanTransformer3DModel``).

Wan is a DiT with unified 3D self-attention (``attn1``) + text cross-attention
(``attn2``) and complex RoPE — the substrate the Self-Forcing / CausVid recipe
was built on, which is why it is the recommended base over the SVD UNet. We load
the public Wan2.1-T2V-1.3B transformer, swap its self-attention for the
block-causal + KV-cached processor (``backbones/_attn.py``), and expose the two
forward modes of :class:`DiTBackbone`.

* ``forward_train`` runs the full clip; RoPE is computed over the whole sequence
  by Wan, so the block-causal mask path is exact.
* ``forward_cached`` runs one block, offsetting RoPE to the block's absolute
  frame positions (Wan's stock ``rope`` always starts at frame 0) and letting
  the patched processors read/grow the KV-cache.

Loading the real 1.3B weights needs a GPU + the HF download; ``random_init``
builds a small config so the wiring is unit-testable on CPU.
"""

from __future__ import annotations

import contextlib

import torch

from .base import DiTBackbone
from ._attn import attach_block_causal
from ..causal.context import CausalContext
from ..causal.kv_cache import KVCache
from ..causal.mask import block_ids_for_video, dense_block_causal_mask


def _offset_rope(rope, hidden_states: torch.Tensor, frame_offset: int) -> torch.Tensor:
    """Reproduce ``WanRotaryPosEmbed.forward`` but with the temporal frequencies
    sliced at an absolute ``frame_offset`` instead of always starting at 0."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    ahd = rope.attention_head_dim
    freqs = rope.freqs.to(hidden_states.device)
    freqs = freqs.split_with_sizes([ahd // 2 - 2 * (ahd // 6), ahd // 6, ahd // 6], dim=1)
    f = freqs[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return torch.cat([f, h, w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)


class WanBackbone(DiTBackbone):
    def __init__(self, transformer, *, cross_attn_dim: int):
        super().__init__()
        self.transformer = transformer
        self.in_channels = transformer.config.in_channels
        self.cross_attn_dim = cross_attn_dim
        self.patch_spatial = transformer.config.patch_size[1]
        self.patch_temporal = transformer.config.patch_size[0]
        self.context = attach_block_causal(transformer, CausalContext())
        self.num_self_layers = self.context.num_self_layers
        # Per-frame timestep modulation (diffusion forcing). Composes with the
        # causal processors above: it overrides block.forward but still routes
        # self-attention through attn1.processor. A scalar [B] timestep path is
        # unchanged, so inference (forward_cached, one t per block) is unaffected.
        from .wan_perframe import patch_for_perframe_timestep
        patch_for_perframe_timestep(transformer)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_or_path: str, *, cross_attn_dim: int, torch_dtype=torch.bfloat16):
        from diffusers import WanTransformer3DModel
        tf = WanTransformer3DModel.from_pretrained(
            repo_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )
        return cls(tf, cross_attn_dim=cross_attn_dim)

    @classmethod
    def random_init(cls, *, cross_attn_dim: int = 4096, small: bool = True):
        """Build an untrained Wan transformer. ``small`` shrinks it so CPU/CI can
        instantiate it; drop ``small`` to match the real 1.3B shape."""
        from diffusers import WanTransformer3DModel
        if small:
            tf = WanTransformer3DModel(
                patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=32,
                in_channels=16, out_channels=16, text_dim=cross_attn_dim, freq_dim=256,
                ffn_dim=512, num_layers=2, rope_max_seq_len=1024,
            )
        else:  # Wan2.1-1.3B
            tf = WanTransformer3DModel(
                patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128,
                in_channels=16, out_channels=16, text_dim=cross_attn_dim, freq_dim=256,
                ffn_dim=8960, num_layers=30, rope_max_seq_len=1024,
            )
        return cls(tf, cross_attn_dim=cross_attn_dim)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _to_cfhw(latents):  # [B,F,C,H,W] -> [B,C,F,H,W]
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _to_fchw(latents):  # [B,C,F,H,W] -> [B,F,C,H,W]
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _call(self, x_cfhw, timestep, cond):
        # Coerce inputs to the param dtype (rollout noise / conditioner may arrive
        # in another dtype); then, if autocast_dtype is set (fp32 master weights +
        # bf16 compute), run the heavy transformer matmuls/convs under autocast.
        dt = self.transformer.patch_embedding.weight.dtype
        x_cfhw = x_cfhw.to(dt)
        cond = cond.to(dt) if cond is not None else cond
        ac = self.autocast_dtype
        ctx = (torch.autocast("cuda", dtype=ac)
               if ac is not None and x_cfhw.is_cuda else contextlib.nullcontext())
        with ctx:
            out = self.transformer(
                hidden_states=x_cfhw, timestep=timestep,
                encoder_hidden_states=cond, return_dict=False,
            )
        return out[0] if isinstance(out, (tuple, list)) else out

    # -- forward modes ---------------------------------------------------
    def forward_train(self, latents, timestep, cond, *, frames_per_block, window=None, causal=True):
        B, Fr, C, H, W = latents.shape
        tpf = (H // self.patch_spatial) * (W // self.patch_spatial)
        ctx = self.context
        if causal:
            bids = block_ids_for_video(Fr, tpf, frames_per_block, device=latents.device)
            ctx.mode = "train"
            ctx.dense_mask = dense_block_causal_mask(bids, bids, window=window)
        else:
            ctx.mode = "off"            # full bidirectional attention (teacher mid-training)
            ctx.dense_mask = None
        x = self._call(self._to_cfhw(latents), timestep, cond)
        ctx.mode = "off"
        return self._to_fchw(x)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True):
        B, Fr, C, H, W = latent_block.shape
        ctx = self.context
        ctx.mode = "cache"
        ctx.kv_cache = kv_cache
        ctx.commit = commit
        # offset RoPE to absolute frame positions for this block.
        rope = self.transformer.rope
        orig_forward = rope.forward
        rope.forward = lambda hs: _offset_rope(rope, hs, start_frame)  # type: ignore[assignment]
        try:
            x = self._call(self._to_cfhw(latent_block), timestep, cond)
        finally:
            rope.forward = orig_forward  # type: ignore[assignment]
            ctx.mode = "off"
        return self._to_fchw(x)
