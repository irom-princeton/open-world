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
from ._attn import attach_block_causal, attach_cross_align
from ..causal.context import CausalContext
from ..causal.kv_cache import KVCache
from ..causal.mask import block_ids_for_video, dense_block_causal_mask, frame_aligned_cross_mask


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
    def __init__(self, transformer, *, cross_attn_dim: int,
                 action_mode: str = "cross_attn", action_frame_repeat: int = 1):
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

        # -- action-conditioning mode (see config.ARWMArgs.action_cond_mode) ----
        self.action_mode = action_mode
        # packed latent frames per real (action) frame: 1 for height_stack, V for
        # the time-major sequence_pack layout. Drives the per-frame alignment.
        self.action_frame_repeat = action_frame_repeat
        if action_mode == "cross_attn_aligned":
            # per-frame cross-attn -> patch attn2 to honour ctx.cross_mask.
            attach_cross_align(transformer, self.context)
        elif action_mode == "adaln":
            # project the per-frame action token into the model dim and add it to
            # the AdaLN time embedding (consumed in wan_perframe._condition_embedder_forward).
            inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
            self.action_to_temb = torch.nn.Linear(cross_attn_dim, inner_dim)

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_or_path: str, *, cross_attn_dim: int, torch_dtype=torch.bfloat16,
                        action_mode: str = "cross_attn", action_frame_repeat: int = 1):
        from diffusers import WanTransformer3DModel
        tf = WanTransformer3DModel.from_pretrained(
            repo_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )
        return cls(tf, cross_attn_dim=cross_attn_dim,
                   action_mode=action_mode, action_frame_repeat=action_frame_repeat)

    @classmethod
    def random_init(cls, *, cross_attn_dim: int = 4096, small: bool = True,
                    action_mode: str = "cross_attn", action_frame_repeat: int = 1):
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
        return cls(tf, cross_attn_dim=cross_attn_dim,
                   action_mode=action_mode, action_frame_repeat=action_frame_repeat)

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

    # -- action conditioning ---------------------------------------------
    @staticmethod
    def _as_perframe_timestep(timestep, *, B, Fr):
        """Coerce a scalar/[B] timestep to per-frame [B, Fr] (all-equal). The
        per-frame Wan path is numerically identical to the global one when the
        frames share a timestep, so this only *enables* the AdaLN modulation
        path; it does not change the denoising level."""
        if timestep.ndim == 2:
            return timestep
        t = timestep.reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)
        return t[:, None].expand(B, Fr).contiguous()

    def _prep_action(self, cond, *, Fr, tpf, timestep):
        """Mode-specific prep for one forward. ``cond`` is the action tensor for
        THIS forward (already sliced to the block in the cached path). Returns
        ``(cross_attn_context, timestep)`` and, as side effects, sets
        ``ctx.cross_mask`` (aligned) / stashes ``_action_emb`` (adaln)."""
        ce = self.transformer.condition_embedder
        ce._action_emb = None
        self.context.cross_mask = None
        mode = self.action_mode
        if mode in ("cross_attn", "cross_attn_pe"):
            return cond, timestep                       # global cross-attn (baseline / +PE)
        rep = self.action_frame_repeat
        Lkv = cond.shape[1]
        if mode == "cross_attn_aligned":
            self.context.cross_mask = frame_aligned_cross_mask(
                Fr, tpf, Lkv, frame_repeat=rep, device=cond.device)
            return cond, timestep
        if mode == "adaln":
            amod = self.action_to_temb(cond)            # [B, Lkv, dim]
            if rep != 1:
                amod = amod.repeat_interleave(rep, dim=1)
            if amod.shape[1] != Fr:
                raise ValueError(f"adaln action frames {amod.shape[1]} != latent frames {Fr} "
                                 f"(check action_frame_repeat / cond slicing)")
            ce._action_emb = amod
            timestep = self._as_perframe_timestep(timestep, B=cond.shape[0], Fr=Fr)
            # action goes solely through AdaLN -> feed a null cross-attn context.
            return cond.new_zeros(cond.shape[0], 1, self.cross_attn_dim), timestep
        raise ValueError(f"unknown action_mode {mode!r}")

    def _clear_action(self):
        self.transformer.condition_embedder._action_emb = None
        self.context.cross_mask = None

    def _block_action_slice(self, cond, start_frame, Fr):
        """Slice the full action sequence to the real frames covered by a cached
        block (aligned / adaln only; the global modes feed the full cond)."""
        if self.action_mode not in ("cross_attn_aligned", "adaln"):
            return cond
        rep = self.action_frame_repeat
        lo, hi = start_frame // rep, (start_frame + Fr) // rep
        return cond[:, lo:hi]

    def slice_cond_to_frames(self, cond, start_frame, num_frames):
        # Same frame->action mapping the cached rollout uses per block, so the DMD
        # score path can align cond to a generated sub-window (see base docstring).
        return self._block_action_slice(cond, start_frame, num_frames)

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
        cond, timestep = self._prep_action(cond, Fr=Fr, tpf=tpf, timestep=timestep)
        try:
            x = self._call(self._to_cfhw(latents), timestep, cond)
        finally:
            ctx.mode = "off"
            self._clear_action()
        return self._to_fchw(x)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True):
        B, Fr, C, H, W = latent_block.shape
        tpf = (H // self.patch_spatial) * (W // self.patch_spatial)
        ctx = self.context
        ctx.mode = "cache"
        ctx.kv_cache = kv_cache
        ctx.commit = commit
        cond = self._block_action_slice(cond, start_frame, Fr)
        cond, timestep = self._prep_action(cond, Fr=Fr, tpf=tpf, timestep=timestep)
        # offset RoPE to absolute frame positions for this block.
        rope = self.transformer.rope
        orig_forward = rope.forward
        rope.forward = lambda hs: _offset_rope(rope, hs, start_frame)  # type: ignore[assignment]
        try:
            x = self._call(self._to_cfhw(latent_block), timestep, cond)
        finally:
            rope.forward = orig_forward  # type: ignore[assignment]
            ctx.mode = "off"
            self._clear_action()
        return self._to_fchw(x)
