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
from ..causal.kv_cache import KVCache, StaticLayerKVCache
from ..causal.mask import (
    block_ids_for_video, dense_block_causal_mask, dense_teacher_forcing_mask,
    frame_aligned_cross_mask,
)


def _widen_patch_embedding(transformer, extra: int) -> None:
    """Grow ``transformer.patch_embedding`` (Conv3d) input channels by ``extra``,
    copying the pretrained weights into the first channels and zero-initializing
    the new ones. Output channels / kernel / bias are unchanged, so the model is
    numerically identical to the baseline until the new channels receive signal."""
    old = transformer.patch_embedding
    in_old = old.in_channels
    new = torch.nn.Conv3d(in_old + extra, old.out_channels, kernel_size=old.kernel_size,
                          stride=old.stride, padding=old.padding,
                          bias=old.bias is not None).to(old.weight.device, old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :in_old].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    transformer.patch_embedding = new
    transformer.config.in_channels = in_old + extra


def _offset_rope(rope, hidden_states: torch.Tensor, frame_offset: int) -> torch.Tensor:
    """Reproduce ``WanRotaryPosEmbed.forward`` but with the temporal frequencies
    sliced at an absolute ``frame_offset`` instead of always starting at 0."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    ahd = rope.attention_head_dim
    # Cache the freqs table on-device once. The stock code re-runs ``.to(device)``
    # every call -- a host-side op that, if this ran inside a compiled/graphed
    # region, would force a graph partition. Caching keeps the per-step RoPE pure.
    freqs = getattr(rope, "_freqs_dev", None)
    if freqs is None or freqs.device != hidden_states.device:
        freqs = rope.freqs.to(hidden_states.device)
        rope._freqs_dev = freqs
    freqs = freqs.split_with_sizes([ahd // 2 - 2 * (ahd // 6), ahd // 6, ahd // 6], dim=1)
    f = freqs[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return torch.cat([f, h, w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)


def _rope_from_positions(rope, hidden_states: torch.Tensor, frame_pos) -> torch.Tensor:
    """Like ``_offset_rope`` but the temporal frequencies are gathered at ARBITRARY
    per-frame absolute positions ``frame_pos`` (a 1-D int tensor of length = post-patch
    frames) instead of a contiguous ``offset..offset+F``. Used for WEAVER-style sparse
    history where the priming frames sit at true temporal gaps (e.g. 0,4,8,12 then
    13,14,...). Spatial (h,w) freqs are unchanged. Returns ``[1,1,F'*H'*W',ahd/2]``."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    ahd = rope.attention_head_dim
    freqs = getattr(rope, "_freqs_dev", None)
    if freqs is None or freqs.device != hidden_states.device:
        freqs = rope.freqs.to(hidden_states.device)
        rope._freqs_dev = freqs
    freqs = freqs.split_with_sizes([ahd // 2 - 2 * (ahd // 6), ahd // 6, ahd // 6], dim=1)
    fpos = torch.as_tensor(frame_pos, device=freqs[0].device, dtype=torch.long).reshape(-1)
    assert fpos.numel() == ppf, f"frame_pos len {fpos.numel()} != post-patch frames {ppf}"
    f = freqs[0][fpos].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)     # gather at true positions
    h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return torch.cat([f, h, w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)


class WanBackbone(DiTBackbone):
    def __init__(self, transformer, *, cross_attn_dim: int,
                 action_mode: str = "cross_attn", action_frame_repeat: int = 1,
                 extra_in_channels: int = 0, state_pred: bool = False, state_pred_dim: int = 16):
        super().__init__()
        self.transformer = transformer
        # Pixel-space action conditioning: widen the patch-embed conv to accept K
        # extra (clean) input channels, zero-initialized so the model is identical
        # to the baseline at init and learns to use the anchor from there.
        self.extra_in_channels = int(extra_in_channels)
        if self.extra_in_channels > 0:
            _widen_patch_embedding(transformer, self.extra_in_channels)
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
        if action_mode in ("cross_attn_aligned", "adaln_aligned"):
            # per-frame cross-attn -> patch attn2 to honour ctx.cross_mask.
            attach_cross_align(transformer, self.context)
        if action_mode in ("adaln", "adaln_aligned"):
            # project the per-frame action token into the model dim and add it to
            # the AdaLN time embedding (consumed in wan_perframe._condition_embedder_forward).
            inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
            self.action_to_temb = torch.nn.Linear(cross_attn_dim, inner_dim)
            if action_mode == "adaln_aligned":
                # zero-init the AdaLN path so at step 0 it is a no-op and the model is
                # identical to cross_attn_aligned; it learns to use the always-on
                # modulation (conditioning strength) on top of per-frame binding.
                torch.nn.init.zeros_(self.action_to_temb.weight)
                torch.nn.init.zeros_(self.action_to_temb.bias)

        # -- optional auxiliary state-prediction head (Finding-2, off by default) ----
        # A small MLP over the per-frame pooled transformer feature [B, Fr, dim] that
        # predicts the absolute proprioceptive state of each frame. `_stash_state_feat`
        # flips on the (otherwise-absent) feature tap in wan_perframe._model_forward.
        self.state_pred = bool(state_pred)
        if self.state_pred:
            inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
            self.state_head = torch.nn.Sequential(
                torch.nn.Linear(inner_dim, inner_dim // 2), torch.nn.SiLU(),
                torch.nn.Linear(inner_dim // 2, int(state_pred_dim)),
            )
            transformer._stash_state_feat = True

    def predict_state(self) -> torch.Tensor:
        """Apply the aux state head to the per-frame feature stashed by the most recent
        forward. Returns [B, Fr, state_pred_dim]. Only valid when state_pred=True and a
        forward has run this step."""
        feat = getattr(self.transformer, "_state_feat", None)
        if feat is None:
            raise RuntimeError("predict_state() called but no _state_feat was stashed "
                               "(state_pred disabled, or no forward ran this step)")
        # feat comes from the bf16-autocast'd transformer forward; state_head holds the
        # fp32 master weights and this runs OUTSIDE autocast -> cast feat to the head's
        # dtype to avoid "mat1 and mat2 must have the same dtype" (bf16 vs fp32).
        return self.state_head(feat.to(next(self.state_head.parameters()).dtype))

    def compile_blocks(self, *, mode: str = "default", fullgraph: bool = False,
                       dynamic: bool | None = False) -> None:
        """Compile the transformer-block loop for faster inference (see
        ``wan_perframe.enable_block_compile``). No-op-safe to call once after load."""
        from .wan_perframe import enable_block_compile
        enable_block_compile(self.transformer, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    # -- KV cache --------------------------------------------------------
    def make_kv_cache(self, *, max_blocks: int | None = None, static: bool = False,
                      anchor_blocks: int = 0, memory_stride: int = 0,
                      memory_blocks: int = 0, recent_blocks: int | None = None) -> KVCache:
        """Build a rollout KV cache. ``static=True`` returns a fixed-shape ring cache
        whose per-layer K/V are **registered-buffer submodules attached onto each
        ``attn1``** (``attn1.kv``) -- module state the compiled block loop can track
        and CUDA-graph can persist. Attachment is idempotent and reuses existing
        submodules (stable buffer addresses across rollout resets), so a captured
        graph stays valid; only the ring bookkeeping resets.

        The ``anchor_blocks``/``memory_stride``/``memory_blocks``/``recent_blocks``
        knobs (consistency_aug) select anchored+strided+windowed eviction on the
        growing cache; they are unsupported for the static ring (raises in KVCache)."""
        if not static:
            return KVCache(self.num_self_layers, max_blocks=max_blocks,
                           anchor_blocks=anchor_blocks, memory_stride=memory_stride,
                           memory_blocks=memory_blocks, recent_blocks=recent_blocks)
        caches = self._attach_static_caches(max_blocks)
        return KVCache(self.num_self_layers, max_blocks=max_blocks, static=True, layer_caches=caches)

    def _attach_static_caches(self, max_blocks: int | None) -> list:
        """Attach (idempotently) one :class:`StaticLayerKVCache` submodule per
        self-attention block, in the same registration order as the processors'
        ``layer_idx``. Returns the per-layer caches (== ``KVCache.self_attn``)."""
        if max_blocks is None:
            raise ValueError("static KV cache requires a finite max_kv_blocks (the attention window)")
        caches = []
        for module in self.transformer.modules():
            if module.__class__.__name__ == "WanTransformerBlock":
                kv = getattr(module.attn1, "kv", None)
                if kv is None or kv.max_blocks != max_blocks:
                    kv = StaticLayerKVCache(max_blocks)
                    module.attn1.kv = kv          # registered as an nn.Module submodule
                caches.append(kv)
        return caches

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, repo_or_path: str, *, cross_attn_dim: int, torch_dtype=torch.bfloat16,
                        action_mode: str = "cross_attn", action_frame_repeat: int = 1,
                        extra_in_channels: int = 0, state_pred: bool = False, state_pred_dim: int = 16):
        from diffusers import WanTransformer3DModel
        tf = WanTransformer3DModel.from_pretrained(
            repo_or_path, subfolder="transformer", torch_dtype=torch_dtype
        )
        return cls(tf, cross_attn_dim=cross_attn_dim,
                   action_mode=action_mode, action_frame_repeat=action_frame_repeat,
                   extra_in_channels=extra_in_channels,
                   state_pred=state_pred, state_pred_dim=state_pred_dim)

    @classmethod
    def random_init(cls, *, cross_attn_dim: int = 4096, small: bool = True,
                    action_mode: str = "cross_attn", action_frame_repeat: int = 1,
                    extra_in_channels: int = 0, state_pred: bool = False, state_pred_dim: int = 16):
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
                   action_mode=action_mode, action_frame_repeat=action_frame_repeat,
                   extra_in_channels=extra_in_channels,
                   state_pred=state_pred, state_pred_dim=state_pred_dim)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _to_cfhw(latents):  # [B,F,C,H,W] -> [B,C,F,H,W]
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _to_fchw(latents):  # [B,C,F,H,W] -> [B,F,C,H,W]
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _call(self, x_cfhw, timestep, cond, static_kv=None, clean_x=None, pixel_cfhw=None):
        # Coerce inputs to the param dtype (rollout noise / conditioner may arrive
        # in another dtype); then, if autocast_dtype is set (fp32 master weights +
        # bf16 compute), run the heavy transformer matmuls/convs under autocast.
        dt = self.transformer.patch_embedding.weight.dtype
        x_cfhw = x_cfhw.to(dt)
        cond = cond.to(dt) if cond is not None else cond
        clean_x = clean_x.to(dt) if clean_x is not None else clean_x
        # Pixel-space action conditioning: append the (clean) heatmap channels to
        # the latent along C before the widened patch-embed. Kept out of the noised
        # target -- the trainer noises only the latent channels and predicts them.
        # The teacher-forcing/CD path runs the SAME widened patch-embed on clean_x
        # (wan_perframe: patch_embedding(clean_x)), so clean_x MUST receive the same
        # extra channels or the conv sees the wrong in_channels. clean_x shares each
        # frame's own pixel_cond (same frame positions as the noisy half).
        if pixel_cfhw is not None:
            p = pixel_cfhw.to(dt)
            x_cfhw = torch.cat([x_cfhw, p], dim=1)
            if clean_x is not None:
                clean_x = torch.cat([clean_x, p], dim=1)
        ac = self.autocast_dtype
        ctx = (torch.autocast("cuda", dtype=ac)
               if ac is not None and x_cfhw.is_cuda else contextlib.nullcontext())
        with ctx:
            out = self.transformer(
                hidden_states=x_cfhw, timestep=timestep,
                encoder_hidden_states=cond, return_dict=False,
                static_kv=static_kv, clean_x=clean_x,
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
        if mode in ("cross_attn", "cross_attn_pe", "none"):
            # global cross-attn (baseline / +PE); "none" feeds the null [B,1,D] token
            # from ActionConditioner -> no vector action signal.
            return cond, timestep
        rep = self.action_frame_repeat
        Lkv = cond.shape[1]
        if mode in ("cross_attn_aligned", "adaln_aligned"):
            self.context.cross_mask = frame_aligned_cross_mask(
                Fr, tpf, Lkv, frame_repeat=rep, device=cond.device)
            if mode == "cross_attn_aligned":
                return cond, timestep
            # adaln_aligned: ALSO drive the AdaLN modulation, but keep the real cond in
            # the (aligned) cross-attn -- per-frame binding + always-on strength.
            amod = self.action_to_temb(cond)            # [B, Lkv, dim]
            if rep != 1:
                amod = amod.repeat_interleave(rep, dim=1)
            if amod.shape[1] != Fr:
                raise ValueError(f"adaln_aligned action frames {amod.shape[1]} != latent frames {Fr} "
                                 f"(check action_frame_repeat / cond slicing)")
            ce._action_emb = amod
            timestep = self._as_perframe_timestep(timestep, B=cond.shape[0], Fr=Fr)
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
        if self.action_mode not in ("cross_attn_aligned", "adaln", "adaln_aligned"):
            return cond
        rep = self.action_frame_repeat
        lo, hi = start_frame // rep, (start_frame + Fr) // rep
        return cond[:, lo:hi]

    def slice_cond_to_frames(self, cond, start_frame, num_frames):
        # Same frame->action mapping the cached rollout uses per block, so the DMD
        # score path can align cond to a generated sub-window (see base docstring).
        return self._block_action_slice(cond, start_frame, num_frames)

    # -- forward modes ---------------------------------------------------
    def forward_train(self, latents, timestep, cond, *, frames_per_block, window=None, causal=True,
                      clean_x=None, pixel_cond=None, frame_pos=None):
        B, Fr, C, H, W = latents.shape
        tpf = (H // self.patch_spatial) * (W // self.patch_spatial)
        ctx = self.context
        if clean_x is not None:
            # Teacher-forcing / clean-context CD (Causal Forcing++ L2c). The token
            # sequence is doubled to [clean || noisy]; a per-frame timestep is
            # required (the clean half is fed timestep 0 inside the transformer).
            if not causal:
                raise ValueError("clean_x (teacher forcing) requires causal=True")
            bids = block_ids_for_video(Fr, tpf, frames_per_block, device=latents.device)
            ctx.mode = "train"
            ctx.dense_mask = dense_teacher_forcing_mask(bids, window=window)   # [2S, 2S]
            timestep = self._as_perframe_timestep(timestep, B=B, Fr=Fr)
        elif causal:
            bids = block_ids_for_video(Fr, tpf, frames_per_block, device=latents.device)
            ctx.mode = "train"
            ctx.dense_mask = dense_block_causal_mask(bids, bids, window=window)
        else:
            ctx.mode = "off"            # full bidirectional attention (teacher mid-training)
            ctx.dense_mask = None
        cond, timestep = self._prep_action(cond, Fr=Fr, tpf=tpf, timestep=timestep)
        if clean_x is not None:
            # The doubled [clean || noisy] sequence needs its per-frame action
            # conditioning doubled to match. The clean half shares each frame's own
            # action (same frame positions as the noisy half): repeat both the
            # aligned cross-attn mask ([S, Lkv] -> [2S, Lkv]) and the adaln action
            # embedding ([B, F, dim] -> [B, 2F, dim]). Global cross-attn modes carry
            # no per-frame state and need no change.
            if ctx.cross_mask is not None:
                ctx.cross_mask = torch.cat([ctx.cross_mask, ctx.cross_mask], dim=0)
            ce = self.transformer.condition_embedder
            if getattr(ce, "_action_emb", None) is not None:
                ce._action_emb = torch.cat([ce._action_emb, ce._action_emb], dim=1)
        # WEAVER-style sparse history: build RoPE from the clip's true temporal
        # positions (frame_pos), so strided history frames get their real gaps.
        # frame_pos [B, Fr]; all rows share the sampling pattern (per-sample offsets
        # cancel — RoPE is relative), so row 0 defines the positions. Monkeypatch
        # rope.forward for this call only (same technique as forward_cached).
        rope = self.transformer.rope
        orig_rope = rope.forward
        if frame_pos is not None:
            fp0 = frame_pos[0] if frame_pos.ndim == 2 else frame_pos
            if clean_x is not None:
                raise ValueError("frame_pos (sparse history) is not supported with clean_x")
            rope.forward = lambda hs: _rope_from_positions(rope, hs, fp0)  # type: ignore[assignment]
        try:
            x = self._call(self._to_cfhw(latents), timestep, cond,
                           clean_x=None if clean_x is None else self._to_cfhw(clean_x),
                           pixel_cfhw=None if pixel_cond is None else self._to_cfhw(pixel_cond))
        finally:
            rope.forward = orig_rope  # type: ignore[assignment]
            ctx.mode = "off"
            self._clear_action()
        return self._to_fchw(x)

    def forward_cached(self, latent_block, timestep, cond, *, kv_cache: KVCache, start_frame, commit=True,
                       pixel_cond=None, rope_pos=None):
        # ``rope_pos`` (WEAVER-style sparse history): override ONLY the RoPE frame offset
        # for this block, leaving ``start_frame`` to drive cond-slicing + cache write
        # order. So a strided/clamped history frame is committed in cache order but gets
        # its TRUE temporal position in RoPE (matching training's frame_pos). None ->
        # rope_pos == start_frame (the dense/contiguous default, unchanged).
        B, Fr, C, H, W = latent_block.shape
        tpf = (H // self.patch_spatial) * (W // self.patch_spatial)
        ctx = self.context
        ctx.mode = "cache"
        ctx.kv_cache = kv_cache
        ctx.commit = commit
        # Static (fixed-shape) cache: refresh the shared validity mask + ring write
        # position in eager mode before the (possibly compiled) transformer, keeping
        # the block-count bookkeeping out of the compiled region, then thread them in
        # as explicit inputs. No-op for the growing cache.
        static_kv = None
        if getattr(kv_cache, "static", False):
            # Buffer dtype must match the keys the growing cache stores, so the
            # windowed attention is numerically identical. The block-causal processor
            # bakes RoPE with ``type_as(key)`` and qk-norm restores fp32, so the
            # cached keys come out in the *param* dtype even under bf16 autocast (the
            # fp32-master path) -- NOT the autocast dtype. Allocating at the param
            # dtype keeps a pure-bf16 deploy at bf16 and the fp32-master path at fp32,
            # both matching the dynamic cache exactly (any other choice silently
            # rounds the cached keys -- a ~0.3% bf16 error that amplifies over the AR
            # rollout).
            kvd = self.transformer.patch_embedding.weight.dtype
            tcfg = self.transformer.config
            kv_cache.begin_forward(
                commit=commit, block_tok=Fr * tpf,
                num_heads=tcfg.num_attention_heads, head_dim=tcfg.attention_head_dim,
                batch=B, device=latent_block.device, dtype=kvd)
            static_kv = (kv_cache.attn_mask, kv_cache.write_pos, commit)
        cond = self._block_action_slice(cond, start_frame, Fr)
        cond, timestep = self._prep_action(cond, Fr=Fr, tpf=tpf, timestep=timestep)
        # offset RoPE to absolute frame positions for this block (rope_pos overrides
        # start_frame for sparse history; defaults to start_frame = contiguous).
        rpos = start_frame if rope_pos is None else int(rope_pos)
        rope = self.transformer.rope
        orig_forward = rope.forward
        rope.forward = lambda hs: _offset_rope(rope, hs, rpos)  # type: ignore[assignment]
        try:
            x = self._call(self._to_cfhw(latent_block), timestep, cond, static_kv=static_kv,
                           pixel_cfhw=None if pixel_cond is None else self._to_cfhw(pixel_cond))
        finally:
            rope.forward = orig_forward  # type: ignore[assignment]
            ctx.mode = "off"
            self._clear_action()
        return self._to_fchw(x)
