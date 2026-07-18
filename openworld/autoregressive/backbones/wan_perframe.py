"""Per-frame timestep support for the diffusers ``WanTransformer3DModel``.

Stock Wan applies a single timestep to every token (the AdaLN modulation
``scale_shift_table + temb`` broadcasts one ``[B, ...]`` embedding over the whole
sequence). Causal **diffusion forcing** (omni-dreams' L2a student-init, CausVid,
Self-Forcing) instead needs an *independent noise level per block*: within one
training clip, earlier blocks may be (near-)clean while the current block is
noisy, so the model learns the exact "clean context + noisy current" condition it
faces during the autoregressive KV-cache rollout.

``patch_for_perframe_timestep(transformer)`` monkeypatches a loaded transformer so
its ``forward`` accepts ``timestep`` of shape ``[B]`` (unchanged, global) **or**
``[B, T]`` (per latent frame). With a ``[B, T]`` timestep each frame's tokens get
their own AdaLN modulation -- applied by *reshaping* the token sequence to
``[B, T, tokens_per_frame, dim]`` and broadcasting ``[B, T, 1, dim]`` modulation,
so nothing of size ``[B, seq, ...]`` is materialized (same activation cost as the
stock global path). A ``[B, T]`` timestep whose frames are all equal is
numerically identical to the ``[B]`` path (verified in tests), so inference
(forward_cached, one timestep per block) is unaffected.
"""

from __future__ import annotations

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers


def _affine_perframe(x, scale, shift, Fr, tpf):
    """x [B, S=Fr*tpf, D]; scale/shift [B, Fr, D] -> per-frame ``x*(1+scale)+shift``."""
    x = x.unflatten(1, (Fr, tpf))                       # [B, Fr, tpf, D]
    x = x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
    return x.flatten(1, 2)                              # [B, S, D]


def _gate_perframe(x, gate, Fr, tpf):
    """x [B, S, D]; gate [B, Fr, D] -> per-frame gated ``x*gate``."""
    return (x.unflatten(1, (Fr, tpf)) * gate.unsqueeze(2)).flatten(1, 2)


def _kv_kwargs(static_kv):
    """Static-cache self-attn kwargs (empty when not static). Shared across layers:
    only the validity mask, ring write position, and commit flag are threaded as
    inputs -- the per-layer K/V live as registered buffers on each ``attn1.kv``."""
    if static_kv is None:
        return {}
    kv_mask, kv_wpos, kv_commit = static_kv
    return dict(kv_mask=kv_mask, kv_wpos=kv_wpos, kv_commit=kv_commit)


def _condition_embedder_forward(self, timestep, encoder_hidden_states, encoder_hidden_states_image=None):
    """WanTimeTextImageEmbedding.forward supporting ``timestep`` [B] or [B, T].

    For [B, T] the time embedding is computed per frame (flatten -> embed ->
    reshape), returning ``temb`` [B, T, dim] and ``timestep_proj`` [B, T, proj]."""
    perframe = timestep.ndim == 2
    if perframe:
        B, T = timestep.shape
        timestep = timestep.reshape(-1)                 # [B*T]
    timestep = self.timesteps_proj(timestep)
    time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
    if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
        timestep = timestep.to(time_embedder_dtype)
    temb = self.time_embedder(timestep).type_as(encoder_hidden_states)     # [N, dim]
    if perframe:
        temb = temb.unflatten(0, (B, T))                # [B, T, dim]
    # adaln action conditioning ("adaln" mode): add the per-frame action
    # embedding to the time embedding *before* the projection, so it modulates
    # every block's (shift, scale, gate) and the final norm. The backbone stashes
    # ``_action_emb`` [B, T, dim] and guarantees a per-frame timestep here; it is
    # None (default) for every other mode, leaving the stock path bit-identical.
    # (Reshaping temb to [B, T, dim] before the projection is mathematically the
    # same as projecting flat then reshaping -- time_proj acts on the last dim.)
    action_emb = getattr(self, "_action_emb", None)
    if action_emb is not None:
        assert perframe, "adaln action conditioning requires a per-frame timestep"
        temb = temb + action_emb.type_as(temb)          # [B, T, dim]
    timestep_proj = self.time_proj(self.act_fn(temb))   # [N, proj] or [B, T, proj]
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


def _block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                   kv_mask=None, kv_wpos=None, kv_commit=False):
    """WanTransformerBlock.forward supporting per-frame modulation.

    ``temb`` is the projection, either ``[B, 6, dim]`` (global) or
    ``[B, Fr, 6, dim]`` (per-frame). Per-frame modulation reshapes the token
    sequence so each frame's ``tpf`` tokens share that frame's (shift, scale,
    gate); no ``[B, seq, ...]`` tensor is materialized.

    ``kv_*`` (when set) thread this forward's static KV-cache validity mask + ring
    write position + commit flag to the self-attention processor as explicit inputs
    (so a compiled block loop tracks the in-place cache mutation, which lives in the
    ``attn1.kv`` registered buffers); only attn1 (self) gets them."""
    _kv = dict(kv_mask=kv_mask, kv_wpos=kv_wpos, kv_commit=kv_commit)
    if temb.ndim == 4:  # per-frame: [B, Fr, 6, dim]
        Fr = temb.shape[1]
        tpf = hidden_states.shape[1] // Fr
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            m.squeeze(2) for m in (self.scale_shift_table.unsqueeze(1) + temb.float()).chunk(6, dim=2))
        # each [B, Fr, dim]
        norm_hidden_states = _affine_perframe(
            self.norm1(hidden_states.float()), scale_msa, shift_msa, Fr, tpf).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, **_kv)
        hidden_states = (hidden_states.float()
                         + _gate_perframe(attn_output.float(), gate_msa, Fr, tpf)).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = _affine_perframe(
            self.norm3(hidden_states.float()), c_scale_msa, c_shift_msa, Fr, tpf).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float()
                         + _gate_perframe(ff_output.float(), c_gate_msa, Fr, tpf)).type_as(hidden_states)
        return hidden_states

    # global path (identical to stock WanTransformerBlock.forward)
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table + temb.float()).chunk(6, dim=1)
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb, **_kv)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
    hidden_states = hidden_states + attn_output
    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
    return hidden_states


def _model_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    return_dict=True,
    attention_kwargs=None,
    static_kv=None,
):
    """WanTransformer3DModel.forward supporting ``timestep`` of shape [B] or [B, T].

    ``static_kv`` (set by the backbone for the static-cache rollout) is
    ``(mask, write_pos, commit)`` -- the shared validity mask + ring write position +
    commit flag -- threaded to each block's self-attention as explicit inputs so a
    compiled block loop tracks the in-place cache mutation (the K/V themselves are
    registered buffers on each ``attn1.kv``). ``None`` -> the ctx-based path (dynamic
    cache / train / eager)."""
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w
    tpf = post_patch_height * post_patch_width            # tokens per (post-patch) frame

    rotary_emb = self.rope(hidden_states)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)   # [B, seq=(F'*H'*W'), dim]

    perframe = timestep.ndim == 2
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    if perframe:
        # temb [B, F, dim], timestep_proj [B, F, proj] -> [B, F, 6, dim] (NOT expanded
        # to seq -- the blocks reshape hidden_states per frame instead).
        assert timestep_proj.shape[1] == post_patch_num_frames, (
            f"per-frame timestep T={timestep_proj.shape[1]} != post-patch frames {post_patch_num_frames}")
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))  # [B, F, 6, dim]
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))   # [B, 6, dim]

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    runner = getattr(self, "_compiled_block_runner", None)
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    elif runner is not None:
        # Compiled block loop (inference only). RoPE (start_frame-dependent) is
        # computed above in eager mode and enters only as the ``rotary_emb`` tensor;
        # the static KV cache enters as ``static_kv`` (mask/write_pos/commit) explicit
        # inputs while its K/V are persistent ``attn1.kv`` buffers -- so shapes are
        # constant and the in-place cache mutation is tracked. One graph per
        # commit/denoise. See ``enable_block_compile``.
        hidden_states = runner(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, static_kv)
    else:
        _kv = _kv_kwargs(static_kv)
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, **_kv)

    # Auxiliary state-prediction feature tap (WEAVER-style joint obs+state
    # prediction): pool the transformer's per-frame hidden states over spatial tokens
    # -> [B, Fr, D], stashed for the backbone's state head. Gated by `_stash_state_feat`
    # so it is a no-op (and zero overhead) unless state_pred is enabled.
    if getattr(self, "_stash_state_feat", False):
        Fr_sf = hidden_states.shape[1] // tpf
        self._state_feat = hidden_states.unflatten(1, (Fr_sf, tpf)).mean(dim=2)   # [B, Fr, D]

    # Output norm, projection & unpatchify
    if perframe:
        Fr = temb.shape[1]
        mod = self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2).float()  # [1,1,2,dim]+[B,F,1,dim]
        shift, scale = (m.squeeze(2) for m in mod.chunk(2, dim=2))             # each [B, F, dim]
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = _affine_perframe(
            self.norm_out(hidden_states.float()), scale, shift, Fr, tpf).type_as(hidden_states)
    else:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)  # each [B, 1, dim]
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def enable_block_compile(transformer, *, mode: str = "default", fullgraph: bool = False,
                         dynamic: bool | None = False) -> None:
    """Compile the transformer-block loop for faster inference (idempotent).

    Compiles only the per-block stack -- not ``self.rope`` / patch-embed / output
    projection -- so the absolute ``start_frame`` (which changes every rollout step)
    affects the compiled region only through the precomputed ``rotary_emb`` tensor,
    keeping shapes stable across steps. The KV cache mutates inside the blocks, so
    Dynamo may insert graph breaks there; ``mode="default"`` tolerates that, whereas
    ``mode="reduce-overhead"`` (CUDA graphs) needs the cache shapes to have settled.

    Inference-only: the grad/gradient-checkpointing path in ``_model_forward`` always
    bypasses the compiled runner, so training is unaffected.
    """
    import torch

    if getattr(transformer, "_compiled_block_runner", None) is not None:
        return
    blocks = transformer.blocks

    def _runner(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, static_kv=None):
        _kv = _kv_kwargs(static_kv)
        for block in blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, **_kv)
        return hidden_states

    transformer._compiled_block_runner = torch.compile(
        _runner, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


def patch_for_perframe_timestep(transformer) -> None:
    """Bind per-frame ``forward`` onto a WanTransformer3DModel and its blocks.

    Idempotent: re-patching is a no-op. The patched forwards are a strict superset
    of the originals -- a ``[B]`` (or all-equal ``[B, T]``) timestep reproduces the
    stock output bit-for-bit.
    """
    import types

    if getattr(transformer, "_perframe_patched", False):
        return
    transformer.forward = types.MethodType(_model_forward, transformer)
    transformer.condition_embedder.forward = types.MethodType(
        _condition_embedder_forward, transformer.condition_embedder)
    for block in transformer.blocks:
        block.forward = types.MethodType(_block_forward, block)
    transformer._perframe_patched = True
