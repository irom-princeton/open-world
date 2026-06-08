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
    timestep_proj = self.time_proj(self.act_fn(temb))                      # [N, proj]
    if perframe:
        temb = temb.unflatten(0, (B, T))                # [B, T, dim]
        timestep_proj = timestep_proj.unflatten(0, (B, T))  # [B, T, proj]
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


def _block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """WanTransformerBlock.forward supporting per-frame modulation.

    ``temb`` is the projection, either ``[B, 6, dim]`` (global) or
    ``[B, Fr, 6, dim]`` (per-frame). Per-frame modulation reshapes the token
    sequence so each frame's ``tpf`` tokens share that frame's (shift, scale,
    gate); no ``[B, seq, ...]`` tensor is materialized."""
    if temb.ndim == 4:  # per-frame: [B, Fr, 6, dim]
        Fr = temb.shape[1]
        tpf = hidden_states.shape[1] // Fr
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            m.squeeze(2) for m in (self.scale_shift_table.unsqueeze(1) + temb.float()).chunk(6, dim=2))
        # each [B, Fr, dim]
        norm_hidden_states = _affine_perframe(
            self.norm1(hidden_states.float()), scale_msa, shift_msa, Fr, tpf).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
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
    attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
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
):
    """WanTransformer3DModel.forward supporting ``timestep`` of shape [B] or [B, T]."""
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

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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
