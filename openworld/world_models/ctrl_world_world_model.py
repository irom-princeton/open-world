"""Ctrl-World adapter for OpenWorld.

This wraps the vendored Ctrl-World model (``external/ctrl_world``) and exposes
it through the :class:`WorldModel` interface, alongside :class:`VidWMWorldModel`.

The two backbones share almost all I/O conventions (SVD-style latent space,
3 stacked camera views, 5 future / 6 history frames, 7-D actions), so this
class subclasses :class:`VidWMWorldModel` to reuse its state/history/action
plumbing.  Only the three things that genuinely differ are overridden:

  * ``load_checkpoint`` — instantiates ``CrtlWorld`` and loads its flat
    ``state_dict`` directly (vidwm splits weights by prefix).
  * ``rollout`` — calls ``CtrlWorldDiffusionPipeline`` without flow-matching
    kwargs and uses ``Action_encoder2`` (returns a tensor, not a dict).
  * ``_debug_log_rollout_inputs`` — does not reference flow-matching fields.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from openworld.world_models.ctrl_world_loader import ensure_ctrl_world_on_path
from openworld.world_models.vidwm_world_model import VidWMConfig, VidWMWorldModel

logger = logging.getLogger(__name__)


@dataclass
class CtrlWorldConfig(VidWMConfig):
    """Configuration for the Ctrl-World world model.

    Inherits all shape / view / normalization / history fields from
    :class:`VidWMConfig`.  Defaults that differ between the two backbones
    (inference steps, guidance scale, history sparsity) are overridden here.

    Flow-matching fields inherited from VidWMConfig are unused and ignored.
    """

    # Inference parameters – Ctrl-World uses standard SVD scheduler.
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    min_guidance_scale: float = 1.0
    noise_aug_strength: float = 0.02

    # Ctrl-World's rollout reference script uses a denser sparse history
    # (see scripts/rollout_replay_traj.py:338 in upstream).
    history_idx: tuple[int, ...] = (0, 0, -8, -6, -4, -2)


class CtrlWorldWorldModel(VidWMWorldModel):
    """World model backed by the vendored Ctrl-World diffusion model."""

    def __init__(self, config: Optional[Union[CtrlWorldConfig, Dict[str, Any]]] = None, **kwargs: Any):
        if config is None:
            config = CtrlWorldConfig(**kwargs)
        elif isinstance(config, dict):
            config = CtrlWorldConfig(**config)
        super().__init__(config=config)

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load Ctrl-World pipeline + UNet + action encoder + CLIP."""
        cfg = self.config

        ensure_ctrl_world_on_path(cfg.repo_path)

        from ctrl_world.ctrl_world import CrtlWorld  # noqa: E402

        # CrtlWorld(args) expects argparse-style attribute access.
        svd_path = cfg.svd_model_path
        if svd_path is None:
            from huggingface_hub import snapshot_download

            svd_path = snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid")

        clip_path = cfg.clip_model_path
        if clip_path is None:
            from huggingface_hub import snapshot_download

            clip_path = snapshot_download(repo_id="openai/clip-vit-base-patch32")

        ctrl_args = SimpleNamespace(
            svd_model_path=svd_path,
            clip_model_path=clip_path,
            action_dim=cfg.action_dim,
            num_history=cfg.num_history,
            num_frames=cfg.num_frames,
            text_cond=cfg.text_cond,
            frame_level_cond=cfg.frame_level_cond,
            his_cond_zero=cfg.his_cond_zero,
            motion_bucket_id=cfg.motion_bucket_id,
            fps=cfg.fps,
        )

        logger.info("Building CrtlWorld with SVD=%s CLIP=%s", svd_path, clip_path)
        model = CrtlWorld(ctrl_args)

        logger.info("Loading Ctrl-World weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("CrtlWorld load_state_dict missing %d keys (first 3: %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning("CrtlWorld load_state_dict unexpected %d keys (first 3: %s)", len(unexpected), unexpected[:3])

        # ---- expose the same attribute layout VidWMWorldModel.rollout reads ----
        # CrtlWorld wraps the pipeline (with UNet/VAE/image_encoder swapped in)
        # plus the action encoder and CLIP text encoder + tokenizer.
        # CrtlWorld.__init__ builds a base StableVideoDiffusionPipeline, which
        # doesn't accept the action/history kwargs our rollout passes. Rebuild
        # as CtrlWorldDiffusionPipeline (a subclass with the matching __call__)
        # reusing the same components.
        from ctrl_world.pipeline_ctrl_world import CtrlWorldDiffusionPipeline  # noqa: E402

        base_pipe = model.pipeline
        self.pipeline = CtrlWorldDiffusionPipeline(
            vae=base_pipe.vae,
            image_encoder=base_pipe.image_encoder,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            feature_extractor=base_pipe.feature_extractor,
        )
        self.action_encoder = model.action_encoder
        self.text_encoder = model.text_encoder
        self.tokenizer = model.tokenizer
        # CLIP for SVD is the ViT projection model used by CtrlWorld training.
        self.text_encoder_is_vit = True

        # ---- free raw state dict memory ----
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # ---- move to device and set precision ----
        self.pipeline.vae.to(self._dtype)
        self.pipeline.image_encoder.to(self._dtype)
        self.pipeline.unet.to(self._dtype)
        self.action_encoder.to(self._dtype)

        self._move_to_device()

        # ---- freeze everything for inference ----
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.image_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.action_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.pipeline.unet.eval()
        self.action_encoder.eval()

        logger.info("Ctrl-World world model loaded successfully")

    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single Ctrl-World rollout."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        cfg = self.config
        device = self._device

        # ---- unpack state (growing history buffer + sparse selection) ----
        current_latent, history_latents, history_buffer = self._unpack_state(state, observation)
        B = current_latent.shape[0]

        # ---- prepare actions (sparse history sampling + DROID normalization) ----
        state_buffer = self._get_or_init_state_buffer(state, action_chunk)
        actions, last_future_state = self._prepare_actions(action_chunk, B, state_buffer=state_buffer)
        self._debug_log_rollout_inputs(
            state=state,
            observation=observation,
            action_chunk=action_chunk,
            prepared_actions=actions,
            instruction=instruction,
        )

        # ---- encode actions through Ctrl-World's Action_encoder2 ----
        # Note: Action_encoder2.forward returns the action tensor directly
        # (no dict wrapper) and takes `text_tokinizer` (sic) instead of
        # `text_tokenizer`.
        with torch.no_grad():
            if cfg.text_cond and instruction is not None:
                action_combined = self.action_encoder(
                    actions,
                    texts=[instruction],
                    text_tokinizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    frame_level_cond=cfg.frame_level_cond,
                )
            else:
                action_combined = self.action_encoder(
                    actions,
                    frame_level_cond=cfg.frame_level_cond,
                )
        action_combined = action_combined.to(self._dtype)

        # ---- call the diffusion pipeline ----
        with torch.no_grad():
            frame_out, pred_latents = self.pipeline(
                image=current_latent,
                text=action_combined,
                width=cfg.width,
                height=cfg.height * len(cfg.view_order),  # views stacked vertically
                num_frames=cfg.num_frames,
                history=history_latents,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                min_guidance_scale=cfg.min_guidance_scale,
                max_guidance_scale=cfg.guidance_scale,
                noise_aug_strength=cfg.noise_aug_strength,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                output_type="latent",
                return_dict=False,
                frame_level_cond=cfg.frame_level_cond,
                his_cond_zero=cfg.his_cond_zero,
            )

        # ---- build next state ----
        new_current = pred_latents[:, -1]
        state_buffer.append(last_future_state)
        history_buffer.append(new_current.detach())
        next_state = {
            "current_latent": new_current,
            "_history_buffer": history_buffer,
            "_state_buffer": state_buffer,
        }

        # ---- decode frames if requested ----
        if cfg.decode_to_rgb:
            frames = self._decode_latents(pred_latents)
        else:
            frames = [pred_latents[:, i] for i in range(pred_latents.shape[1])]

        return {
            "frames": frames,
            "next_state": next_state,
            "latents": pred_latents,
        }

    # ------------------------------------------------------------------
    # Override debug logger so it doesn't reference flow-matching fields.
    # ------------------------------------------------------------------

    def _debug_log_rollout_inputs(
        self,
        *,
        state: Any,
        observation: Any,
        action_chunk: Any,
        prepared_actions: torch.Tensor,
        instruction: Optional[str],
    ) -> None:
        if not self.config.debug or self._debug_logs_emitted >= self.config.debug_log_limit:
            return

        def _shape(value: Any) -> Any:
            try:
                return tuple(np.asarray(value).shape)
            except Exception:
                return getattr(value, "shape", type(value).__name__)

        action_np = prepared_actions.detach().float().cpu().numpy()
        logger.info(
            "CtrlWorld debug[%d]: num_inference_steps=%d num_frames=%d num_history=%d "
            "instruction=%r raw_action_shape=%s prepared_action_shape=%s "
            "prepared_action_min=%.4f prepared_action_max=%.4f",
            self._debug_logs_emitted,
            self.config.num_inference_steps,
            self.config.num_frames,
            self.config.num_history,
            instruction,
            _shape(action_chunk),
            tuple(prepared_actions.shape),
            float(action_np.min()),
            float(action_np.max()),
        )
        self._debug_logs_emitted += 1
