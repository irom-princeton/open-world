"""VidWM (Video World Model) adapter for OpenWorld.

This wraps the shortcut flow-matching video model and exposes it through the
:class:`WorldModel` interface.

The model operates entirely in VAE latent space:
  - State: dict with keys ``current_latent`` (B,4,72,40),
    ``history_latents`` (B,num_history,4,72,40), and
    ``history_action_embeds`` (B,num_history,1024).
  - Observation: the current image latent (B,4,72,40).
  - Action chunk: raw actions (chunk_size, action_dim) -- encoded internally.
  - Output frames: decoded RGB numpy arrays (H,W,3) uint8.

Requires the ``vidwm`` package to be importable (add its repo to sys.path or
install it).  All heavy imports are deferred to avoid import errors when the
package is not available.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch

from openworld.world_models.base_world_model import WorldModel
from openworld.world_models.vidwm_loader import ensure_vidwm_repo_on_path

logger = logging.getLogger(__name__)


@dataclass
class VidWMConfig:
    """Configuration for the VidWM world model."""

    repo_path: Optional[str] = None
    debug: bool = False
    debug_log_limit: int = 3

    # Pretrained model paths
    svd_model_path: Optional[str] = None
    clip_model_path: Optional[str] = None
    unet_model_path: Optional[str] = None

    # Model architecture
    num_frames: int = 5
    num_history: int = 6
    action_dim: int = 7
    width: int = 320
    height: int = 192
    frame_level_cond: bool = True
    text_cond: bool = True
    his_cond_zero: bool = False
    dtype: str = "bfloat16"

    # Inference parameters
    motion_bucket_id: int = 127
    fps: int = 7
    guidance_scale: float = 2.0
    num_inference_steps: int = 4
    decode_chunk_size: int = 7

    # Flow-matching / shortcut parameters
    flow_map_type: str = "flow_map"
    flow_map_loss_type: str = "psd"
    distance_conditioning: bool = True
    SIGMA_MAX: float = 700.0
    SIGMA_MIN: float = 0.02

    # Action encoder
    action_encoder_type: str = "unaligned"
    action_encoder_hidden_dim: int = 1024

    # Device
    device: Optional[str] = None

    # Whether to decode latents to RGB (set False to stay in latent space)
    decode_to_rgb: bool = True
    view_order: tuple[str, ...] = ("exterior_right", "exterior_left", "wrist")
    action_downsample: int = 1

    # History indices – selects which entries from the growing per-rollout
    # history buffer are fed to the model as context.  len == num_history.
    # Negative indices count from the end of the buffer (most recent).
    # Default uses dense recent frames (last 6 rollouts), which works well
    # for typical eval horizons (10-20 rollouts).  For very long rollouts
    # (30+), consider sparser indices like (0, 0, -12, -9, -6, -3) to
    # capture longer temporal context.
    history_idx: tuple[int, ...] = (0, 0, -12, -9, -6, -3)

    # Ground-truth history mode – when True, the model expects
    # ``gt_latents`` (B, T, 4, H, W) and ``gt_actions`` (B, T, action_dim)
    # in the state dict.  Both history latents and action embeddings are
    # built from these GT tensors instead of from model predictions,
    # allowing controlled quality comparisons.
    # ``gt_actions`` should already be normalized to [-1, 1] (matching the
    # svd_ac_video_model dataset convention).
    use_gt_history: bool = False

    # Action normalization – DROID dataset percentile statistics used during
    # Ctrl-World training.  Actions are normalized to [-1, 1] via:
    #   normed = 2*(action - p01) / (p99 - p01 + eps) - 1
    action_normalize: bool = True
    action_stat_path: Optional[str] = None
    action_state_p01: tuple[float, ...] = (
        0.2676655471324921, -0.441715806722641, -0.042784303426742554,
        -3.1373724937438965, -1.214390754699707, -2.11581574678421, 0.0,
    )
    action_state_p99: tuple[float, ...] = (
        0.781050771474838, 0.4366717040538788, 0.7843997478485107,
        3.137465238571167, 0.9039141565561147, 1.9915846586227417,
        0.9911894202232361,
    )


class VidWMWorldModel(WorldModel):
    """World model backed by the VidWM shortcut flow-matching video model.

    The model is loaded lazily on the first call to :meth:`load_checkpoint`
    (or on construction if ``checkpoint_path`` is passed to the config).
    """

    def __init__(self, config: Optional[Union[VidWMConfig, Dict[str, Any]]] = None, **kwargs: Any):
        if config is None:
            config = VidWMConfig(**kwargs)
        elif isinstance(config, dict):
            config = VidWMConfig(**config)

        # Load normalization stats from JSON file if provided
        if config.action_stat_path is not None:
            import json
            with open(config.action_stat_path) as f:
                stats = json.load(f)
            config.action_state_p01 = tuple(stats["state_01"])
            config.action_state_p99 = tuple(stats["state_99"])

        self.config = config

        # These are populated by load_checkpoint
        self.pipeline = None
        self.action_encoder = None
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_is_vit: bool = True
        self._device: torch.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = torch.bfloat16 if "bfloat16" in config.dtype else torch.float16
        self._debug_logs_emitted = 0

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load VidWM pipeline, UNet weights, action encoder, and CLIP.

        ``checkpoint_path`` should point to the combined UNet+action-encoder
        ``.pt`` file (e.g. ``checkpoint-120000.pt``).  The SVD base model and
        CLIP paths are taken from ``self.config``.
        """
        cfg = self.config

        ensure_vidwm_repo_on_path(cfg.repo_path)

        # ---- lazy imports from the vidwm package ----
        from vidwm.video_models.vidwm_diffusion import VidWMDiffusionPipeline
        from vidwm.video_models.utils.svd_unet_utils import UNetSpatioTemporalConditionModel
        from vidwm.video_models.utils.svd_model_utils import load_clip

        # ---- load pretrained SVD pipeline ----
        svd_path = cfg.svd_model_path
        if svd_path is None:
            from huggingface_hub import snapshot_download
            svd_path = snapshot_download(repo_id="stabilityai/stable-video-diffusion-img2vid")

        logger.info("Loading SVD pipeline from %s", svd_path)
        self.pipeline = VidWMDiffusionPipeline.from_pretrained(svd_path)
        self.pipeline.set_progress_bar_config(disable=True)

        # ---- replace UNet with distance-conditioned variant ----
        unet = UNetSpatioTemporalConditionModel(
            distance_conditioning=cfg.distance_conditioning,
        )

        # Load fine-tuned UNet weights
        logger.info("Loading UNet weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        unet_state_dict = {
            k[len("unet."):]: v
            for k, v in state_dict.items()
            if k.startswith("unet.")
        }
        unet.load_state_dict(unet_state_dict, strict=False)

        self.pipeline.unet = unet

        # ---- load CLIP text encoder ----
        clip_path = cfg.clip_model_path
        if clip_path is None:
            from huggingface_hub import snapshot_download
            clip_path = snapshot_download(repo_id="openai/clip-vit-base-patch32")

        self.text_encoder_is_vit = "RN50" not in str(clip_path)
        logger.info("Loading CLIP from %s (is_vit=%s)", clip_path, self.text_encoder_is_vit)
        self.text_encoder, self.tokenizer = load_clip(clip_model_path=clip_path)

        # ---- load action encoder ----
        action_num = cfg.num_history + cfg.num_frames
        if cfg.action_encoder_type == "unaligned":
            from vidwm.action_encoders.unaligned_action_encoder import ActionEncoderUnaligned

            self.action_encoder = ActionEncoderUnaligned(
                action_dim=cfg.action_dim,
                action_num=action_num,
                hidden_dim=cfg.action_encoder_hidden_dim,
                text_cond=cfg.text_cond,
            )
        elif "clip_aligned" in cfg.action_encoder_type:
            from vidwm.action_encoders.clip_aligned_action_encoder import ActionEncoderCLIPAligned

            self.action_encoder = ActionEncoderCLIPAligned(
                action_dim=cfg.action_dim,
                action_num=action_num,
                hidden_dim=cfg.action_encoder_hidden_dim,
                text_cond=cfg.text_cond,
            )
        else:
            raise ValueError(f"Unknown action encoder type: {cfg.action_encoder_type}")

        # Load action encoder weights from same checkpoint
        ae_state_dict = {
            k[len("action_encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("action_encoder.")
        }
        if ae_state_dict:
            logger.info("Loading action encoder weights (%d keys)", len(ae_state_dict))
            self.action_encoder.load_state_dict(ae_state_dict, strict=False)

        # ---- free raw state dict memory ----
        state_dict.clear()
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

        logger.info("VidWM world model loaded successfully")

    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single world-model rollout.

        Args:
            state: Dict with keys:
                ``current_latent``: (B, 4, H, W) or (4, H, W) current image latent.
                ``history_latents``: (B, num_history, 4, H, W) history latents.
                  If not enough history is available, the current frame is
                  broadcast to fill the gap.
            observation: Ignored (redundant with state["current_latent"]).
            action_chunk: (chunk_size, action_dim) or (B, chunk_size, action_dim)
                raw actions.  The chunk is split internally into history-action
                and future-action portions for the model.
            instruction: Optional language instruction string.

        Returns:
            Dict with:
                ``frames``: list of (H, W, 3) uint8 numpy arrays (if decode_to_rgb)
                    or list of (4, latH, latW) latent tensors.
                ``next_state``: updated state dict for the next rollout call.
                ``latents``: (num_frames, 4, H, W) predicted latent tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        cfg = self.config
        device = self._device
        gt_mode = (
            cfg.use_gt_history
            and isinstance(state, dict)
            and state.get("gt_latents") is not None
        )

        # ---- unpack state (growing history buffer + sparse selection) ----
        current_latent, history_latents, history_buffer = self._unpack_state(state, observation)
        B = current_latent.shape[0]

        if gt_mode and state.get("gt_actions") is not None:
            # ---- GT action path: matches svd_ac_video_model exactly ----
            action_combined = self._prepare_action_combined_gt(state, instruction)
            # state_buffer / last_future_state not needed in GT mode
            state_buffer = state.get("_state_buffer", [])
            last_future_state = np.zeros(cfg.action_dim, dtype=np.float32)
        else:
            # ---- normal action path ----
            # ---- get / init per-rollout state buffer ----
            state_buffer = self._get_or_init_state_buffer(state, action_chunk)

            # ---- prepare actions (sparse history sampling via history_idx) ----
            actions, last_future_state = self._prepare_actions(action_chunk, B, state_buffer=state_buffer)
            self._debug_log_rollout_inputs(
                state=state,
                observation=observation,
                action_chunk=action_chunk,
                prepared_actions=actions,
                instruction=instruction,
            )

            # ---- encode actions ----
            with torch.no_grad():
                action_embeds = self.action_encoder(
                    actions,
                    texts=[instruction] if instruction else None,
                    text_tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    frame_level_cond=cfg.frame_level_cond,
                    text_encoder_is_vit=self.text_encoder_is_vit,
                    device=device,
                )
                action_latent = action_embeds["action_with_text_embeds"]  # (B, T, 1024)

            # ---- split action embeddings into history + future ----
            his_action = action_latent[:, :cfg.num_history]
            future_action = action_latent[:, cfg.num_history:]

            # ---- build combined action embedding ----
            action_combined = torch.cat([his_action, future_action], dim=1)
            action_combined = action_combined.to(self._dtype)

        # ---- call the diffusion pipeline ----
        with torch.no_grad():
            frame_out, pred_latents = self.pipeline(
                image=current_latent,
                text=action_combined,
                width=cfg.width,
                height=cfg.height * 3,  # 3 camera views stacked
                num_frames=cfg.num_frames,
                history=history_latents,
                num_inference_steps=cfg.num_inference_steps,
                decode_chunk_size=cfg.decode_chunk_size,
                max_guidance_scale=cfg.guidance_scale,
                fps=cfg.fps,
                motion_bucket_id=cfg.motion_bucket_id,
                output_type="latent",
                return_dict=False,
                frame_level_cond=cfg.frame_level_cond,
                his_cond_zero=cfg.his_cond_zero,
                flow_map_type=cfg.flow_map_type,
                flow_map_loss_type=cfg.flow_map_loss_type,
            )

        # ---- build next state ----
        new_current = pred_latents[:, -1]  # (B, 4, 72, 40)
        state_buffer.append(last_future_state)

        if gt_mode:
            # GT-history mode: advance cursor by (num_frames - 1) to match
            # the reference convention (last predicted frame = next current).
            gt_cursor = state.get("_gt_cursor", 0)
            gt_lat = state["gt_latents"]
            T_dim = gt_lat.shape[1] if gt_lat.dim() == 5 else gt_lat.shape[0]
            next_gt_cursor = min(gt_cursor + cfg.num_frames - 1, T_dim - 1)
            next_state = {
                "gt_latents": gt_lat,
                "_gt_cursor": next_gt_cursor,
                "_state_buffer": state_buffer,
            }
            # Carry GT actions and cached embeddings forward
            if state.get("gt_actions") is not None:
                next_state["gt_actions"] = state["gt_actions"]
            if state.get("_gt_action_embeds") is not None:
                next_state["_gt_action_embeds"] = state["_gt_action_embeds"]
        else:
            # Normal mode: append last predicted frame to growing history buffer.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_to_device(self) -> None:
        """Move all model components to the configured device."""
        self.pipeline.unet.to(self._device)
        self.pipeline.vae.to(self._device)
        self.pipeline.image_encoder.to(self._device)
        self.action_encoder.to(self._device)
        self.text_encoder.to(self._device)

    def _unpack_state(self, state: Any, observation: Any = None) -> tuple:
        """Unpack state, returning (current_latent, history_latents, history_buffer).

        The history buffer is a growing list of per-rollout latent frames.
        ``history_latents`` is built by sparse-sampling the buffer using
        ``config.history_idx``, matching the Ctrl-World training convention.

        When ``config.use_gt_history`` is True and ``state["gt_latents"]`` is
        available, the history (and current latent) are drawn from the GT
        trajectory instead of from model predictions.  A ``_gt_cursor`` in the
        state tracks how far along the trajectory we are.
        """
        cfg = self.config
        device = self._device

        # ---- GT-history mode ----
        if cfg.use_gt_history and isinstance(state, dict) and state.get("gt_latents") is not None:
            return self._unpack_state_gt(state)

        # ---- normal (autoregressive) mode ----
        history_buffer = None
        if isinstance(state, dict):
            current_latent = state.get("current_latent")
            history_buffer = state.get("_history_buffer")
        else:
            current_latent = None

        if current_latent is None:
            current_latent = self._bootstrap_current_latent(observation)
            history_buffer = None

        # Ensure batch dimension
        if current_latent.dim() == 3:
            current_latent = current_latent.unsqueeze(0)
        current_latent = current_latent.to(device=device, dtype=self._dtype)

        # Initialize buffer with copies of current frame if needed.
        # Size = num_history * 4, matching the Ctrl-World reference
        # (rollout_interact_pi.py:338).  This ensures that negative
        # history_idx entries (e.g. -3, -6, …) continue to point at
        # initial-frame copies for many rollouts, giving the model a
        # stable warm-up period before predicted frames enter the
        # history.  A smaller init_size (e.g. max(|neg_indices|))
        # causes predicted frames to appear in history too early,
        # leading to compounding error and rollout divergence.
        if history_buffer is None:
            init_size = cfg.num_history * 4
            history_buffer = [current_latent.clone() for _ in range(init_size)]

        # Build sparse history_latents using history_idx
        history_latents = torch.stack(
            [history_buffer[idx] for idx in cfg.history_idx], dim=1,
        )  # (B, num_history, 4, H, W)
        history_latents = history_latents.to(device=device, dtype=self._dtype)

        return current_latent, history_latents, history_buffer

    def _unpack_state_gt(self, state: dict) -> tuple:
        """Build current_latent and history from ground-truth latents.

        ``state["gt_latents"]`` should be a ``(B, T, 4, H, W)`` tensor
        covering the full trajectory.  ``state["_gt_cursor"]`` tracks
        the current position; it is advanced by ``num_frames - 1`` each
        rollout (matching the svd_ac_video_model convention of reusing
        the last predicted frame as the next current frame).

        History is built by padding with copies of frame 0 and then
        appending all GT frames up to the cursor, then selecting the
        last ``num_history`` entries — mirroring the dense sequential
        history used in the reference VideoPredictor.
        """
        cfg = self.config
        device = self._device

        gt_latents = state["gt_latents"]  # (B, T, 4, H, W)
        if gt_latents.dim() == 4:
            gt_latents = gt_latents.unsqueeze(0)
        gt_latents = gt_latents.to(device=device, dtype=self._dtype)

        cursor = state.get("_gt_cursor", 0)

        # Current frame is the GT frame at cursor position
        current_latent = gt_latents[:, cursor]  # (B, 4, H, W)

        # Build a dense history from all GT frames up to (not including) cursor,
        # padded at the front with copies of frame 0.
        gt_frames_so_far = [gt_latents[:, i] for i in range(cursor + 1)]
        pad_size = max(cfg.num_history - len(gt_frames_so_far), 0)
        padded = [gt_latents[:, 0].clone() for _ in range(pad_size)] + gt_frames_so_far
        # Select the last num_history entries (dense sequential, like his_skip=1)
        history_latents = torch.stack(padded[-cfg.num_history:], dim=1)

        # Return an empty list as history_buffer — it won't be used in GT mode
        # since we rebuild from gt_latents each time.
        return current_latent, history_latents, []

    def _prepare_action_combined_gt(
        self,
        state: dict,
        instruction: Optional[str],
    ) -> torch.Tensor:
        """Build the (B, num_history + num_frames, 1024) action embedding from
        GT actions, matching the svd_ac_video_model VideoPredictor logic.

        The reference implementation:
        1. Encodes ALL trajectory actions at once through the action encoder.
        2. Splits into history (first ``num_history``) and future portions.
        3. Pads history action embeddings with copies of the first future
           action embedding.
        4. Per rollout, selects history embeddings via ``his_id`` and future
           embeddings via ``action_idx`` based on the rollout index.

        ``state["gt_actions"]`` must be a **(B, T, action_dim)** tensor of
        already-normalised actions ([-1, 1]).  The encoded embeddings are
        cached in ``state["_gt_action_embeds"]`` so encoding only happens once.
        """
        cfg = self.config
        device = self._device

        gt_actions = state["gt_actions"]  # (B, T, action_dim), already normalised
        if not isinstance(gt_actions, torch.Tensor):
            gt_actions = torch.as_tensor(gt_actions, dtype=self._dtype, device=device)
        if gt_actions.dim() == 2:
            gt_actions = gt_actions.unsqueeze(0)
        gt_actions = gt_actions.to(device=device, dtype=self._dtype)

        # ---- encode once and cache ----
        if state.get("_gt_action_embeds") is None:
            with torch.no_grad():
                embeds = self.action_encoder(
                    gt_actions,
                    texts=[instruction] if instruction else None,
                    text_tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    frame_level_cond=cfg.frame_level_cond,
                    text_encoder_is_vit=self.text_encoder_is_vit,
                    device=device,
                )
            # Cache on the state dict so subsequent rollouts reuse it.
            state["_gt_action_embeds"] = embeds["action_with_text_embeds"]  # (B, T, 1024)

        all_embeds = state["_gt_action_embeds"]  # (B, T, 1024)

        # ---- split into history / future (same as svd_ac_video_model) ----
        # The first num_history embeddings correspond to history-frame timestamps;
        # the rest are future actions.
        future_embeds = all_embeds[:, cfg.num_history:]  # (B, T_future, 1024)

        # ---- determine rollout index from cursor ----
        cursor = state.get("_gt_cursor", 0)
        # rollout_idx: how many rollouts have been done (cursor advances by
        # num_frames-1 per rollout, starting from 0).
        rollout_idx = cursor // max(cfg.num_frames - 1, 1) if cursor > 0 else 0

        # ---- history action embeddings ----
        # Pad with copies of the first future embedding, then append all
        # future embeddings consumed so far.
        num_consumed = rollout_idx * (cfg.num_frames - 1)
        his_pad = future_embeds[:, 0:1].expand(-1, cfg.num_history, -1)  # (B, num_history, 1024)
        if num_consumed > 0:
            consumed = future_embeds[:, :num_consumed]  # (B, num_consumed, 1024)
            his_stack = torch.cat([his_pad, consumed], dim=1)  # (B, num_history + num_consumed, 1024)
        else:
            his_stack = his_pad

        # Select the last num_history entries (dense sequential, his_skip=1)
        his_action = his_stack[:, -cfg.num_history:]  # (B, num_history, 1024)

        # ---- future action embeddings for this rollout ----
        action_start = rollout_idx * (cfg.num_frames - 1)
        action_end = action_start + cfg.num_frames
        T_future = future_embeds.shape[1]
        action_idx = np.arange(action_start, action_end)
        action_idx = np.clip(action_idx, 0, T_future - 1)
        future_action = future_embeds[:, action_idx]  # (B, num_frames, 1024)

        return torch.cat([his_action, future_action], dim=1)  # (B, num_history + num_frames, 1024)

    def _bootstrap_current_latent(self, observation: Any) -> torch.Tensor:
        """Create an initial latent state from raw RGB observations.

        Each camera view is VAE-encoded independently and the resulting
        latents are concatenated along the spatial height dimension. 
        """
        view_frames = self._get_observation_view_frames(observation)

        view_latents = []
        with torch.no_grad():
            for frame in view_frames:
                rgb_tensor = (
                    torch.from_numpy(frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0 * 2.0 - 1.0
                )
                view_latents.append(self.encode_image(rgb_tensor))

        # Concatenate along the spatial height dim: 3 × (B, 4, 24, 40) → (B, 4, 72, 40)
        return torch.cat(view_latents, dim=2)

    def _get_observation_view_frames(self, observation: Any) -> List[np.ndarray]:
        """Return a list of per-view RGB frames from the observation.

        Each element is an (H, W, 3) uint8 numpy array for one camera view,
        ordered according to ``config.view_order``.
        """
        cfg = self.config

        if isinstance(observation, dict):
            views = observation.get("views", observation)
            if isinstance(views, dict):
                frames = []
                for view_name in cfg.view_order:
                    if view_name not in views:
                        raise ValueError(
                            f"Observation is missing required view '{view_name}'. "
                            f"Available views: {list(views)}"
                        )
                    frames.append(self._load_rgb_frame(views[view_name]))
                return frames

        # Single image – assume views are stacked vertically.
        frame = self._load_rgb_frame(observation)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                "VidWM bootstrap expects an RGB frame or a dict of named RGB views."
            )
        n_views = len(cfg.view_order)
        if frame.shape[0] % n_views != 0:
            raise ValueError(
                f"Stacked image height {frame.shape[0]} is not divisible by "
                f"{n_views} views."
            )
        split_h = frame.shape[0] // n_views
        return [frame[i * split_h : (i + 1) * split_h] for i in range(n_views)]

    def _stack_observation_views(self, observation: Any) -> np.ndarray:
        cfg = self.config

        if isinstance(observation, dict):
            views = observation.get("views", observation)
            if isinstance(views, dict):
                frames = []
                for view_name in cfg.view_order:
                    if view_name not in views:
                        raise ValueError(
                            f"Observation is missing required view '{view_name}'. "
                            f"Available views: {list(views)}"
                        )
                    frames.append(self._load_rgb_frame(views[view_name]))
                return np.concatenate(frames, axis=0)

        frame = self._load_rgb_frame(observation)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                "VidWM bootstrap expects an RGB frame or a dict of named RGB views."
            )
        return frame

    @staticmethod
    def _load_rgb_frame(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, str):
            with Image.open(value) as image:
                return np.asarray(image.convert("RGB"))
        raise ValueError(f"Unsupported observation frame value: {type(value)!r}")

    def _prepare_actions(
        self,
        action_chunk: Any,
        batch_size: int,
        state_buffer: Optional[List] = None,
    ) -> tuple:
        """Convert action chunk to tensor (B, num_history + num_frames, action_dim).

        History actions are sampled from *state_buffer* using the same sparse
        ``config.history_idx`` that selects history latent frames, keeping the
        two modalities temporally aligned.

        Returns:
            (action_tensor, last_future_state) where *last_future_state* is the
            raw (un-normalised) action/state at the end of the prediction
            horizon, suitable for appending to the state buffer.
        """
        cfg = self.config
        stride = max(int(cfg.action_downsample), 1)

        future_actions = np.asarray(action_chunk, dtype=np.float32)
        if future_actions.ndim == 1:
            future_actions = future_actions.reshape(1, -1)
        future_actions = future_actions[::stride]
        if future_actions.shape[0] > cfg.num_frames:
            future_actions = future_actions[: cfg.num_frames]

        if future_actions.shape[0] == 0:
            raise ValueError("VidWM requires at least one future action.")

        # Save the last future state before any padding/normalisation – this
        # corresponds to the state at the end of the prediction horizon
        # (equivalent to cartesian_pose[pred_step-1] in the reference).
        last_future_state = future_actions[-1].copy()

        # Sample history actions using the same sparse history_idx as latents
        if state_buffer is not None and len(state_buffer) > 0:
            history_actions = np.stack(
                [state_buffer[idx] for idx in cfg.history_idx], axis=0,
            )  # (num_history, action_dim)
        else:
            history_actions = np.repeat(
                future_actions[:1], len(cfg.history_idx), axis=0,
            )

        # Pad future if needed
        if future_actions.shape[0] < cfg.num_frames:
            future_pad = np.repeat(future_actions[-1:], cfg.num_frames - future_actions.shape[0], axis=0)
            future_actions = np.concatenate([future_actions, future_pad], axis=0)

        action_array = np.concatenate([history_actions, future_actions], axis=0)

        # Normalize actions to [-1, 1] using DROID dataset percentile stats,
        # matching the Ctrl-World training convention.
        if cfg.action_normalize:
            p01 = np.asarray(cfg.action_state_p01, dtype=np.float32)
            p99 = np.asarray(cfg.action_state_p99, dtype=np.float32)
            eps = 1e-8
            action_array = np.clip(
                2.0 * (action_array - p01) / (p99 - p01 + eps) - 1.0,
                -1.0, 1.0,
            )

        action_tensor = torch.as_tensor(action_array, dtype=self._dtype, device=self._device).unsqueeze(0)

        if action_tensor.shape[0] == 1 and batch_size > 1:
            action_tensor = action_tensor.expand(batch_size, -1, -1)

        return action_tensor, last_future_state

    def _get_or_init_state_buffer(
        self,
        state: Any,
        action_chunk: Any,
    ) -> List[np.ndarray]:
        """Return the per-rollout state buffer, initialising if needed.

        The state buffer parallels the latent history buffer: one entry per
        rollout, indexed with the same ``history_idx``.  It is initialised
        with copies of the current robot state (or the first action if no
        robot state is available).
        """
        cfg = self.config

        if isinstance(state, dict) and state.get("_state_buffer") is not None:
            return list(state["_state_buffer"])

        # Determine initial state value.  Prefer the pre-advancement initial
        # robot state passed by the env (``_initial_robot_state``), since by
        # the time rollout() is called ``robot["state"]`` has been advanced
        # through all policy actions and no longer reflects the true initial
        # pose.
        initial_state = None
        if isinstance(state, dict):
            if state.get("_initial_robot_state") is not None:
                initial_state = np.asarray(state["_initial_robot_state"], dtype=np.float32).reshape(-1)
            elif isinstance(state.get("robot"), dict) and "state" in state["robot"]:
                initial_state = np.asarray(state["robot"]["state"], dtype=np.float32).reshape(-1)
        if initial_state is None:
            raw = np.asarray(action_chunk, dtype=np.float32)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            initial_state = raw[0]

        init_size = cfg.num_history * 4  # match Ctrl-World reference
        return [initial_state.copy() for _ in range(init_size)]

    def _decode_latents(self, latents: torch.Tensor) -> List[np.ndarray]:
        """Decode latent tensor to a list of RGB uint8 numpy frames.

        Args:
            latents: (B, T, 4, 72, 40) predicted latent frames where the
                height dimension contains multiple camera views stacked
                vertically (e.g. 72 = 3 views × 24).

        Returns:
            List of (H_stacked, W, 3) uint8 numpy arrays, one per frame.
            Frames from all batches are flattened into the list.
        """
        import einops

        B, T, C, H_stacked, W = latents.shape
        n_views = len(self.config.view_order)
        H_view = H_stacked // n_views
        vae = self.pipeline.vae
        vae_dtype = vae.dtype
        cfg = self.config

        # Split views BEFORE decoding to avoid cross-view convolution bleed.
        # (B, T, C, n_views*H_view, W) → (B*n_views, T, C, H_view, W)
        latents = einops.rearrange(
            latents, "b t c (m h) w -> (b m) t c h w", m=n_views,
        )

        # Flatten batch and time for chunked VAE decode.
        x = latents.reshape(-1, C, H_view, W)  # (B*n_views*T, C, H_view, W)

        decoded_chunks = []
        for i in range(0, x.shape[0], cfg.decode_chunk_size):
            # Divide by scaling_factor in float32 BEFORE casting to vae dtype.
            chunk = x[i : i + cfg.decode_chunk_size] / vae.config.scaling_factor
            chunk = chunk.to(vae_dtype)
            decode_kwargs = {}
            if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_in"):
                decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_chunks.append(vae.decode(chunk, **decode_kwargs).sample)
        decoded = torch.cat(decoded_chunks, dim=0)  # (B*n_views*T, 3, H_px, W_px)

        # Reshape back to per-view, reassemble stacked image.
        # (B*n_views*T, 3, H_px, W_px) → (B, T, 3, n_views*H_px, W_px)
        decoded = einops.rearrange(
            decoded, "(b m t) c h w -> b t c (m h) w", b=B, m=n_views, t=T,
        )

        # Convert to uint8 numpy frames.
        decoded = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255)
        decoded = decoded.detach().float().cpu().numpy()

        all_frames = []
        for b in range(B):
            for t in range(T):
                # (3, H_stacked_px, W_px) → (H_stacked_px, W_px, 3)
                frame = decoded[b, t].transpose(1, 2, 0).astype(np.uint8)
                all_frames.append(frame)

        return all_frames

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

        obs_summary = type(observation).__name__
        if isinstance(observation, dict):
            if isinstance(observation.get("views"), dict):
                obs_summary = f"views={sorted(observation['views'].keys())}"
            else:
                obs_summary = f"obs_keys={sorted(observation.keys())}"

        state_summary = type(state).__name__
        if isinstance(state, dict):
            state_summary = f"state_keys={sorted(state.keys())}"
            if "robot" in state and isinstance(state["robot"], dict):
                state_summary += f" robot_keys={sorted(state['robot'].keys())}"
                if "state_representation" in state["robot"]:
                    state_summary += f" state_rep={state['robot']['state_representation']}"
            if "current_latent" in state:
                state_summary += f" current_latent={tuple(state['current_latent'].shape)}"
            if "_history_buffer" in state:
                state_summary += f" history_buffer_len={len(state['_history_buffer'])}"
            if "_state_buffer" in state:
                state_summary += f" state_buffer_len={len(state['_state_buffer'])}"

        action_np = prepared_actions.detach().float().cpu().numpy()
        logger.info(
            "VidWM debug[%d]: flow_map_type=%s loss_type=%s num_inference_steps=%d num_frames=%d num_history=%d instruction=%r raw_action_shape=%s prepared_action_shape=%s prepared_action_min=%.4f prepared_action_max=%.4f observation=%s state=%s",
            self._debug_logs_emitted,
            self.config.flow_map_type,
            self.config.flow_map_loss_type,
            self.config.num_inference_steps,
            self.config.num_frames,
            self.config.num_history,
            instruction,
            _shape(action_chunk),
            tuple(prepared_actions.shape),
            float(action_np.min()),
            float(action_np.max()),
            obs_summary,
            state_summary,
        )
        self._debug_logs_emitted += 1

    # ------------------------------------------------------------------
    # Convenience: encode a raw RGB image into a latent
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an RGB image tensor to VAE latent space.

        Args:
            image: (B, 3, H, W) float tensor in [0, 1] or [-1, 1].

        Returns:
            (B, 4, H//8, W//8) latent tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        image = image.to(device=self._device, dtype=self.pipeline.vae.dtype)
        # The SVD VAE expects input in [-1, 1]. If the image is in [0, 1],
        # convert it first.
        if image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 2.0 - 1.0
        latent = self.pipeline.vae.encode(image).latent_dist.mode()
        # Scale to match Ctrl-World training convention: latents are stored
        # multiplied by the VAE scaling factor.  The pipeline divides by this
        # factor when building the conditioning signal, so we must apply it
        # here to keep the round-trip correct.
        latent = latent * self.pipeline.vae.config.scaling_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a single latent frame to RGB.

        Args:
            latent: (B, 4, H, W) latent tensor.

        Returns:
            (B, 3, H, W) float32 RGB tensor.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        latent = latent.to(device=self._device, dtype=self.pipeline.vae.dtype)
        scaled = latent / self.pipeline.vae.config.scaling_factor
        decoded = self.pipeline.vae.decode(scaled).sample
        return decoded.float()
