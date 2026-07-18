"""Closed-loop ``WorldModel`` adapter for the autoregressive Wan student.

Bridges :class:`openworld.autoregressive.model.ARWorldModel` (whose native
``rollout(history_latents, cond, ...)`` works in Wan-VAE latent space over
ground-truth action sequences) to the policy-eval contract expected by
:class:`openworld.envs.WorldModelEnv`::

    rollout(state, observation, action_chunk, instruction) -> {frames, next_state}

It drives the model through the *validated live-inference* path
(:class:`openworld.autoregressive.infer.interactive.InteractiveRoller`): a
persistent warm KV-cache primed once from the initial observation, then one
generated block per env step, a sliding KV window (``max_kv_blocks``) so long
rollouts stay in RoPE-distribution, and windowed VAE decoding for seam-free
frames. (Re-implementing the rollout in the adapter — per-block decode in
isolation + an ever-growing re-primed history — broke the first frames and
drifted; using InteractiveRoller fixes both.)

Action source (per the eval design): one action per block = the env's current
**absolute** robot state (7-D ``cartesian_position_with_gripper``), read live
from ``state["robot"]["state"]`` (the post-chunk pose; falls back to
``state["_robot_state_history"]``). This is exactly the ``states`` the AR model
was trained on; we normalize with the model's own train-set percentiles
(``stats.json``), matching the open-loop replay convention.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from openworld.world_models.base_world_model import WorldModel

logger = logging.getLogger(__name__)


class ARWanWorldModel(WorldModel):
    def __init__(
        self,
        *,
        config_path: str,
        stats_root: str,
        vae_dir: str = "external/Wan2.1-T2V-1.3B-Diffusers",
        num_inference_steps: int = 32,
        num_cams: int = 3,
        width: int = 320,
        height: int = 192,
        view_order: tuple = ("exterior_right", "exterior_left", "wrist"),
        device: str = "cuda",
        debug: bool = False,
        debug_log_limit: int = 3,
        max_context_blocks: Optional[int] = None,
        bf16: bool = False,
        decode_context: Optional[int] = None,
        **_ignored: Any,
    ) -> None:
        self.config_path = config_path
        self.stats_root = stats_root
        self.vae_dir = vae_dir
        self.num_inference_steps = int(num_inference_steps)
        self.num_cams = int(num_cams)
        # Sliding-window KV cap (in blocks) for the persistent-cache roller. RoPE is
        # relative in attention, so bounding the window to the trained span keeps
        # long rollouts in-distribution. None -> set from cfg in load_checkpoint
        # (num_history_blocks + rollout_blocks); 0 -> unbounded.
        self.max_context_blocks = max_context_blocks
        # Deploy in pure bf16 (params + VAE) instead of the fp32-master + bf16-autocast
        # *training* dtype. Inference-only; ~1.3x end-to-end with no quality change.
        self.bf16 = bool(bf16)
        # Latent frames of left-context the rolling VAE decode keeps for seam-free
        # stitching. None -> num_history_blocks*frames_per_block (the previous fixed
        # default). Smaller -> less redundant decode per step (the dominant cost), at
        # some risk to block seams; validate on a real rollout before lowering.
        self._decode_context = decode_context
        self._roller = None
        self._max_kv_blocks = None
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.debug_log_limit = debug_log_limit
        self._rollout_count = 0

        # Exposed for Evaluator.run_episode (it reads world_model.config.view_order
        # to stack the initial observation the same way as predicted frames).
        self.config = SimpleNamespace(
            view_order=tuple(view_order), width=width, height=height,
        )

        self.model = None
        self.enc = None
        self.dec = None
        self.sched = None
        self.cfg = None
        self._p01 = None
        self._p99 = None
        self._dtype = None

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> None:
        # The AR training configs import as ``configs.training.*`` (a top-level
        # repo package), so the repo root must be on sys.path — it isn't when
        # this runs from the generate_videos.py subprocess (sys.path[0] is
        # scripts/). Add it explicitly, derived from this file's location.
        import os
        import sys

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from diffusers import AutoencoderKLWan

        from openworld.autoregressive.data.decode import VaeLatentDecoder
        from openworld.autoregressive.data.encode import VaeLatentEncoder
        from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
        from openworld.autoregressive.infer import load_action_stats
        from openworld.autoregressive.model import ARWorldModel
        from openworld.autoregressive.train_self_forcing import _load_config

        cfg = _load_config(self.config_path)
        self.cfg = cfg

        # Deploy dtype. bf16: pure-bf16 params, no autocast (params already carry the
        # compute dtype). Default: the training dtype (fp32 master params + bf16
        # autocast) -- ``self._dtype`` is the roller's autocast dtype either way.
        param_dtype = torch.bfloat16 if self.bf16 else cfg.dtype
        self._dtype = None if self.bf16 else (cfg.autocast_dtype or cfg.dtype)

        # Checkpoints are fp32 master weights; load into fp32, then cast the assembled
        # model to the deploy dtype (cast-after-load avoids any fp32->bf16->fp32 churn
        # in load_state_dict and keeps the checkpoint values exact pre-cast).
        model = ARWorldModel(cfg).to(self._device, cfg.dtype).eval()
        sd = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if param_dtype != cfg.dtype:
            model = model.to(param_dtype)
        logger.info(
            "ARWanWorldModel: loaded %s (missing=%d unexpected=%d) param_dtype=%s",
            checkpoint_path, len(missing), len(unexpected), param_dtype,
        )
        self.model = model

        # Encoder VAE stays fp32 (used once at bootstrap; precision is cheap there).
        # The decoder VAE -- the per-step hot path -- runs at the deploy dtype, so a
        # bf16 deploy decodes in bf16 (~1.5x). Separate instances so casting the
        # decoder doesn't perturb the encoder.
        vae_enc = AutoencoderKLWan.from_pretrained(
            self.vae_dir, subfolder="vae", torch_dtype=torch.float32
        )
        self.enc = VaeLatentEncoder(vae_enc, device=self._device, dtype=torch.float32)
        dec_dtype = torch.bfloat16 if self.bf16 else torch.float32
        vae_dec = (
            AutoencoderKLWan.from_pretrained(self.vae_dir, subfolder="vae", torch_dtype=dec_dtype)
            if self.bf16 else vae_enc
        )
        self.dec = VaeLatentDecoder(vae_dec, device=self._device, dtype=dec_dtype)

        self._p01, self._p99 = load_action_stats(self.stats_root)

        # "Slow" student: a uniform many-step flow-matching schedule (the
        # student-init checkpoint is pre-distillation, so it needs many steps).
        ts = cfg.num_train_timestep
        n = self.num_inference_steps
        steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))
        self.sched = FlowMatchScheduler(steps, num_train_timestep=ts, warp=False)

        # Sliding-window KV cap for the persistent-cache roller: bound to the
        # trained clip span so the (relative) RoPE offsets stay in-distribution
        # over long rollouts. None -> num_history_blocks + rollout_blocks; 0 ->
        # unbounded (cfg.max_kv_blocks).
        if self.max_context_blocks is None:
            self._max_kv_blocks = int(
                getattr(cfg, "num_history_blocks", 2) + getattr(cfg, "rollout_blocks", 8)
            )
        elif int(self.max_context_blocks) <= 0:
            self._max_kv_blocks = cfg.max_kv_blocks
        else:
            self._max_kv_blocks = int(self.max_context_blocks)
        logger.info(
            "ARWanWorldModel ready: steps=%d fpb=%d num_cams=%d in_ch=%d view_order=%s "
            "max_kv_blocks=%s (persistent-cache InteractiveRoller)",
            n, cfg.frames_per_block, self.num_cams, cfg.in_channels, self.config.view_order,
            self._max_kv_blocks,
        )

    # ------------------------------------------------------------------
    def _load_rgb(self, src: Any) -> np.ndarray:
        """Load one view as an (H, W, 3) uint8 array, resized to (height, width)."""
        from PIL import Image

        if isinstance(src, np.ndarray):
            img = Image.fromarray(src.astype(np.uint8)[..., :3])
        else:
            img = Image.open(str(src)).convert("RGB")
        if img.size != (self.config.width, self.config.height):
            img = img.resize((self.config.width, self.config.height), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _bootstrap_history(self, observation: Any) -> torch.Tensor:
        """Encode the initial 3-view observation into height-stacked latent history.

        Returns ``[hist_lat, C, V*h, w]`` (fp32, on device).
        """
        cfg = self.cfg
        hist_lat = cfg.num_history_blocks * cfg.frames_per_block
        n_rgb = (hist_lat - 1) * self.enc.temporal_factor + 1

        if isinstance(observation, dict):
            views = observation.get("views", observation)
            frames = [self._load_rgb(views[v]) for v in self.config.view_order]
        else:  # single stacked image: split along height into num_cams bands
            full = self._load_rgb(observation)
            h = full.shape[0] // self.num_cams
            frames = [full[i * h:(i + 1) * h] for i in range(self.num_cams)]

        per_cam = []
        for rgb in frames:                                   # each (H, W, 3)
            clip = np.repeat(rgb[None], n_rgb, axis=0)        # [n_rgb, H, W, 3]
            lat = self.enc.encode_video(clip).float()         # [C, hist_lat, h, w]
            per_cam.append(lat)
        # height-stack cameras: list of [C, Lf, h, w] -> [Lf, C, V*h, w]
        stacked = torch.cat(per_cam, dim=2)                   # [C, Lf, V*h, w]
        return stacked.permute(1, 0, 2, 3).contiguous().to(self._device)

    # ------------------------------------------------------------------
    def _current_cartesian(self, state: Any) -> np.ndarray:
        """Current absolute 7-D cartesian robot state (one action per block).

        Prefer the live post-chunk state (``robot.state``); fall back to the
        (chunk-start) history tail, then the initial state.
        """
        A = len(self._p01)
        robot = state.get("robot") if isinstance(state, dict) else None
        if isinstance(robot, dict) and robot.get("state") is not None:
            v = np.asarray(robot["state"], dtype=np.float32).reshape(-1)
            if v.shape[0] == A:
                return v
        rsh = state.get("_robot_state_history") if isinstance(state, dict) else None
        if rsh is not None and len(rsh):
            return np.asarray(rsh, dtype=np.float32)[-1].astype(np.float32)
        init = state.get("_initial_robot_state") if isinstance(state, dict) else None
        return (np.asarray(init, dtype=np.float32).reshape(-1)
                if init is not None else np.zeros(A, np.float32))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        state: Any,
        observation: Any,
        action_chunk: Any,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")
        from openworld.autoregressive.infer import normalize_actions
        from openworld.autoregressive.infer.interactive import InteractiveRoller

        cfg = self.cfg
        fpb = cfg.frames_per_block

        # One action per block: the current absolute cartesian state, normalized
        # exactly as in training (cross_attn_aligned uses one action per block).
        cur = self._current_cartesian(state)
        cur_norm = normalize_actions(cur[None], self._p01, self._p99)[0].astype(np.float32)

        started = isinstance(state, dict) and state.get("_ar_started")
        if not started or self._roller is None:
            # New episode: prime a fresh persistent-cache roller (warm KV cache +
            # sliding window + windowed decode), mirroring the validated live demo
            # path (InteractiveRoller). We only have a single initial image, so the
            # history blocks are a short clip of it; the sliding window flushes that
            # transient out as real generated frames accumulate -- it does NOT stay
            # re-primed forever the way the old growing-history adapter did.
            hist_lat = self._bootstrap_history(observation)          # [hist_frames, C, V*h, w]
            init = state.get("_initial_robot_state") if isinstance(state, dict) else None
            init = (np.asarray(init, dtype=np.float32).reshape(-1) if init is not None else cur)
            seed = normalize_actions(
                np.repeat(init[None], hist_lat.shape[0], axis=0), self._p01, self._p99
            ).astype(np.float32)
            decode_context = (
                self._decode_context if self._decode_context is not None
                else cfg.num_history_blocks * fpb
            )
            self._roller = InteractiveRoller(
                self.model, self.dec, num_cams=self.num_cams, scheduler=self.sched,
                device=self._device, autocast_dtype=self._dtype,
                max_kv_blocks=self._max_kv_blocks,
                decode_context=decode_context,
            )
            self._roller.reset(hist_lat, seed)                       # prime (returns bootstrap RGB)
            self._rollout_count = 0

        # Generate + decode exactly one block, conditioned on the current state.
        rgb = self._roller.step(cur_norm)                            # [fpb*4, V*H, W, 3] uint8
        frames = [rgb[t] for t in range(rgb.shape[0])]

        next_state = dict(state) if isinstance(state, dict) else {}
        next_state["_ar_started"] = True

        if self.debug and self._rollout_count < self.debug_log_limit:
            logger.info(
                "AR rollout[%d]: start_frame=%d max_kv_blocks=%s action_n[%.3f,%.3f] out=%s",
                self._rollout_count, self._roller.start, self._max_kv_blocks,
                float(cur_norm.min()), float(cur_norm.max()), rgb.shape,
            )
        self._rollout_count += 1
        return {"frames": frames, "next_state": next_state, "latents": None}
