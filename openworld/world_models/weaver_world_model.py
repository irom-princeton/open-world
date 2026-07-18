"""Closed-loop ``WorldModel`` adapter for the WEAVER world model.

WEAVER source is vendored in-repo as a git submodule at ``external/WEAVER``
(``_DEFAULT_WEAVER_REPO``); it runs in its own torch-2.7 / diffusers-0.35 venv
(``.venv-weaver``, or the legacy external WEAVER ``.venv``). This adapter is
imported only inside that venv (the weaver eval job), driven by
``scripts/run_evaluation.py`` while pi0.5 runs out-of-process via an OpenPI
websocket server. Checkpoints stay external, referenced via
``checkpoints/weaver`` (a symlink to the WEAVER checkpoint dir).

WEAVER's native ``generate_videos_full(obs, actions, instructions, horizon,
memory, bootstrap)`` generates a whole clip autoregressively from a *full* action
sequence. For closed-loop policy eval we instead call it once per env step with a
short horizon (predict ``bootstrap`` frames), feeding the freshly predicted
frames back as the next step's history — mirroring the validated open-loop replay
(``cmp/tri_weaver_run.py``) frame-by-frame.

Action convention (matches WEAVER training + the user's "absolute state from env
history" choice): WEAVER consumes 8-D **joint+gripper deltas** normalized by its
own ``norm_stats``. The env's openpi action adapter keeps ``robot.joint_position``
+ gripper correctly advanced each step, so we read that absolute joint state,
interpolate across the chunk to per-frame deltas, and normalize. Observations
carry the absolute joint+gripper state (normalized by the state stats).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from openworld.world_models.base_world_model import WorldModel

logger = logging.getLogger(__name__)


class WeaverWorldModel(WorldModel):
    # Repo root (…/open-world-autoregressive), three levels up from this file.
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Vendored WEAVER source (git submodule). Checkpoints stay external.
    _DEFAULT_WEAVER_REPO = os.path.join(_REPO_ROOT, "external", "WEAVER")

    def __init__(
        self,
        *,
        weaver_repo: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        val_steps: int = 4,            # "fast" few-step distilled sampler
        horizon: int = 8,              # trained eval_horizon (per-chunk predict length)
        bootstrap: int = 5,            # trained eval_bootstrap (frames kept per internal chunk)
        width: int = 320,
        height: int = 192,
        # weaver view name -> dataset (0617) view name
        view_map: Optional[dict] = None,
        device: str = "cuda",
        debug: bool = False,
        debug_log_limit: int = 3,
        **_ignored: Any,
    ) -> None:
        self.weaver_repo = weaver_repo or self._DEFAULT_WEAVER_REPO
        self.norm_stats_path = norm_stats_path or os.path.join(
            self._REPO_ROOT, "checkpoints/weaver/norm_stats_relabel.json"
        )
        self.val_steps = int(val_steps)
        self.horizon = int(horizon)
        self.bootstrap = int(bootstrap)
        self.width = int(width)
        self.height = int(height)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.debug_log_limit = debug_log_limit
        self._rollout_count = 0

        # weaver img_keys in stack order (top -> bottom); dataset names parallel.
        self.img_keys = ["wrist_left", "exterior_1_left"]
        self.view_map = view_map or {"wrist_left": "wrist", "exterior_1_left": "exterior_left"}
        ds_order = tuple(self.view_map[k] for k in self.img_keys)
        # Exposed for Evaluator.run_episode (stacks the initial obs in this order).
        self.config = SimpleNamespace(view_order=ds_order, width=width, height=height)

        self.model = None
        self.cfg = None
        self.n_hist = None
        self.n_mem = None
        self.t_mem = None
        self._sm = self._ss = self._am = self._asd = None

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> None:
        # checkpoint_path is the weaver checkpoint *dir* (contains config.yaml).
        if self.weaver_repo not in sys.path:
            sys.path.insert(0, self.weaver_repo)
        os.chdir(self.weaver_repo)  # weaver build uses repo-relative assets

        from weaver.generate_views import build_model, clean_state_dict, load_eval_config
        from weaver.utils.tools import load_checkpoint as wv_load_checkpoint

        ckpt_dir = checkpoint_path
        overrides = [
            f"dataset.norm_stats_path={self.norm_stats_path}",
            f"model.val_steps={self.val_steps}",
            f"eval_horizon={self.horizon}",
            f"eval_bootstrap={self.bootstrap}",
            "inference.pyramid_stagger_width=1",
            "inference.pyramid_schedule=cosine",
        ]
        cfg = load_eval_config(ckpt_dir, overrides)
        model, img_keys = build_model(cfg, str(self._device))
        ck = wv_load_checkpoint(ckpt_dir, str(self._device), weights_only=True)
        model.load_state_dict(clean_state_dict(ck["model"]), strict=False)
        if "ema" in ck:
            model.ema.to("cpu")
            model.ema.load_state_dict(ck["ema"])
            model.ema.apply_to(model)
            torch.cuda.empty_cache()
        model.eval()
        model._inference_steps = self.val_steps
        model._pyramid_schedule_cache = {}

        self.model = model
        self.cfg = cfg
        self.n_hist = cfg.n_history
        self.n_mem = cfg.n_memory_frames
        self.t_mem = cfg.t_memory

        ns = json.load(open(self.norm_stats_path))["norm_stats"]
        self._sm = np.asarray(ns["state"]["mean"], dtype=np.float32)
        self._ss = np.asarray(ns["state"]["std"], dtype=np.float32)
        self._am = np.asarray(ns["actions"]["mean"], dtype=np.float32)
        self._asd = np.asarray(ns["actions"]["std"], dtype=np.float32)
        logger.info(
            "WeaverWorldModel ready: val_steps=%d img_keys=%s n_hist=%d n_mem=%d t_mem=%d "
            "horizon=%d boot=%d",
            self.val_steps, img_keys, self.n_hist, self.n_mem, self.t_mem,
            self.horizon, self.bootstrap,
        )

    # ------------------------------------------------------------------
    def _load_rgb(self, src: Any) -> np.ndarray:
        from PIL import Image

        if isinstance(src, np.ndarray):
            img = Image.fromarray(src.astype(np.uint8)[..., :3])
        else:
            img = Image.open(str(src)).convert("RGB")
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def _read_joint_state(self, state: Any) -> np.ndarray:
        """Return absolute 8-D [7 joint + 1 gripper] from the env state."""
        robot = state.get("robot", {}) if isinstance(state, dict) else {}
        jp = robot.get("joint_position", robot.get("joint_positions"))
        gp = robot.get("gripper_position")
        if jp is None:  # fall back to the initial joints stashed at reset
            jp = state.get("_initial_joint_position") if isinstance(state, dict) else None
        jp = np.asarray(jp, dtype=np.float32).reshape(-1)[:7] if jp is not None else np.zeros(7, np.float32)
        if jp.shape[0] < 7:
            jp = np.pad(jp, (0, 7 - jp.shape[0]))
        g = float(np.asarray(gp, dtype=np.float32).reshape(-1)[0]) if gp is not None else 0.0
        return np.concatenate([jp, [g]]).astype(np.float32)  # [8]

    def _to_img(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr = np.stack(frames).astype(np.float32)            # [T,H,W,3]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).div(255.0).unsqueeze(0)
        return t.to(self._device)                            # [1,T,3,H,W]

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
        from einops import rearrange
        m = self.model
        n_hist, n_mem, horizon, boot = self.n_hist, self.n_mem, self.horizon, self.bootstrap

        # --- persistent LATENT state across env-steps (NO RGB round-trip) ---
        # We carry weaver's predicted *latents* (x1_hist) + memory_tokens forward
        # exactly like generate_videos_full's internal loop, decoding only for the
        # output video. This avoids the decode->re-encode VAE round-trip + memory
        # reset that collapsed the per-step micro-call version.
        st = state.get("_wv_lat") if isinstance(state, dict) else None
        if st is None:
            views = observation.get("views", observation) if isinstance(observation, dict) else observation
            init_rgb = {wk: self._load_rgb(views[self.view_map[wk]]) for wk in self.img_keys}
            j0 = self._read_joint_state(state)
            jn0 = ((j0 - self._sm) / self._ss).astype(np.float32)
            obs0 = {wk: self._to_img([init_rgb[wk]] * n_hist) for wk in self.img_keys}
            obs0["states"] = torch.from_numpy(np.repeat(jn0[None], n_hist, 0)).float()[None].to(self._device)
            mem0 = {wk: self._to_img([init_rgb[wk]] * n_mem) for wk in self.img_keys}
            mem0["states"] = torch.from_numpy(np.repeat(jn0[None], n_mem, 0)).float()[None].to(self._device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x1_hist = m.encode_obs(obs0)              # {wk:(1,n_hist,N,D), states:(1,n_hist,8)}
                mem_tokens = m.encode_memory_obs(mem0)    # (1, n_mem*N_total, n_embed)
            st = {
                "x1": x1_hist, "mem": mem_tokens,
                "joint_abs": j0.astype(np.float32),             # current absolute [7 joint + 1 gripper]
                "hist_act": np.zeros((n_hist, 8), np.float32),  # carried history action deltas (normalized)
            }

        x1_hist, mem_tokens = st["x1"], st["mem"]

        # --- WEAVER's joint-space PI->WM action conversion (relabel, RGB_SKIP) ---
        # The OpenWorld env is cartesian-centric (it FK-converts PI joint velocities
        # to cartesian for ctrl-world/AR), but WEAVER conditions in joint space at
        # video rate. The OpenPI adapter already runs the dynamics model (model2_15_9)
        # to predict the full CONTROL-rate absolute joint+gripper trajectory; we get
        # it via robot["_pi_joint_traj"], subsample at the WEAVER video rate
        # (RGB_SKIP), and take consecutive position DELTAS == the relabel action the
        # checkpoint was trained on. (The old code reconstructed deltas from the
        # env's cartesian-rate joint state + interpolated -> wrong rate -> exterior
        # drift; a raw joint-velocity sum is the wrong *scale* for a relabel model.)
        from weaver.robot.actions import RGB_SKIP

        robot = state.get("robot") if isinstance(state, dict) else None
        traj = robot.get("_pi_joint_traj") if isinstance(robot, dict) else None
        if traj is None:
            raise RuntimeError(
                "WeaverWorldModel needs the control-rate joint trajectory in "
                "state['robot']['_pi_joint_traj'] (set by OpenPIPolicy)."
            )
        traj = np.asarray(traj, dtype=np.float32)[:, :8]         # [n_ctrl, 8] abs joint+gripper

        cur_abs = np.asarray(st["joint_abs"], dtype=np.float32)
        sub_idx = np.arange(RGB_SKIP - 1, traj.shape[0], RGB_SKIP)   # video-rate frame indices
        vid_abs = traj[sub_idx] if sub_idx.size else cur_abs[None]   # [Tv, 8] absolute video-rate states
        # relabel action = consecutive video-rate position deltas (from current).
        deltas = np.diff(np.concatenate([cur_abs[None], vid_abs], axis=0), axis=0)  # [Tv, 8]
        fut_act = ((deltas - self._am) / self._asd).astype(np.float32)
        fut_state = ((vid_abs - self._sm) / self._ss).astype(np.float32)
        if fut_act.shape[0] < horizon:                           # pad tail (kept boot frames are real)
            pad = horizon - fut_act.shape[0]
            fut_act = np.concatenate([fut_act, np.zeros((pad, 8), np.float32)], axis=0)
            fut_state = np.concatenate([fut_state, np.repeat(fut_state[-1:], pad, 0)], axis=0)
        fut_act, fut_state = fut_act[:horizon], fut_state[:horizon]

        # History frames carry their real incoming deltas (carried from prev chunk).
        hist_act = np.asarray(st["hist_act"], dtype=np.float32)  # [n_hist, 8] normalized
        act_norm = np.concatenate([hist_act, fut_act], axis=0)   # [n_hist+horizon, 8]
        act = torch.from_numpy(act_norm).float()[None].to(self._device)  # (1, n_hist+horizon, 8)

        # x1_chunk: latent history + zero (noised) future
        x1_chunk = {}
        for wk in self.img_keys:
            h = x1_hist[wk]
            fut = torch.zeros((1, horizon) + tuple(h.shape[2:]), device=h.device, dtype=h.dtype)
            x1_chunk[wk] = torch.cat([h, fut], dim=1)
        hs = x1_hist["states"]
        x1_chunk["states"] = torch.cat(
            [hs, torch.zeros((1, horizon, hs.shape[-1]), device=hs.device, dtype=hs.dtype)], dim=1)

        with m.ema.use_ema_weights(), torch.autocast("cuda", dtype=torch.bfloat16):
            xt = m.generate_latent_rollouts(x1_chunk, act, memory_tokens=mem_tokens)

        # --- decode only the `boot` kept frames for the output video ---
        kept_lat = {wk: xt[wk][:, n_hist:n_hist + boot] for wk in self.img_keys}
        kept_lat["states"] = xt["states"][:, n_hist:n_hist + boot]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            decoded = m.decode_obs(kept_lat, chunk_size=16)
        per_view = [decoded[wk][0].float().cpu().permute(0, 2, 3, 1).clamp(0, 1).mul(255).byte().numpy()
                    for wk in self.img_keys]
        T = min(p.shape[0] for p in per_view)
        stacked = np.concatenate([p[:T] for p in per_view], axis=1)       # [T, V*H, W, 3]
        frames = [stacked[t] for t in range(T)]

        # --- advance latent state: history = last n_hist of kept window ---
        new_x1 = {wk: xt[wk][:, boot:n_hist + boot] for wk in self.img_keys}
        new_x1["states"] = xt["states"][:, boot:n_hist + boot]
        # update memory_tokens in latent space (slide window, append frame n_hist-1)
        N_per = mem_tokens.shape[1] // n_mem
        new_frame = {wk: xt[wk][:, n_hist - 1:n_hist] for wk in self.img_keys}
        new_frame["states"] = xt["states"][:, n_hist - 1:n_hist]
        with m.ema.use_ema_weights(), torch.autocast("cuda", dtype=torch.bfloat16):
            ne = m.wm.encode_memory(new_frame)
            za = torch.zeros(ne.shape[0], 1, m.wm._n_actions, device=ne.device, dtype=ne.dtype)
            ae = rearrange(m.wm.mem_inp_prj['actions'](za), 'b m d -> b m 1 d')
            te = rearrange(m.wm.timestep_encoder(torch.ones(ne.shape[0], device=ne.device)), 'b d -> b 1 1 d')
            nt = rearrange(torch.cat([ne, ae, te], dim=2), 'b m n d -> b (m n) d')
            mem_tokens = torch.cat([mem_tokens[:, N_per:], nt], dim=1)

        # Advance the integrated joint state to the end of the kept window, and
        # carry that window's last n_hist action deltas as next-chunk history actions
        # (the carried history frames are the kept frames [boot:n_hist+boot]).
        idx = min(boot - 1, fut_state.shape[0] - 1)
        new_joint_abs = (fut_state[idx] * self._ss + self._sm).astype(np.float32)
        new_hist_act = fut_act[max(0, boot - n_hist):boot]
        if new_hist_act.shape[0] < n_hist:
            new_hist_act = np.concatenate(
                [np.zeros((n_hist - new_hist_act.shape[0], 8), np.float32), new_hist_act], axis=0)
        st = {"x1": new_x1, "mem": mem_tokens, "joint_abs": new_joint_abs, "hist_act": new_hist_act}
        next_state = dict(state) if isinstance(state, dict) else {}
        next_state["_wv_lat"] = st

        if self.debug and self._rollout_count < self.debug_log_limit:
            logger.info(
                "weaver rollout[%d]: kept=%d | fut_act[%.3f,%.3f] act_n[%.2f,%.2f] out=%s",
                self._rollout_count, T, float(fut_act.min()), float(fut_act.max()),
                float(act_norm.min()), float(act_norm.max()), stacked.shape,
            )
        self._rollout_count += 1
        return {"frames": frames, "next_state": next_state, "latents": None}
