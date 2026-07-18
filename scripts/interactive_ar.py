"""Interactive, keyboard-controlled autoregressive world-model demo (browser UI).

Loads a trained AR student (Wan backbone + action conditioner), primes it with the
first frame(s) of a recorded episode, then lets YOU drive the robot's end-effector
live with the keyboard while the model autoregressively dreams the video. Everything
is served over a tiny dependency-free HTTP server (stdlib only): the generated video
is an MJPEG stream you watch in any browser, key state is POSTed back.

Controls (click a D-pad button / press a key for ONE normalized-EEF nudge, or
hold to keep moving block-by-block; the button stays lit until executed):
    w / s : +x / -x        (forward / back)
    a / d : +y / -y        (left / right)
    o / l : +z / -z        (up / down)
    k / j : open / close gripper
    (idle -> EEF holds its pose; the world pauses unless --roll-when-idle)

Run on a GPU node, then SSH-tunnel the port to your laptop:

    cd open-world
    .venv/bin/python scripts/interactive_ar.py \
        --config configs/inference/ar_wan_droid_2view_cartesian.py \
        --checkpoint <trained_student.pt> \
        --port 8000
    # from your laptop:  ssh -N -L 8000:<gpu-node>:8000 <you>@<cluster-login>
    # then open http://localhost:8000

Swap --config to an inference config matching the checkpoint's view/action-space.

TELEOPERATION (SpaceMouse): with a 4-step DISTILLED checkpoint, add --distilled
--continuous (and optionally --record-dir runs/teleop) here, then drive it from
your laptop with ``scripts/spacemouse_client.py`` (reads the SpaceMouse via
robosuite and POSTs the normalized pose to POST /action over the same tunnel).
See docs/TELEOPERATION.md.

INFERENCE & LATENCY OPTIONS (see claude_notes/speedup.md for the full analysis)
-----------------------------------------------------------------------------
Speedup flags (stack them):
    --bf16            pure-bf16 deploy (params + decode VAE), drop autocast (~1.3x)
    --static-cache    fixed-shape ring KV cache (needs finite --max-kv-blocks)
    --compile         torch.compile(reduce-overhead) the block loop (~1.41x/forward;
                      implies --static-cache + finite window; first blocks compile)
    --decode-context  rolling decode window (default 2; smaller = faster decode)
    --decode-device   run the VAE decode on a 2nd GPU, overlapped with the next block

THROUGHPUT vs FELT LATENCY -- these are NOT the same number, and they trade off:
  * One step() = generate + decode ONE block = frames_per_block latent frames =
    fpb*4 RGB frames (Wan's 4x temporal upsample). The "block_s" in /state is the
    THROUGHPUT period (how often a fresh block appears), not action->pixels latency.
  * Felt action->pixels latency = (queue wait) + (one block of generation) +
    (decode) + (display buffer) [+ network]. It is a MULTIPLE of block_s.
  * --decode-device is a THROUGHPUT win but a LATENCY COST: in overlapped (async)
    mode step() returns the PREVIOUS block's frames, so your action's pixels arrive
    one whole block later. For TELEOP FEEL prefer SYNCHRONOUS decode (omit
    --decode-device): a sync step (--bf16 --compile --static-cache, decode on the
    same GPU) has lower action->pixels latency even though its blk/s is lower.
    Generation already runs faster than world-time, so the overlap's extra
    throughput is not needed for teleop.

Recommended TELEOP config (lowest latency):
    --distilled --continuous --bf16 --compile --static-cache --max-kv-blocks N
    (NO --decode-device).  Recommended THROUGHPUT/recording config: add
    --decode-device cuda:1.

Latency hygiene already in this file (kept tight for teleop feel):
  * teleop poses are coalesced to a latest-pose queue (maxlen=1) and the FRESHEST
    pose is fed per block -- no model_hz time-throttle (that only added staleness).
  * the MJPEG playback buffer is bounded to ~1 block and paced to block production,
    so the displayed frame does not stand multiple blocks behind the latest.
  * the HUD polls /state and /ping sparingly so HTTP handlers don't steal the GIL
    from the single generator thread mid-block.

Use --measure-latency to log per-block action->pixels timing + p50/p90/p99 (and an
a2e_ms field in /state) so you can A/B sync-vs-async / 4-step-vs-2-step with NUMBERS
rather than feel.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Config files do ``from configs.training... import``; ``configs`` is not a package
# in the editable install, so put the repo root on sys.path regardless of cwd.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from openworld.autoregressive.data.decode import VaeLatentDecoder
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.infer import (
    InteractiveRoller,
    build_preview_scheduler,
    load_action_stats,
    load_full_episode,
    load_init_frame,
    normalize_actions,
)
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config
from openworld.utils.video import save_rollout_video


# Direction name -> (action index, sign). Indices: 0=x 1=y 2=z 3,4,5=orient 6=gripper.
#   forward/back -> +x/-x,  left/right -> +y/-y,  up/down -> +z/-z
DIRS = {
    "forward": (0, +1), "back": (0, -1),
    "left":    (1, +1), "right": (1, -1),
    "up":      (2, +1), "down":  (2, -1),
    "grip_open": (6, +1), "grip_close": (6, -1),
}

# Keyboard convenience: physical key -> direction (one nudge per keypress).
KEYMAP = {
    "w": "forward", "s": "back",
    "a": "left",    "d": "right",
    "o": "up",      "l": "down",
    "k": "grip_open", "j": "grip_close",
}


# --------------------------------------------------------------------------- #
# Live control state
# --------------------------------------------------------------------------- #
class Controls:
    """Shared control state for click-to-queue driving.

    A click (or keypress) enqueues ONE directional nudge; the generator consumes
    exactly one queued nudge per block. A direction renders as 'pressed' from the
    moment it is queued until the block conditioned on it finishes computing
    (i.e. while it sits in the queue OR is the action of the in-flight block).
    """

    def __init__(self, seed_action: np.ndarray, step: float, model_hz: float = 5.0):
        self._lock = threading.Lock()
        self.action = seed_action.astype(np.float32).copy()
        self.step = float(step)
        self._queue: deque[str] = deque()
        self._executing: str | None = None
        self._fresh = False
        # --- model_hz action downsampling ---------------------------------
        # The model's native action rate is `model_hz` (the dataset's latent/
        # action frequency); the SpaceMouse client POSTs at ~20Hz. We snapshot
        # the driven pose at most model_hz times/s (and only when it actually
        # CHANGES) into a FIFO queue, and the generator consumes exactly ONE
        # queued pose per AR block -- feeding the operator's trajectory to the
        # model in order, at its native frequency, with no free-run drift.
        self.model_hz = float(model_hz)
        self._sample_dt = 1.0 / max(0.1, float(model_hz))
        # Coalesce to the LATEST sampled pose (maxlen=1): when the model can't
        # keep up (consume ~2/s << ~5Hz input), newer poses REPLACE older ones
        # instead of piling into a long backlog that "replays" your whole session
        # after you stop. The generator always rolls toward your most recent
        # intent; intermediate poses during fast motion are dropped (latest-pose).
        self._aqueue: deque = deque(maxlen=1)         # holds (pose, t_enqueue) tuples
        self._last_sample_t = 0.0
        self._last_queued = self.action.copy()
        self.last_pop_t: float | None = None          # time.time() the last popped pose was enqueued

    def _enqueue_sample(self, now: float):
        """Enqueue the latest pose on any real change (assumes self._lock is held).

        ``_aqueue`` is maxlen=1 (latest-pose coalescing), so the generator always
        consumes the FRESHEST operator pose -- one per AR block. We deliberately do
        NOT throttle to ``model_hz`` here: with a latest-only queue a time-rate gate
        cannot prevent a backlog (there is none, maxlen=1), it only lets the pose the
        next block conditions on go up to ``1/model_hz`` (~200ms) stale -- pure
        action->pixels latency, and the per-block delta is unchanged either way (it's
        the operator's net motion between block consumptions). The change-gate (>1e-3)
        still suppresses re-POSTs of an unchanged pose so a still SpaceMouse triggers
        no rollout / AR drift."""
        if float(np.max(np.abs(self.action - self._last_queued))) > 1e-3:
            self._aqueue.append((self.action.copy(), now))   # (pose, t_enqueue) for latency
            self._last_queued = self.action.copy()
            self._last_sample_t = now
            self._fresh = True

    def has_queued(self) -> bool:
        with self._lock:
            return bool(self._aqueue)

    def pop_queued(self):
        with self._lock:
            if not self._aqueue:
                return None
            pose, t = self._aqueue.popleft()
            self.last_pop_t = t
            return pose

    def queued_depth(self) -> int:
        with self._lock:
            return len(self._aqueue)

    def set_step(self, step: float):
        with self._lock:
            self.step = float(step)

    # -- continuous (teleop) driving: an external device owns the absolute pose --
    def set_action(self, arr) -> np.ndarray:
        """Replace the absolute normalized pose (clipped to [-1, 1]).

        Used by the SpaceMouse client, which integrates device deltas on the
        laptop and POSTs the resulting absolute pose. Length must match the
        model's action_dim (the dim of the seed action)."""
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        now = time.time()
        with self._lock:
            if a.shape[0] != self.action.shape[0]:
                raise ValueError(
                    f"action dim {a.shape[0]} != expected {self.action.shape[0]}")
            self.action = np.clip(a, -1.0, 1.0)
            # Downsample to model_hz + change-gate: a still device re-POSTing
            # the same pose at 20Hz enqueues nothing (no rollout, no AR drift);
            # real motion lands at most model_hz samples/s in the FIFO queue.
            self._enqueue_sample(now)
            return self.action.copy()

    def apply_delta(self, arr) -> np.ndarray:
        """Add a normalized delta to the absolute pose (clipped to [-1, 1])."""
        d = np.asarray(arr, dtype=np.float32).reshape(-1)
        now = time.time()
        with self._lock:
            if d.shape[0] != self.action.shape[0]:
                raise ValueError(
                    f"delta dim {d.shape[0]} != expected {self.action.shape[0]}")
            self.action = np.clip(self.action + d, -1.0, 1.0)
            self._enqueue_sample(now)
            return self.action.copy()

    def fresh(self) -> bool:
        with self._lock:
            return self._fresh

    def clear_fresh(self):
        with self._lock:
            self._fresh = False

    def enqueue(self, direction: str):
        if direction in DIRS:
            with self._lock:
                # cap depth + dedupe so a held key can't pile up many queued nudges
                # (which would keep the button "pressed" for seconds while they drain)
                if direction not in self._queue and len(self._queue) < 2:
                    self._queue.append(direction)

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue) or self._executing is not None

    def take(self) -> np.ndarray | None:
        """Pop one queued nudge, apply it to the accumulated pose, mark it
        executing, and return the new action (or None if the queue is empty)."""
        with self._lock:
            if not self._queue:
                return None
            direction = self._queue.popleft()
            idx, sign = DIRS[direction]
            self.action[idx] = float(np.clip(self.action[idx] + sign * self.step, -1.0, 1.0))
            self._executing = direction
            return self.action.copy()

    def current(self) -> np.ndarray:
        with self._lock:
            return self.action.copy()

    def done(self):
        with self._lock:
            self._executing = None

    def _active(self) -> list[str]:
        dirs = set(self._queue)
        if self._executing is not None:
            dirs.add(self._executing)
        return sorted(dirs)

    def snapshot(self) -> dict:
        with self._lock:
            return {"action": [round(float(v), 3) for v in self.action],
                    "active": self._active(), "step": self.step,
                    "queued": len(self._aqueue)}


# --------------------------------------------------------------------------- #
# Frame hub: bounded FIFO of JPEG frames, paced out by the MJPEG stream
# --------------------------------------------------------------------------- #
class FrameHub:
    """Broadcast hub: every open stream gets its OWN bounded queue, so a stale
    connection left over from a browser refresh can't steal frames from the live
    viewer. The most recent frame is cached and replayed to any new subscriber so
    a freshly-loaded / refreshed tab paints immediately instead of going blank.
    """

    def __init__(self, maxlen: int = 96, play_maxlen: int = 16):
        self._cond = threading.Condition()
        self._subs: list[deque[bytes]] = []
        self._last: bytes | None = None
        self._maxlen = maxlen
        # FIFO playback buffer drained one frame per /frame poll -> smooth, in-order
        # playback of each generation burst (bounded so latency stays low).
        self._play: deque[bytes] = deque(maxlen=play_maxlen)

    def push(self, jpeg: bytes):
        with self._cond:
            self._last = jpeg
            self._play.append(jpeg)
            for q in self._subs:
                q.append(jpeg)
                while len(q) > self._maxlen:        # bound a slow/zombie client
                    q.popleft()
            self._cond.notify_all()

    def next(self) -> bytes | None:
        """/frame: pop the next queued frame in order; hold last when drained."""
        with self._cond:
            if self._play:
                self._last = self._play.popleft()
            return self._last

    def subscribe(self) -> deque[bytes]:
        q: deque[bytes] = deque()
        with self._cond:
            if self._last is not None:
                q.append(self._last)                # instant paint on connect
            self._subs.append(q)
        return q

    def unsubscribe(self, q: deque[bytes]):
        with self._cond:
            if q in self._subs:
                self._subs.remove(q)

    def clear(self):
        """Drop the cached last frame and every subscriber's backlog (used on
        reseed so old-episode frames don't drain interleaved with the new ones)."""
        with self._cond:
            self._last = None
            self._play.clear()
            for q in self._subs:
                q.clear()

    def last(self) -> bytes | None:
        with self._cond:
            return self._last

    def pop_from(self, q: deque[bytes], timeout: float = 1.0):
        with self._cond:
            if not q:
                self._cond.wait(timeout=timeout)
            return q.popleft() if q else None

    def size(self) -> int:
        """Depth of the /frame playback buffer (drives generator backpressure)."""
        with self._cond:
            return len(self._play)


# --------------------------------------------------------------------------- #
# Latency meter: action -> pixels timing (the number that governs teleop feel)
# --------------------------------------------------------------------------- #
class LatencyMeter:
    """Track the action->pixels latency and its components, per AR block.

    ``a2e`` (action->emit) is the wall time from when an operator pose became
    available (enqueued) to when the block conditioned on it is pushed to the
    frame hub -- the controllable, server-side part of "I move, then I see it".
    It is broken down into ``qwait`` (time the pose sat queued before the
    generator consumed it) and the ``step``/``fwd`` compute. Network RTT
    (laptop<->server) is measured separately by the HUD's /ping and adds on top;
    the residual display-buffer lag is surfaced as ``buf`` (frames buffered at
    emit) since the playback buffer drains at block rate.

    Always tracks ``last_a2e_ms`` cheaply (exposed in /state); only logs per-block
    lines + periodic percentile summaries when ``enabled`` (--measure-latency)."""

    def __init__(self, enabled: bool, log_every: float = 5.0):
        self.enabled = enabled
        self.log_every = log_every
        self.last_a2e_ms = 0.0
        self._a2e: deque[float] = deque(maxlen=1024)
        self._qwait: deque[float] = deque(maxlen=1024)
        self._step: deque[float] = deque(maxlen=1024)
        self._fwd: deque[float] = deque(maxlen=1024)
        self._t_last_log = time.perf_counter()

    def add(self, a2e_s: float, qwait_s: float, step_s: float, fwd_s: float,
            async_mode: bool, buf_depth: int):
        self.last_a2e_ms = a2e_s * 1e3
        if not self.enabled:
            return
        self._a2e.append(a2e_s * 1e3); self._qwait.append(qwait_s * 1e3)
        self._step.append(step_s * 1e3); self._fwd.append(fwd_s * 1e3)
        print(f"[lat] a2e={a2e_s*1e3:6.0f}ms  qwait={qwait_s*1e3:5.0f}  step={step_s*1e3:5.0f}  "
              f"fwd={fwd_s*1e3:5.0f}  buf={buf_depth:2d}  {'async' if async_mode else 'sync'}",
              flush=True)
        now = time.perf_counter()
        if now - self._t_last_log >= self.log_every:
            self.summary(); self._t_last_log = now

    @staticmethod
    def _pct(d, p):
        s = sorted(d); return s[min(len(s) - 1, int(p * len(s)))]

    def summary(self):
        if not self._a2e:
            return
        a, q, st, fw = self._a2e, self._qwait, self._step, self._fwd
        print(f"[lat] === action->pixels over {len(a)} blocks (ms): "
              f"a2e p50={self._pct(a,.5):.0f} p90={self._pct(a,.9):.0f} p99={self._pct(a,.99):.0f}  |  "
              f"qwait p50={self._pct(q,.5):.0f}  step p50={self._pct(st,.5):.0f}  "
              f"fwd p50={self._pct(fw,.5):.0f} ===", flush=True)


# --------------------------------------------------------------------------- #
# Recorder: capture the teleoperated trajectory (actions + dreamed frames)
# --------------------------------------------------------------------------- #
class Recorder:
    """Accumulate the live session so it can be replayed/inspected later.

    One ``action`` is logged per generated AR block (the normalized EEF pose the
    block was conditioned on); ``frames_per_block * 4`` RGB frames are produced
    per block (Wan's 4x temporal upsample), so frames and actions differ in
    count by that cadence (recorded in ``meta.json``). On :meth:`save` the
    normalized actions are also de-normalized back to raw units via the same
    percentile stats used to normalize them, so the dump is directly comparable
    to dataset actions.
    """

    def __init__(self, out_dir: str, p01: np.ndarray, p99: np.ndarray,
                 meta: dict, fps: int, max_frames: int = 6000):
        self.out_dir = out_dir
        self.p01 = np.asarray(p01, dtype=np.float32)
        self.p99 = np.asarray(p99, dtype=np.float32)
        self.meta = dict(meta)
        self.fps = int(fps)
        self.max_frames = int(max_frames)
        self._lock = threading.Lock()
        self.actions: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []
        self._warned = False

    def add(self, action_norm: np.ndarray, rgb: np.ndarray):
        with self._lock:
            self.actions.append(np.asarray(action_norm, dtype=np.float32).copy())
            for frame in rgb:                       # [H, W, 3] uint8
                if len(self.frames) < self.max_frames:
                    self.frames.append(np.asarray(frame))
                elif not self._warned:
                    print(f"[interactive] recorder hit --record-max-frames "
                          f"({self.max_frames}); dropping further frames "
                          f"(actions still logged)", flush=True)
                    self._warned = True

    def save(self):
        with self._lock:
            if not self.actions:
                print("[interactive] recorder: nothing to save", flush=True)
                return
            os.makedirs(self.out_dir, exist_ok=True)
            actions_norm = np.stack(self.actions, axis=0)                 # [B, A]
            # de-normalize: inverse of clip(2*(a-p01)/(p99-p01)-1, -1, 1)
            actions_raw = (actions_norm + 1.0) / 2.0 * (self.p99 - self.p01) + self.p01
            np.save(os.path.join(self.out_dir, "actions_norm.npy"), actions_norm)
            np.save(os.path.join(self.out_dir, "actions_raw.npy"), actions_raw)
            meta = {**self.meta, "num_blocks": int(actions_norm.shape[0]),
                    "num_frames": int(len(self.frames)), "fps": self.fps}
            with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            if self.frames:
                save_rollout_video(self.frames,
                                   os.path.join(self.out_dir, "teleop.mp4"), fps=self.fps)
            print(f"[interactive] recorder: wrote {actions_norm.shape[0]} actions "
                  f"+ {len(self.frames)} frames to {self.out_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Engine: owns the model + roller + generator thread (all GPU work, one thread)
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = _load_config(args.config)
        if args.latent_root:
            self.cfg.latent_root = args.latent_root
        self.split = args.split
        self.fpb = self.cfg.frames_per_block

        # --- speedup flag resolution / validation (see claude_notes/speedup.md) ---
        # --compile needs the static (fixed-shape) cache to be correct + fast under
        # reduce-overhead; the static cache needs a finite sliding window.
        self._static_cache = args.static_cache or args.compile
        self._max_kv_blocks = args.max_kv_blocks if args.max_kv_blocks >= 0 else self.cfg.max_kv_blocks
        if self._static_cache and self._max_kv_blocks is None:
            raise SystemExit(
                "--static-cache/--compile require a finite --max-kv-blocks (the attention window); "
                f"cfg.max_kv_blocks is unbounded. Try --max-kv-blocks {self.cfg.num_history_blocks + 8}.")
        if args.compile:
            import torch._dynamo as _dyn
            _dyn.config.recompile_limit = 16
            torch.set_grad_enabled(False)   # cudagraph capture needs grad off

        print(f"[interactive] building ARWorldModel ({self.cfg.backbone}, "
              f"action_cond_mode={self.cfg.action_cond_mode}) ...", flush=True)
        self.model = ARWorldModel(self.cfg).to(self.device).eval()
        if args.checkpoint:
            print(f"[interactive] loading checkpoint {args.checkpoint}", flush=True)
            sd = torch.load(args.checkpoint, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[interactive] {len(missing)} missing keys (e.g. {missing[:3]})")
            if unexpected:
                print(f"[interactive] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        else:
            print("[interactive] WARNING: no --checkpoint; untrained weights (plumbing smoke only).")

        # Pure-bf16 deploy: cast AFTER loading the fp32 checkpoint (keeps checkpoint
        # values exact pre-cast); the roller then drops autocast (autocast_dtype=None).
        if args.bf16:
            self.model = self.model.to(torch.bfloat16)
            print("[interactive] bf16 deploy: params cast to bfloat16 (autocast off).", flush=True)
        if args.compile:
            print("[interactive] torch.compile(reduce-overhead) block loop "
                  "(first few blocks compile, then steady-state) ...", flush=True)
            self.model.backbone.compile_blocks(mode="reduce-overhead")

        from diffusers import AutoencoderKLWan
        # Decode device: same GPU as the model by default, or a second GPU for the
        # overlapped (pipelined) decode. Putting the VAE on the decode device frees
        # the generation GPU from the ~57%-of-step decode cost.
        self._decode_device = torch.device(args.decode_device) if args.decode_device else self.device
        dec_dtype = torch.bfloat16 if args.bf16 else torch.float32
        vae = AutoencoderKLWan.from_pretrained(args.vae_dir, subfolder="vae", torch_dtype=dec_dtype)
        self.decoder = VaeLatentDecoder(vae, device=self._decode_device, dtype=dec_dtype)

        # Two sampling regimes:
        #   --distilled : few-step DEPLOYMENT schedule (cfg.denoising_step_list,
        #                 warped) -- correct for a 4-step DISTILLED checkpoint.
        #   default     : many-step uniform PREVIEW schedule -- correct for a
        #                 mid-training/non-distilled (studentinit) backbone, on
        #                 which the 4-step list would yield a blurry colour-wash.
        if args.distilled:
            sched = FlowMatchScheduler(
                self.cfg.denoising_step_list,
                num_train_timestep=self.cfg.num_train_timestep,
                warp=self.cfg.warp_denoising_step,
            )
            n_steps = sched.num_steps
            print(f"[interactive] DISTILLED deployment schedule: {n_steps} steps "
                  f"{tuple(self.cfg.denoising_step_list)} "
                  f"(warp={self.cfg.warp_denoising_step})", flush=True)
        else:
            n_steps = args.denoising_steps or self.cfg.preview_denoising_steps
            print(f"[interactive] sampling with {n_steps} denoising steps/block "
                  f"(non-distilled backbone -> many-step uniform schedule)", flush=True)
            sched = build_preview_scheduler(n_steps, num_train_timestep=self.cfg.num_train_timestep)

        self._async_decode = self._decode_device != self.device
        async_decoder = None
        if self._async_decode:
            from openworld.autoregressive.infer.interactive import AsyncWindowDecoder
            async_decoder = AsyncWindowDecoder(
                self.decoder, num_cams=self.cfg.num_cams, emit_latent_frames=self.fpb)
            print(f"[interactive] overlapped decode on {self._decode_device} "
                  f"(generation on {self.device}); one-block display lag.", flush=True)
        self.roller = InteractiveRoller(
            self.model, self.decoder, num_cams=self.cfg.num_cams, scheduler=sched,
            device=self.device,
            autocast_dtype=None if args.bf16 else torch.bfloat16,
            max_kv_blocks=self._max_kv_blocks,
            decode_context=args.decode_context,
            async_decoder=async_decoder,
            static_cache=self._static_cache,
        )
        stats_file = getattr(self.cfg, "stats_file", None) or "stats.json"
        # Action-normalization percentiles. Normally these live in --latent-root next
        # to the preprocessed episodes, but the latent-free teleop path (--benchmark-root
        # only, no .pt episodes) falls back to a stats.json bundled with the inits so a
        # fresh clone needs nothing downloaded -- see assets/teleop_inits/.
        try:
            self.p01, self.p99 = load_action_stats(self.cfg.latent_root, stats_file)
        except (FileNotFoundError, TypeError):
            self.p01, self.p99 = load_action_stats(args.benchmark_root, stats_file)
            print(f"[interactive] no {stats_file} under latent-root; using the one bundled "
                  f"with --benchmark-root ({args.benchmark_root})", flush=True)

        # Bound the playback buffer to ~1 block. During teleop the generator skips
        # backpressure (run()), so a larger buffer just stands full -> the displayed
        # frame trails the freshest by that depth (pure latency). 1 block keeps
        # intra-block playback smooth (a new block's frames are never dropped; only a
        # stale prior-block tail is trimmed) while minimizing standing display lag.
        self.hub = FrameHub(play_maxlen=self.fpb * 4)
        self.controls: Controls | None = None
        self.jpeg_quality = args.jpeg_quality
        self.fps = args.fps
        self._stop = threading.Event()
        self.status = "loading"
        self._compute_t0 = 0.0
        self.n_frames = 0                  # total RGB frames emitted to the stream
        # action->pixels latency meter (always tracks last_a2e_ms cheaply; verbose
        # per-block logging only with --measure-latency). In async (--decode-device)
        # mode step() emits the PREVIOUS block, so the frames emitted this iteration
        # belong to the action consumed last iteration -- carried in _prev_action_t.
        self._meter = LatencyMeter(enabled=bool(args.measure_latency))
        self._prev_action_t: float | None = None
        self._prev_consume_t: float | None = None
        self._seed_lock = threading.Lock()
        self._pending_seed: tuple[str, str] | None = None   # (source, id): source in {droid, benchmark}
        self.cur_episode: str | None = None
        self._warmed = False               # compiled graphs warmed once at first seed
        # Prime with the model's TRAINED history depth by default. The old
        # hardcoded 1 under-conditions the backbone (cfg.num_history_blocks is
        # 2) AND decodes the seed from too few latents -> blurry/bad initial
        # image. -1 -> use the cfg value; an explicit >0 still overrides.
        self.history_blocks = (args.history_blocks if args.history_blocks > 0
                               else int(self.cfg.num_history_blocks))
        self.step_size = args.step_size
        self.rotate_wrist = bool(getattr(args, "rotate_wrist", False))   # display-only wrist 180 deg
        # Teleop wants the world to keep dreaming while the SpaceMouse holds a
        # pose, so --continuous (like --roll-when-idle) disables the idle pause.
        self.pause_when_idle = not (args.roll_when_idle or args.continuous)

        # Optional trajectory recorder (actions + dreamed frames -> --record-dir).
        self.recorder: Recorder | None = None
        if args.record_dir:
            meta = {
                "config": args.config, "checkpoint": args.checkpoint,
                "distilled": bool(args.distilled), "denoising_steps": int(n_steps),
                "num_cams": int(self.cfg.num_cams), "frames_per_block": int(self.fpb),
                "rgb_frames_per_block": int(self.fpb * 4),
                "action_space": getattr(self.cfg, "action_space", "cartesian"),
                "action_dim": int(self.cfg.action_dim), "stats_file": stats_file,
            }
            self.recorder = Recorder(args.record_dir, self.p01, self.p99,
                                     meta, fps=args.fps,
                                     max_frames=args.record_max_frames)
            print(f"[interactive] recording trajectory to {args.record_dir}", flush=True)

        # Two seed sources. (1) DROID episodes: the first frames of a preprocessed
        # .pt clip under --latent-root (needs the latents downloaded). (2) Benchmark
        # inits: a scenegen suite of still initializations (per-view PNGs +
        # initialization.yaml) under --benchmark-root, VAE-encoded on demand and primed
        # by repeating the single latent frame across the history block. The init path
        # is latent-free -- a couple ship in assets/teleop_inits/ so a fresh clone can
        # teleop with nothing downloaded but the model checkpoint.
        self.benchmark_root = args.benchmark_root
        self.episodes = self._list_episodes()
        self.inits = self._list_inits()
        self._enc = None                       # lazily-built VaeLatentEncoder (init seeds only)
        if self.inits:
            print(f"[interactive] {len(self.inits)} benchmark inits under {self.benchmark_root}",
                  flush=True)
        if not self.episodes and not self.inits:
            raise RuntimeError(
                f"no seed sources: no episodes under {self.cfg.latent_root}/{self.split} "
                f"and no benchmark inits under {self.benchmark_root}")
        # Default seed: a DROID episode if any latents are present, else the first
        # benchmark init (latent-free path).
        if self.episodes:
            self._pending_seed = ("droid", args.seed_episode or self.episodes[0])
        else:
            self._pending_seed = ("benchmark", args.seed_episode or self.inits[0])

    # -- seeding ---------------------------------------------------------
    def _list_episodes(self) -> list[str]:
        if not self.cfg.latent_root or not os.path.isdir(os.path.join(self.cfg.latent_root, self.split)):
            return []
        paths = sorted(glob.glob(os.path.join(self.cfg.latent_root, self.split, "*.pt")))
        return [Path(p).stem for p in paths]

    def _list_inits(self) -> list[str]:
        """Subdirs of the benchmark suite holding a still initialization."""
        if not self.benchmark_root or not os.path.isdir(self.benchmark_root):
            return []
        out = []
        for name in sorted(os.listdir(self.benchmark_root)):
            d = os.path.join(self.benchmark_root, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "initialization.yaml")):
                out.append(name)
        return out

    def _encoder(self):
        """Lazy RGB->latent encoder (reuses the decode VAE), for benchmark-init seeds."""
        if self._enc is None:
            from openworld.autoregressive.data.encode import VaeLatentEncoder
            self._enc = VaeLatentEncoder(
                self.decoder.vae, device=self._decode_device, dtype=self.decoder.dtype)
        return self._enc

    def request_seed(self, ep_id: str, source: str = "droid"):
        with self._seed_lock:
            self._pending_seed = (source, ep_id)

    def _do_seed(self, ep_id: str):
        self.status = "seeding"
        self.hub.clear()                      # drop old-episode frames so reseed paints cleanly
        latent_gt, action_raw, text = load_full_episode(
            self.cfg.latent_root, self.split, ep_id, self.cfg.num_cams, self.cfg.wrist_view_idx,
            view_indices=getattr(self.cfg, "view_indices", None))
        hist_frames = self.history_blocks * self.fpb
        if latent_gt.shape[0] < hist_frames:
            raise RuntimeError(f"episode {ep_id} too short ({latent_gt.shape[0]} < {hist_frames})")
        action_norm = normalize_actions(action_raw, self.p01, self.p99)
        seed_actions = action_norm[:hist_frames]
        history_latents = latent_gt[:hist_frames]
        self._prime(history_latents, seed_actions, ep_id, text)

    def _do_seed_init(self, init_id: str):
        """Prime from a still benchmark initialization: encode the per-view PNGs to a
        single latent frame and REPEAT it to fill the history block (no recorded
        clip exists -- only one initial still), with the robot's initial pose as the
        (constant) seed action."""
        self.status = "seeding"
        self.hub.clear()
        init_dir = os.path.join(self.benchmark_root, init_id)
        latent1, action_raw, text = load_init_frame(
            init_dir, self.cfg.num_cams, self.cfg.wrist_view_idx, self._encoder(),
            getattr(self.cfg, "action_space", "cartesian"),
            view_indices=getattr(self.cfg, "view_indices", None))
        hist_frames = self.history_blocks * self.fpb
        history_latents = latent1.repeat(hist_frames, 1, 1, 1)   # [hist_frames, C, V*h, w]
        a = normalize_actions(np.asarray(action_raw)[None], self.p01, self.p99)[0]
        seed_actions = np.broadcast_to(a, (hist_frames, a.shape[0])).astype(np.float32).copy()
        self._prime(history_latents, seed_actions, f"init:{init_id}", text)

    def _prime(self, history_latents, seed_actions, ep_label: str, text: str):
        """Shared (re)seed: prime the cache, snap to the initial frame, warm compiled
        graphs once. Used by both DROID-episode and benchmark-init seeding."""
        rgb0 = self.roller.reset(history_latents, seed_actions)
        self.controls = Controls(seed_actions[-1], self.step_size, model_hz=self.args.model_hz)
        self.cur_episode = ep_label
        if self.recorder is not None:
            self.recorder.meta["seed_episode"] = ep_label
        print(f"[interactive] seeded from {ep_label}  ({text!r})", flush=True)
        # Instant snap to the initial frame on (re)seed: show only the FINAL primed
        # frame, not the whole GT priming clip (which looked like "an action played").
        self._emit(rgb0[-1:])
        # One-time compile warmup. --compile lazily compiles the block loop on the
        # FIRST step(), a ~10-20s stall. Without this it lands on the operator's first
        # SpaceMouse move: the display freezes mid-motion while their hand keeps
        # integrating a large delta, so the first emitted block renders a big jump and
        # looks blurry. Pay it here at boot instead -- the clean frame above is already
        # cached/painted (warmup steps below emit nothing), then re-prime so the world
        # is back at the pristine initial frame. reset() reuses the static KV buffers
        # (idempotent attach), so the captured graph stays valid.
        if getattr(self.args, "compile", False) and not self._warmed:
            self.status = "warming"
            tw = time.time()
            print("[interactive] warming compiled graphs (one-time, ~10-20s) ...", flush=True)
            warm_pose = seed_actions[-1].astype(np.float32)
            for _ in range(4):
                self.roller.step(warm_pose)            # compiles denoise + commit graphs; frames discarded
            if self._async_decode:
                self.roller.flush()
            self.roller.reset(history_latents, seed_actions)   # restore pristine cache (warm, fast)
            self.controls = Controls(seed_actions[-1], self.step_size, model_hz=self.args.model_hz)
            self._warmed = True
            self.status = "idle"
            print(f"[interactive] warmup done in {time.time() - tw:.1f}s; pristine initial frame ready.",
                  flush=True)

    # -- frame emission --------------------------------------------------
    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply DISPLAY-ONLY transforms to one stacked frame ``[V*H, W, 3]``.

        Currently: with ``--rotate-wrist``, rotate the wrist-view band 180 deg so the
        live stream is intuitive given the physical camera mount. The wrist camera is
        always the BOTTOM band of the height-stack (see ``data/views.py``). This is a
        pure presentation tweak on a COPY -- the model's latents/actions and the
        recorded video (which gets the untouched ``rgb``) keep the true orientation."""
        if not self.rotate_wrist:
            return frame
        v = int(self.cfg.num_cams)
        h = frame.shape[0]
        if v <= 0 or h % v != 0:
            return frame
        band = h // v
        out = frame.copy()
        out[(v - 1) * band: v * band] = frame[(v - 1) * band: v * band][::-1, ::-1]  # 180 deg
        return out

    def _emit(self, rgb: np.ndarray):
        for frame in rgb:                               # [V*H, W, 3] uint8
            buf = io.BytesIO()
            Image.fromarray(self._display_frame(frame)).save(buf, format="JPEG", quality=self.jpeg_quality)
            self.hub.push(buf.getvalue())
            self.n_frames += 1

    # -- generator loop (the only thread that touches the GPU) -----------
    def run(self):
        # keep ~1 block buffered so the displayed frame tracks the live state
        max_buffered = self.fpb * 4
        while not self._stop.is_set():
            with self._seed_lock:
                pending = self._pending_seed
                self._pending_seed = None
            if pending is not None:
                src, pid = pending
                try:
                    if src == "benchmark":
                        self._do_seed_init(pid)
                    else:
                        self._do_seed(pid)
                except Exception as e:                  # noqa: BLE001
                    print(f"[interactive] seed failed: {e!r}", flush=True)
                    time.sleep(1.0)
                continue
            if self.controls is None:
                time.sleep(0.05)
                continue
            pending = self.controls.has_pending()      # keyboard nudge waiting/executing
            queued = self.controls.has_queued()        # a downsampled (model_hz) teleop pose waiting
            # pause the world while idle: only roll forward when a keyboard nudge
            # is pending or a downsampled teleop pose is queued. --continuous /
            # --roll-when-idle keep dreaming at the current pose even when idle.
            if not pending and not queued and self.pause_when_idle:
                # Going idle: drain the one block still in flight on the decode GPU
                # so the latest frame shows even when no further step follows.
                if self._async_decode:
                    rgb, rec_action = self.roller.flush()
                    if rgb is not None:
                        emit_t = time.time()
                        self._emit(rgb)
                        if self._prev_action_t is not None:    # the drained block's action
                            consume = (self._prev_consume_t if self._prev_consume_t is not None
                                       else self._prev_action_t)
                            self._meter.add(emit_t - self._prev_action_t, consume - self._prev_action_t,
                                            getattr(self, "_last_block_s", 0.0), self.roller.last_fwd_s,
                                            True, self.hub.size())
                            self._prev_action_t = None         # drained; don't double-count
                        if self.recorder is not None and rec_action is not None:
                            self.recorder.add(rec_action, rgb)
                self.status = "idle"
                time.sleep(0.03)
                continue
            # backpressure: don't dream too far ahead of playback. In queue-driven
            # teleop the bounded action queue already limits how far we run and the
            # frame buffers drop-oldest, so SKIP this gate whenever a teleop pose is
            # queued -- otherwise a slow/stalled viewer that stops draining _play
            # deadlocks the generator (it never pushes, so the stream never drains).
            if not queued and self.hub.size() >= max_buffered:
                time.sleep(0.02)
                continue
            action = self.controls.take()              # keyboard nudge -> absolute pose, or None
            action_t = time.time() if action is not None else None  # keyboard: queue wait ~unmeasured
            if action is None:
                action = self.controls.pop_queued()    # freshest teleop pose (latest-pose queue)
                if action is not None:
                    action_t = self.controls.last_pop_t  # when this pose was enqueued (server-side)
            if action is None:
                action = self.controls.current()       # idle-roll (--continuous): hold current pose
                action_t = None                        # no new input -> not a latency sample
            self.controls.clear_fresh()
            self.status = "computing"
            self._compute_t0 = time.time()
            try:
                rgb = self.roller.step(action)
                self._last_block_s = time.time() - self._compute_t0
                emit_t = time.time()
                # async mode: step() returns the PREVIOUS block's frames (None on the
                # first step after a reseed); record them against the action that
                # produced them (last_emitted_action) to stay aligned despite the lag.
                if rgb is not None:
                    self._emit(rgb)
                    # action->pixels latency. SYNC: emitted frames are THIS action's.
                    # ASYNC: they belong to the action consumed last iteration.
                    if self._async_decode:
                        src_t, src_consume = self._prev_action_t, self._prev_consume_t
                    else:
                        src_t, src_consume = action_t, self._compute_t0
                    if src_t is not None:
                        consume = src_consume if src_consume is not None else src_t
                        self._meter.add(emit_t - src_t, consume - src_t, self._last_block_s,
                                        self.roller.last_fwd_s, self._async_decode, self.hub.size())
                    if self.recorder is not None:
                        rec_action = (self.roller.last_emitted_action
                                      if self._async_decode else action)
                        if rec_action is not None:
                            self.recorder.add(rec_action, rgb)
            except Exception:
                import traceback; traceback.print_exc()   # keep the generator thread alive
            finally:
                # release the 'pressed' state only once the block has computed
                self.controls.done()
                self.status = "idle"
                # async: this iteration's action's frames come back NEXT iteration --
                # carry its timing so that emit can attribute the latency correctly.
                self._prev_action_t, self._prev_consume_t = action_t, self._compute_t0

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                      # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = INDEX_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/ping":
                # cheap round-trip probe so the browser can measure tunnel RTT
                self._json({"t": time.time()})
            elif self.path == "/state":
                ctrl = engine.controls.snapshot() if engine.controls else {}
                computing_for = (round(time.time() - engine._compute_t0, 2)
                                 if engine.status == "computing" else 0.0)
                self._json({"episode": engine.cur_episode,
                            "block_s": round(getattr(engine, "_last_block_s", 0.0), 3),
                            "status": engine.status, "computing_for": computing_for,
                            "frames": engine.n_frames,
                            "fwd_ms": round(getattr(engine.roller, "last_fwd_s", 0.0) * 1000.0, 1),
                            "a2e_ms": round(engine._meter.last_a2e_ms, 1),  # action->pixels latency
                            **ctrl})
            elif self.path == "/episodes":
                self._json({"episodes": engine.episodes[:500], "current": engine.cur_episode})
            elif self.path == "/inits":
                self._json({"inits": engine.inits[:500], "current": engine.cur_episode})
            elif self.path == "/stats":
                # action normalization stats, so a client can den/re-normalize pose
                # dims to do proper rotation composition (see spacemouse_client.py).
                self._json({"action_space": getattr(engine.cfg, "action_space", "cartesian"),
                            "action_dim": int(engine.cfg.action_dim),
                            "p01": [float(v) for v in engine.p01],
                            "p99": [float(v) for v in engine.p99]})
            elif self.path.split("?")[0] == "/frame":
                j = engine.hub.next()
                if j is None:
                    self.send_error(503); return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers(); self.wfile.write(j)
            elif self.path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/press":
                body = self._read_body()
                if engine.controls:
                    engine.controls.enqueue(str(body.get("dir", "")))
                self._json({"ok": True})
            elif self.path == "/action":
                # Continuous teleop: an external device (e.g. the SpaceMouse
                # client) owns the absolute normalized pose. Accepts either an
                # absolute {"action": [...]} or an incremental {"delta": [...]}.
                body = self._read_body()
                if not engine.controls:
                    self._json({"ok": False, "error": "not seeded yet"}, code=409)
                    return
                try:
                    if "action" in body:
                        pose = engine.controls.set_action(body["action"])
                    elif "delta" in body:
                        pose = engine.controls.apply_delta(body["delta"])
                    else:
                        self._json({"ok": False, "error": "need 'action' or 'delta'"}, code=400)
                        return
                except ValueError as e:
                    self._json({"ok": False, "error": str(e)}, code=400)
                    return
                self._json({"ok": True, "action": [round(float(v), 4) for v in pose]})
            elif self.path == "/config":
                body = self._read_body()
                if "step" in body and engine.controls:
                    engine.controls.set_step(float(body["step"]))
                self._json({"ok": True})
            elif self.path == "/seed":
                body = self._read_body()
                ep = str(body.get("episode", "")).strip()
                # source: "droid" (val-split episode, default) or "benchmark" (a
                # still scenegen init under --benchmark-root).
                source = str(body.get("source", "droid")).strip() or "droid"
                if ep:
                    engine.request_seed(ep, source)
                self._json({"ok": True, "episode": ep, "source": source})
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            period = 1.0 / max(1, engine.fps)
            q = engine.hub.subscribe()
            try:
                while not engine._stop.is_set():
                    jpeg = engine.hub.pop_from(q, timeout=1.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    engine.hub.next()  # drain _play so backpressure tracks this live viewer
                    # Pace to block production: spread each block's RGB frames
                    # over the wall time the model spends per block, so the
                    # displayed video frequency tracks how fast blocks (hence
                    # the operator's queued input) arrive -- no bursts/starving.
                    rgbpb = max(1, engine.fpb * 4)
                    bs = getattr(engine, "_last_block_s", 0.0) or (1.0 / max(1, engine.fps))
                    time.sleep(min(0.25, max(0.005, bs / rgbpb)))
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                engine.hub.unsubscribe(q)

    return Handler


INDEX_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AR world model — live control</title>
<style>
  body{background:#111;color:#ddd;font-family:ui-monospace,Menlo,monospace;margin:0;padding:16px}
  #wrap{display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap}
  h1{font-size:16px;margin:0 0 12px}
  /* --- video column --- */
  #view{background:#000;border:1px solid #333;image-rendering:pixelated;max-height:82vh;display:block}
  #react{margin-top:10px;font-size:15px}
  #react b{color:#4caf50;font-variant-numeric:tabular-nums}
  /* --- D-pad column --- */
  .padcol{min-width:240px}
  .pad{display:grid;grid-template-columns:repeat(3,72px);grid-template-rows:repeat(3,72px);gap:8px}
  .zpad{display:grid;grid-template-rows:repeat(2,72px);gap:8px;margin-top:8px;width:72px}
  .btn{background:#1c1c1c;color:#ddd;border:1px solid #555;border-radius:8px;cursor:pointer;
       font-family:inherit;font-size:13px;line-height:1.25;transition:background .05s,border-color .05s;
       user-select:none}
  .btn:hover{border-color:#888}
  .btn small{color:#888;font-size:11px}
  .btn.pressed{background:#2e7d32;border-color:#4caf50;color:#fff}
  .btn.pressed small{color:#cde}
  .arrow{font-size:20px;display:block}
  /* --- info column --- */
  .panel{min-width:280px}
  table{border-collapse:collapse;margin-top:6px}
  td{padding:2px 10px 2px 0;font-variant-numeric:tabular-nums}
  .bar{height:8px;background:#333;border-radius:4px;position:relative;width:160px;display:inline-block;vertical-align:middle}
  .bar>i{position:absolute;top:0;bottom:0;width:2px;background:#4caf50}
  input[type=range]{width:200px}
  select,#reseed{background:#1c1c1c;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 8px}
  .hint{color:#888;font-size:12px;line-height:1.7}
  .legend td{padding:2px 12px 2px 0}
  .kcap{display:inline-block;min-width:20px;text-align:center;border:1px solid #555;border-radius:4px;
        padding:1px 5px;background:#1c1c1c;color:#bbb}
  h2{font-size:13px;color:#999;margin:18px 0 4px;font-weight:600}
  .statusbadge{margin-top:10px;font-size:15px;font-weight:600;padding:6px 11px;border-radius:6px;display:inline-block;font-variant-numeric:tabular-nums}
  .statusbadge.ready{background:#14361a;color:#7fe39a;border:1px solid #2e7d32}
  .statusbadge.busy{background:#3a2f12;color:#ffce5a;border:1px solid #b8860b}
  .statusbadge.wait{background:#3a1a1a;color:#ff9a9a;border:1px solid #a33}
  .telemetry{margin-top:8px;font-size:13px;color:#bbb;font-variant-numeric:tabular-nums;line-height:1.9}
  .telemetry b{color:#7fe39a}
  .telemetry b.warn{color:#ffce5a}.telemetry b.bad{color:#ff6b6b}
</style></head>
<body>
<div id="wrap">

  <!-- 1. video + reaction time -->
  <div>
    <img id="view" src="/stream" alt="stream">
    <div id="status" class="statusbadge wait">● connecting…</div>
    <div class="telemetry">
      forward pass (model compute): <b id="fwd">—</b> ms &nbsp;·&nbsp;
      network (tunnel RTT): <b id="net">—</b> ms<br>
      reaction (full block): <b id="blk">—</b> s &nbsp;·&nbsp;
      input backlog (queued): <b id="q">—</b> &nbsp;·&nbsp;
      video frames elapsed: <b id="nf">—</b>
    </div>
  </div>

  <!-- 2. directional pad -->
  <div class="padcol">
    <h1>drive</h1>
    <div class="hint" style="margin-bottom:10px">Click for one step, or hold to keep moving<br>
      (also W/A/S/D · O/L). Stays lit until the world model executes it.</div>
    <div class="pad">
      <span></span>
      <button class="btn" data-dir="forward"><span class="arrow">▲</span>forward<br><small>W</small></button>
      <span></span>
      <button class="btn" data-dir="left"><span class="arrow">◀</span>left<br><small>A</small></button>
      <span></span>
      <button class="btn" data-dir="right"><span class="arrow">▶</span>right<br><small>D</small></button>
      <span></span>
      <button class="btn" data-dir="back"><span class="arrow">▼</span>back<br><small>S</small></button>
      <span></span>
    </div>
    <div class="zpad">
      <button class="btn" data-dir="up">up&nbsp;(+z)<br><small>O</small></button>
      <button class="btn" data-dir="down">down&nbsp;(−z)<br><small>L</small></button>
    </div>
  </div>

  <!-- 3. state / stride / legend -->
  <div class="panel">
    <h1>AR world model — live control</h1>
    <div class="hint">Cameras are stacked top-to-bottom.<br>episode <span id="ep">—</span></div>

    <h2>end-effector state (normalized)</h2>
    <table id="act"></table>

    <h2>stride / block</h2>
    <div><span id="stepval"></span><br><input id="step" type="range" min="0.01" max="0.25" step="0.01"></div>

    <h2>initialization (seed episode)</h2>
    <div class="hint" style="margin:0 0 6px">Pick a clip to (re)prime the world from, then reseed -- or roll a random one.</div>
    <div><select id="eps"></select> <button id="reseed">reseed</button> <button id="randseed">🎲 random</button></div>

    <div id="initbench" style="display:none">
      <h2>initialization (benchmark suite)</h2>
      <div class="hint" style="margin:0 0 6px">Prime from a scenegen still init (single frame repeated across the history block).</div>
      <div><select id="inits"></select> <button id="reseedinit">reseed</button> <button id="randinit">🎲 random</button></div>
    </div>

    <h2>keys</h2>
    <table class="legend"><tbody>
      <tr><td><span class="kcap">W</span>/<span class="kcap">S</span></td><td>forward / back (±x)</td></tr>
      <tr><td><span class="kcap">A</span>/<span class="kcap">D</span></td><td>left / right (±y)</td></tr>
      <tr><td><span class="kcap">O</span>/<span class="kcap">L</span></td><td>up / down (±z)</td></tr>
      <tr><td><span class="kcap">K</span>/<span class="kcap">J</span></td><td>gripper open / close</td></tr>
    </tbody></table>
  </div>

</div>
<script>
const AXES=["x","y","z","rx","ry","rz","grip"];
const KEYMAP={w:"forward",s:"back",a:"left",d:"right",o:"up",l:"down",k:"grip_open",j:"grip_close"};
// directions just clicked, bridging the gap until /state reports them queued
let optimistic=new Set();
let pressed=new Set();
// directions whose button/key is currently held down -> auto re-armed each poll
let held=new Set();
let mouseDir=null;
function press(dir){if(!dir)return;optimistic.add(dir);paint();
  fetch("/press",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({dir})});}
function paint(){const show=new Set(pressed);optimistic.forEach(d=>show.add(d));held.forEach(d=>show.add(d));
  document.querySelectorAll(".btn").forEach(el=>el.classList.toggle("pressed",show.has(el.dataset.dir)));}
function startHold(dir){if(!dir)return;held.add(dir);press(dir);}      // one nudge now, re-armed while held
function stopHold(dir){held.delete(dir);paint();}
document.querySelectorAll(".btn").forEach(el=>{
  el.addEventListener("mousedown",e=>{e.preventDefault();mouseDir=el.dataset.dir;startHold(mouseDir);});});
addEventListener("mouseup",()=>{if(mouseDir){stopHold(mouseDir);mouseDir=null;}});
addEventListener("keydown",e=>{if(e.repeat)return;const k=e.key.toLowerCase();
  if(KEYMAP[k]){e.preventDefault();startHold(KEYMAP[k]);}});
addEventListener("keyup",e=>{const k=e.key.toLowerCase();
  if(KEYMAP[k]){e.preventDefault();stopHold(KEYMAP[k]);}});
const stepEl=document.getElementById("step");
stepEl.addEventListener("input",()=>{document.getElementById("stepval").textContent=stepEl.value;
  fetch("/config",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({step:parseFloat(stepEl.value)})});});
function doSeed(ep,source){fetch("/seed",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({episode:ep,source:source||"droid"})});}
document.getElementById("reseed").onclick=()=>doSeed(document.getElementById("eps").value,"droid");
document.getElementById("randseed").onclick=()=>{const s=document.getElementById("eps");
  if(!s.options.length)return; s.selectedIndex=Math.floor(Math.random()*s.options.length);
  doSeed(s.value,"droid");};
fetch("/episodes").then(r=>r.json()).then(d=>{const s=document.getElementById("eps");
  d.episodes.forEach(e=>{const o=document.createElement("option");o.value=o.textContent=e;
  if(e===d.current)o.selected=true;s.appendChild(o);});});
// second source: scenegen benchmark inits (shown only if --benchmark-root has any)
document.getElementById("reseedinit").onclick=()=>doSeed(document.getElementById("inits").value,"benchmark");
document.getElementById("randinit").onclick=()=>{const s=document.getElementById("inits");
  if(!s.options.length)return; s.selectedIndex=Math.floor(Math.random()*s.options.length);
  doSeed(s.value,"benchmark");};
fetch("/inits").then(r=>r.json()).then(d=>{const s=document.getElementById("inits");
  if(!d.inits||!d.inits.length)return;
  d.inits.forEach(e=>{const o=document.createElement("option");o.value=o.textContent=e;s.appendChild(o);});
  document.getElementById("initbench").style.display="";});
function poll(){fetch("/state").then(r=>r.json()).then(d=>{
  // server is the source of truth for which directions are still pending/executing
  pressed=new Set(d.active||[]);optimistic.clear();
  // hold-to-repeat: re-arm a held direction once its previous nudge has executed
  // (only when not already queued/in-flight, so the queue never piles up)
  held.forEach(dir=>{if(!pressed.has(dir))press(dir);});
  paint();
  if(d.action){const t=document.getElementById("act");t.innerHTML="";
    d.action.forEach((v,i)=>{const tr=document.createElement("tr");
      const f=(v+1)/2*100;
      tr.innerHTML=`<td>${AXES[i]}</td><td>${v.toFixed(3)}</td>`+
        `<td><span class="bar"><i style="left:${f}%"></i></span></td>`;
      t.appendChild(tr);});}
  if(d.step!==undefined){stepEl.value=d.step;document.getElementById("stepval").textContent=d.step;}
  document.getElementById("blk").textContent=d.block_s??"—";
  document.getElementById("ep").textContent=d.episode??"—";
  document.getElementById("fwd").textContent=(d.fwd_ms??"—");
  // --- inference-status badge + backlog/frames telemetry ---
  const q=d.queued||0, st=d.status||"—";
  const qel=document.getElementById("q");
  qel.textContent=q; qel.className=(q>=6)?"bad":(q>=1)?"warn":"";
  document.getElementById("nf").textContent=(d.frames??"—");
  const sb=document.getElementById("status");
  if(st==="loading"||st==="seeding"){sb.className="statusbadge wait";
    sb.textContent="⏳ "+st.toUpperCase()+" — wait…";}
  else if(st==="computing"||q>0){sb.className="statusbadge busy";
    sb.textContent="● INFERENCE RUNNING — "+(q>0?(q+" queued"):((d.computing_for??0)+"s"))+" — wait";}
  else{sb.className="statusbadge ready";sb.textContent="● READY — drive";}
});}
// Poll the HUD sparingly: every /state (and /ping below) is handled on an HTTP
// thread that contends for the GIL with the single generator thread mid-block,
// inflating per-block latency and its variance. 350ms is still responsive for a
// status readout but cuts the per-block GIL grabs during teleop.
setInterval(poll,350);poll();
// network (tunnel) RTT via a cheap /ping, averaged over recent samples
let _rtts=[];
function ping(){const t0=performance.now();
  fetch("/ping",{cache:"no-store"}).then(r=>r.json()).then(()=>{
    const dt=performance.now()-t0; _rtts.push(dt); if(_rtts.length>20)_rtts.shift();
    const avg=_rtts.reduce((a,b)=>a+b,0)/_rtts.length;
    document.getElementById("net").textContent=avg.toFixed(0);}).catch(()=>{});}
setInterval(ping,1000);ping();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="ARWMArgs config (configs/training/ar_*.py).")
    p.add_argument("--checkpoint", default=None, help="Trained student .pt (ARWorldModel state_dict).")
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers", help="Wan VAE for decoding.")
    p.add_argument("--decode-device", default=None,
                   help="Run the VAE decode on a second device (e.g. 'cuda:1'), overlapped with the "
                        "next block's generation. Hides ~the decode cost; adds a one-block display lag.")
    p.add_argument("--latent-root", default=None, help="Override cfg.latent_root (for seed episodes).")
    p.add_argument("--split", default="val", help="Dataset split to pull seed episodes from.")
    p.add_argument("--seed-episode", default=None, help="Episode id to prime with (default: first).")
    p.add_argument("--benchmark-root", default="assets/teleop_inits",
                   help="Scenegen initialization suite (dir of init_*/ subdirs, each with per-view "
                        "PNGs + initialization.yaml). Exposed as a second 'benchmark suite' dropdown; "
                        "each init primes the world by encoding its still and repeating that single "
                        "latent across the history block. This is the latent-free seed path -- a couple "
                        "of example inits ship in assets/teleop_inits/ (the default), so teleop works "
                        "from a fresh clone with no preprocessed latents. Empty/missing -> dropdown hidden.")
    p.add_argument("--history-blocks", type=int, default=-1,
                   help="GT blocks used to prime the world ('first frame'). -1 (default) uses the "
                        "model's trained cfg.num_history_blocks; priming with fewer is "
                        "out-of-distribution and yields a blurry/bad initial image.")
    p.add_argument("--denoising-steps", type=int, default=0,
                   help="Denoising steps per block (0 -> cfg.preview_denoising_steps, ~32). Lower = faster.")
    p.add_argument("--max-kv-blocks", type=int, default=-1,
                   help="Sliding-window KV cap (-1 -> cfg.max_kv_blocks / unbounded). Bound for long sessions. "
                        "REQUIRED (finite) with --static-cache / --compile.")
    # --- inference speedups (see claude_notes/speedup.md). Stack for the fastest path:
    #     --bf16 --decode-device cuda:1 --static-cache --compile --max-kv-blocks N
    p.add_argument("--bf16", action="store_true",
                   help="Pure-bf16 deploy: cast params + decode VAE to bf16 and drop autocast "
                        "(vs the fp32-master + bf16-autocast default). ~1.3x.")
    p.add_argument("--decode-context", type=int, default=2,
                   help="Latent frames of left-context kept for temporally-continuous decoding "
                        "(rolling decode window = decode_context + frames_per_block). Smaller = faster "
                        "decode; with --decode-device the decode is hidden so this barely matters.")
    p.add_argument("--static-cache", action="store_true",
                   help="Fixed-shape ring-buffer KV cache (requires a finite --max-kv-blocks). "
                        "Prerequisite for --compile; correctness-neutral on its own.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(mode='reduce-overhead') the transformer block loop (~1.41x on "
                        "the forward). Implies --static-cache and a finite --max-kv-blocks. First few "
                        "blocks are slow (compilation) then steady-state. Best with a DISTILLED checkpoint.")
    p.add_argument("--measure-latency", action="store_true",
                   help="Log per-block action->pixels latency (queue wait / step / forward / "
                        "action->emit) and periodic p50/p90/p99 summaries to stdout; also expose "
                        "a2e_ms in /state. Use to A/B sync-vs-async and step counts with numbers.")
    p.add_argument("--step-size", type=float, default=0.08, help="Normalized action nudge per held key per block.")
    p.add_argument("--model-hz", type=float, default=5.0,
                   help="Model's native action rate (Hz), informational. The teleop stream is "
                        "coalesced to a latest-pose queue and fed one pose per AR block; the "
                        "freshest pose is always used (no time-rate throttle -> lower latency).")
    p.add_argument("--roll-when-idle", action="store_true",
                   help="Keep dreaming frames even when no key is held (default: pause while idle).")
    p.add_argument("--distilled", action="store_true",
                   help="Sample with the few-step DEPLOYMENT schedule (cfg.denoising_step_list, "
                        "e.g. 4 steps). Use this with a DISTILLED checkpoint; on a non-distilled "
                        "backbone it produces a blurry colour-wash (omit it for the preview schedule).")
    p.add_argument("--continuous", action="store_true",
                   help="Always roll forward at the current pose (for SpaceMouse teleop via "
                        "POST /action). Like --roll-when-idle; the world never pauses while idle.")
    p.add_argument("--record-dir", default=None,
                   help="If set, record the teleop trajectory (actions_norm/raw.npy + teleop.mp4 "
                        "+ meta.json) to this directory on exit (Ctrl-C).")
    p.add_argument("--record-max-frames", type=int, default=6000,
                   help="Cap on recorded RGB frames (~12 min @ 8 fps). Actions are always logged.")
    p.add_argument("--rotate-wrist", action="store_true",
                   help="DISPLAY-ONLY: rotate the wrist-view band (bottom of the height-stack) 180 deg "
                        "in the live MJPEG stream. Model latents/actions and the recorded video are "
                        "unchanged -- purely makes the stream intuitive for an inverted camera mount.")
    p.add_argument("--fps", type=int, default=8, help="MJPEG playback fps (training previews used 8).")
    p.add_argument("--jpeg-quality", type=int, default=88)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    engine = Engine(args)
    gen = threading.Thread(target=engine.run, name="generator", daemon=True)
    gen.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    host = os.uname().nodename
    print(f"[interactive] serving on http://{args.host}:{args.port}  (node: {host})", flush=True)
    print(f"[interactive] tunnel from your laptop:  ssh -N -L {args.port}:{host}:{args.port} <you>@<login>", flush=True)
    print(f"[interactive] then open  http://localhost:{args.port}", flush=True)
    if args.continuous:
        print(f"[interactive] teleop mode: drive POST /action; "
              f"run scripts/spacemouse_client.py on your laptop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        httpd.shutdown()
        engine._meter.summary()          # final action->pixels percentiles (--measure-latency)
        if engine.recorder is not None:
            engine.recorder.save()


if __name__ == "__main__":
    main()
