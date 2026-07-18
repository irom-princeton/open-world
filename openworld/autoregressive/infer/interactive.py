"""Interactive (live-controlled) autoregressive rollout.

``replay.py`` feeds a *recorded* action sequence open-loop. This module instead
lets an external agent (a keyboard, a policy, ...) supply ONE action at a time and
generates ONE latent block per call, keeping the KV-cache warm across calls so the
cost per step is bounded (it does not re-prime the whole history every step the way
``model.rollout`` does).

It mirrors ``distill.self_forcing.generate_rollout``'s block loop exactly --
``model.forward_cached`` with the full action-sequence ``cond`` and the running
``start_frame`` -- so the per-block action slicing (``cross_attn_aligned`` /
``adaln``), RoPE offset, and commit semantics are identical to the validated
preview/replay path. The only difference is that the action history grows live
instead of being known up front.

The studentinit (L2a) mid-training checkpoints are NOT distilled, so they must be
sampled with a many-step uniform schedule (``build_preview_scheduler``), the same
one ``train_self_forcing._SamplePreviewer`` uses for mid-training previews; the
4-step deployment list would yield a blurry colour-wash.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..data.decode import decode_stacked
from ..distill.scheduler import FlowMatchScheduler


class AsyncWindowDecoder:
    """Decode the rolling latent window on a *second* device, overlapped with the
    next block's generation.

    The VAE decode is ~half the per-step wall-clock (see the latency profile) and
    is causally downstream of generation -- ``decode(block_t)`` needs block t's
    latents, so it can only overlap with the *generation of block t+1*. This worker
    runs the decode on its own device/thread, giving a one-block software pipeline:
    ``submit(window_t)`` starts decoding while the caller generates block t+1, and
    the matching frames come back from the *next* ``take()``. Net effect: the
    generator thread (GPU0) no longer blocks on the VAE (GPU1).

    ``submit`` carries an optional ``meta`` (e.g. the action that produced the
    block) so a consumer can keep frames aligned with their action despite the lag.
    """

    def __init__(self, decoder, *, num_cams: int, emit_latent_frames: int):
        self.decoder = decoder                       # VaeLatentDecoder on the decode device
        self.num_cams = num_cams
        self.emit = emit_latent_frames
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._meta = None

    def submit(self, window: torch.Tensor, meta=None) -> None:
        # Copy to the decode device on the *calling* (generator) thread so the peer
        # copy is ordered after the generation kernels that produced ``window``;
        # the worker then decodes purely on its own device, free of GPU0.
        w = window.detach().to(self.decoder.device)
        self._future, self._meta = self._exec.submit(self._run, w), meta

    def _run(self, window: torch.Tensor) -> np.ndarray:
        rgb = decode_stacked(self.decoder, window, self.num_cams)   # [T, V*H, W, 3]
        Wl = window.shape[0]
        return rgb if self.emit >= Wl else rgb[-(self.emit * 4):]

    def take(self):
        """Block for the previously submitted decode; ``(None, None)`` if none pending."""
        if self._future is None:
            return None, None
        f, m = self._future, self._meta
        self._future = self._meta = None
        return f.result(), m

    def clear(self) -> None:
        """Drop any pending decode (used on reseed)."""
        if self._future is not None:
            try:
                self._future.result()
            except Exception:
                pass
            self._future = self._meta = None


def build_preview_scheduler(n_steps: int, num_train_timestep: int = 1000) -> FlowMatchScheduler:
    """Many-step uniform-sigma schedule for sampling a *non-distilled* backbone.

    Identical to the mid-training previewer (``train_self_forcing.py``): integer
    steps ``~ts .. ts/n`` with ``warp=False``.
    """
    n = max(2, int(n_steps))
    ts = num_train_timestep
    steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))
    return FlowMatchScheduler(steps, num_train_timestep=ts, warp=False)


class InteractiveRoller:
    """Stateful, persistent-KV autoregressive roller for live control.

    Call :meth:`reset` once with the ground-truth history (the "first frame(s)")
    and a seed action, then :meth:`step` repeatedly -- each call appends the given
    action to the running history and generates + decodes exactly one latent block.

    All GPU work must happen on a single thread (one CUDA stream); ``step`` /
    ``reset`` are guarded by a lock so a control thread can't race the generator.
    """

    def __init__(
        self,
        model,
        decoder,
        *,
        num_cams: int,
        scheduler: FlowMatchScheduler,
        device: torch.device,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        max_kv_blocks: int | None = None,
        decode_context: int = 2,
        async_decoder: "AsyncWindowDecoder | None" = None,
        static_cache: bool = False,
    ):
        self.model = model
        self.decoder = decoder
        # Optional second-device decoder: when set, ``step`` overlaps the VAE decode
        # with the next block's generation (one-block pipeline latency; see
        # AsyncWindowDecoder). ``None`` -> the original synchronous decode.
        self._adec = async_decoder
        self.num_cams = num_cams
        self.sched = scheduler
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.fpb = model.cfg.frames_per_block
        self.in_channels = model.cfg.in_channels
        self.pdt = next(model.parameters()).dtype           # param/compute dtype (fp32 master)
        self.max_kv_blocks = (
            max_kv_blocks if max_kv_blocks is not None else model.cfg.max_kv_blocks
        )
        # latent frames of left-context kept for temporally-continuous decoding
        self.decode_context = decode_context
        # fixed-shape ring-buffer KV cache (requires a finite max_kv_blocks); its
        # registered-buffer K/V enable CUDA-graph capture of the per-block forward
        # under torch.compile(mode="reduce-overhead"). Default: the growing cache.
        self.static_cache = static_cache
        self._lock = threading.Lock()
        self._ready = False
        self.last_emitted_action = None              # action paired with the last async-emitted frames
        self.last_fwd_s = 0.0                        # true model-forward GPU time of the last block (s)
        self._fwd_ev = None                          # (start,end) CUDA events, read non-blocking next block

    # -- helpers ---------------------------------------------------------
    def _autocast(self):
        if self.device.type == "cuda" and self.autocast_dtype is not None:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _encode_cond(self) -> torch.Tensor:
        """Full running action sequence -> ``[1, N, cross]`` condition.

        The backbone slices this to the current block by ``start_frame`` (aligned /
        adaln) or attends to all of it (global modes), so it must stay indexed by
        absolute latent-frame position -- hence we keep the whole history.
        """
        acts = np.stack(self._actions, axis=0)[None]        # [1, N, A]
        # match the param/compute dtype: the action encoder's Linear weights carry
        # ``pdt`` (bf16 in a pure-bf16 deploy, fp32 under fp32-master+autocast), and
        # there's no autocast to coerce a fp32 input in the bf16 case.
        a = torch.from_numpy(acts).to(self.device, self.pdt)
        return self.model.encode_cond(a, cfg_drop=False)

    @torch.no_grad()
    def _decode(self, emit_latent_frames: int) -> np.ndarray:
        """Decode the rolling latent window, return the newest RGB frames.

        ``[T, V*H, W, 3] uint8``. Decoding a window (context + new) rather than the
        bare new block gives the Wan VAE's temporal conv left-context so blocks
        stitch without a seam; we then emit only the newest ``emit_latent_frames*4``
        RGB frames (Wan's 4x temporal upsample).
        """
        window = torch.stack(list(self._win), dim=0)        # [Wl, C, V*h, w]
        rgb = decode_stacked(self.decoder, window, self.num_cams)  # [T, V*H, W, 3]
        Wl = window.shape[0]
        if emit_latent_frames >= Wl:
            return rgb
        return rgb[-(emit_latent_frames * 4):]

    # -- public API ------------------------------------------------------
    @torch.no_grad()
    def reset(self, history_latents: torch.Tensor, seed_actions_norm: np.ndarray) -> np.ndarray:
        """Prime the cache with the GT history; return its decoded RGB frames.

        ``history_latents``: ``[hist_frames, C, V*h, w]`` normalized Wan latents.
        ``seed_actions_norm``: ``[hist_frames, A]`` normalized actions (1:1 w/ frames).
        """
        with self._lock:
            if self._adec is not None:
                self._adec.clear()                       # drop any in-flight decode from a prior episode
            hist = history_latents.to(self.device, self.pdt)
            hf = hist.shape[0]
            if hf % self.fpb != 0:
                raise ValueError(f"history frames {hf} not divisible by frames_per_block {self.fpb}")
            self.last_action = seed_actions_norm[-1].astype(np.float32).copy()
            self._actions = [seed_actions_norm[i].astype(np.float32) for i in range(hf)]
            self._win = deque(maxlen=self.decode_context + self.fpb)
            self.kv = self.model.make_kv_cache(max_blocks=self.max_kv_blocks, static=self.static_cache)
            self.start = 0
            zero_t = self.sched.to_timestep(torch.zeros(1, device=self.device))
            with self._autocast():
                cond = self._encode_cond()
                for i in range(0, hf, self.fpb):
                    blk = hist[i:i + self.fpb].unsqueeze(0)          # [1, fpb, C, V*h, w]
                    self.model.forward_cached(
                        blk, zero_t, cond, kv_cache=self.kv,
                        start_frame=self.start, commit=True,
                    )
                    self.start += self.fpb
            for f in range(hf):
                self._win.append(hist[f])
            self._ready = True
            return self._decode(emit_latent_frames=len(self._win))

    @torch.no_grad()
    def step(self, action_norm: np.ndarray) -> np.ndarray:
        """Generate + decode ONE block conditioned on ``action_norm`` ``[A]``.

        Returns the newest RGB frames ``[fpb*4, V*H, W, 3] uint8``.
        """
        with self._lock:
            if not self._ready:
                raise RuntimeError("call reset() before step()")
            # Read the prior block's forward timing now that it has surely
            # finished -- non-blocking (query), so we never stall the stream.
            if self._fwd_ev is not None and self._fwd_ev[1].query():
                self.last_fwd_s = self._fwd_ev[0].elapsed_time(self._fwd_ev[1]) / 1000.0
                self._fwd_ev = None
            a = action_norm.astype(np.float32)
            for _ in range(self.fpb):                                # 1:1 action<->latent-frame
                self._actions.append(a)
            block_shape = (1, self.fpb, self.in_channels,
                           self._win[-1].shape[-2], self._win[-1].shape[-1])
            n = self.sched.num_steps
            _cuda = self.device.type == "cuda"
            if _cuda:
                _ev0 = torch.cuda.Event(enable_timing=True)
                _ev1 = torch.cuda.Event(enable_timing=True)
                _ev0.record()
            with self._autocast():
                cond = self._encode_cond()
                x = torch.randn(block_shape, device=self.device, dtype=self.pdt)
                x0_clean = x
                for i in range(n):
                    sigma_i = self.sched.sigmas[i].to(self.device).expand(1)
                    t_i = self.sched.to_timestep(sigma_i)
                    v = self.model.forward_cached(
                        x, t_i, cond, kv_cache=self.kv,
                        start_frame=self.start, commit=False,
                    )
                    x0_clean = self.sched.x0_from_velocity(x, v, sigma_i)
                    if i < n - 1:
                        sigma_next = self.sched.sigmas[i + 1].to(self.device).expand(1)
                        x = self.sched.add_noise(x0_clean, torch.randn_like(x0_clean), sigma_next)
                # commit the clean block as context for subsequent blocks
                zero_t = self.sched.to_timestep(torch.zeros(1, device=self.device))
                self.model.forward_cached(
                    x0_clean, zero_t, cond, kv_cache=self.kv,
                    start_frame=self.start, commit=True,
                )
            if _cuda:
                # TRUE model-forward latency: GPU span of encode + 4 denoise +
                # commit on cuda:0, via CUDA events. Read NEXT block (non-blocking)
                # so timing never stalls the pipeline. Distinct from block_s, which
                # also includes scheduler/host overhead, GIL contention with the
                # HTTP threads, and waiting on the overlapped decode.
                _ev1.record(); self._fwd_ev = (_ev0, _ev1)
            self.start += self.fpb
            block = x0_clean[0]                                       # [fpb, C, V*h, w]
            for f in range(self.fpb):
                self._win.append(block[f])
            if self._adec is not None:
                # Pipeline: hand this block's window to the decode device and return
                # the PREVIOUS block's frames (decoded during this step's generation).
                # ``last_emitted_action`` is the action that produced those frames, so
                # a recorder can stay aligned despite the one-block lag.
                prev_rgb, prev_action = self._adec.take()
                self._adec.submit(torch.stack(list(self._win), dim=0), meta=a)
                self.last_emitted_action = prev_action
                return prev_rgb
            return self._decode(emit_latent_frames=self.fpb)

    @torch.no_grad()
    def flush(self):
        """Drain the final overlapped decode (async mode only); returns ``(rgb, action)``.

        After the last :meth:`step`, one block remains in flight on the decode
        device -- call this once to emit it. ``(None, None)`` in synchronous mode."""
        with self._lock:
            if self._adec is None:
                return None, None
            rgb, action = self._adec.take()
            self.last_emitted_action = action
            return rgb, action
