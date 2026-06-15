"""Self-forcing / DMD training entry point for the AR world model.

Mirrors ``openworld.training.world_model.train_wm`` conventions: a ``--config``
points at a Python file exposing ``get_args() -> ARWMArgs``. Run with accelerate
for multi-GPU.

    accelerate launch -m openworld.autoregressive.train_self_forcing \
        --config configs/training/ar_wan_1_3b.py

A ``--smoke`` mode runs a few steps on a tiny random-init backbone with synthetic
latents/actions — no weights or dataset needed — to validate the pipeline end to
end (used by the test suite and for CI):

    python -m openworld.autoregressive.train_self_forcing --smoke

Data note: Wan/Cosmos consume 16-channel latents from their own VAE; the existing
LIBERO latents are 4-channel SVD-VAE. Real runs need the dataset re-encoded with
the backbone VAE (see docs/AUTOREGRESSIVE.md). The real-data branch below reuses
``LiberoLatentDataset`` and is the integration point for that.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ARWMArgs
from .distill.scheduler import FlowMatchScheduler
from .distill.self_forcing import SelfForcingTrainer
from .model import build_training_stack

_WARMUP_STEPS = 10        # ignore the first few steps (CUDA/compile warmup) in throughput


class _CheckpointKeeper:
    """Step-based checkpoint retention for crash-safe, space-bounded training.

    Two cadences (the step counts are sized per config to ~1h / ~8h of wall-clock
    using the measured iteration speed):

    * **rolling** every ``rolling_steps`` -- a throwaway safety net. The new file
      is written first; only once it is on disk is the *previous* rolling file
      deleted. So a crash mid-save never leaves you with zero checkpoints.
    * **permanent** every ``permanent_steps`` -- kept forever. A permanent save
      also reclaims the pending rolling file, so every intermediate checkpoint
      between two permanents is removed and at most one rolling file + the
      permanents occupy disk.

    Permanent files use the canonical ``checkpoint-{step}.pt`` name (clean for
    downstream use); rolling files carry a ``-rolling`` suffix so they are easy
    to spot and never collide with a permanent at the same step.
    """

    def __init__(self, out_dir, rolling_steps, permanent_steps):
        self.out_dir = Path(out_dir)
        self.rolling_steps = rolling_steps
        self.permanent_steps = permanent_steps
        self._pending_rolling = None        # deletable rolling checkpoint, if any

    def due(self, step):
        rolling = self.rolling_steps > 0 and step % self.rolling_steps == 0
        permanent = self.permanent_steps > 0 and step % self.permanent_steps == 0
        return rolling or permanent

    def save(self, state_dict, step, *, force_permanent=False):
        permanent = force_permanent or (
            self.permanent_steps > 0 and step % self.permanent_steps == 0
        )
        name = f"checkpoint-{step}.pt" if permanent else f"checkpoint-{step}-rolling.pt"
        out = self.out_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, out)
        # delete the prior rolling file only after the new checkpoint is on disk
        if out.exists() and self._pending_rolling and self._pending_rolling != out \
                and self._pending_rolling.exists():
            self._pending_rolling.unlink()
        self._pending_rolling = None if permanent else out
        return out, permanent


def _fsdp_shard_models(models, *, enabled: bool) -> bool:
    """Shard each model (transformer blocks first, then the root module) across all
    ranks with FSDP2 ``fully_shard``. Params/grads/optimizer stay in their built
    dtype (fp32 master) but live sharded -> per-GPU memory drops ~world_size x,
    which is what lets the 3-model self-forcing stack fit. Compute precision is the
    backbone's bf16 autocast, untouched here. No-op (returns False) on a single
    process. Must run BEFORE the optimizers are constructed.
    """
    import torch.distributed as dist
    if not (enabled and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return False
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    # IMPORTANT: this code invokes custom methods (forward_cached / forward_train /
    # encode_cond), NOT module.__call__ on the ARWorldModel root -- so a root-level
    # fully_shard's unshard hook would never fire. Shard only the modules that ARE
    # called via __call__: the transformer (+ its blocks) and the trainable action
    # conditioner. Together these cover every trainable param of each model.
    for m in models:
        tf = m.backbone.transformer if hasattr(m, "backbone") else getattr(m, "transformer", None)
        if tf is not None:
            blocks = getattr(tf, "blocks", None) or getattr(tf, "transformer_blocks", None)
            for blk in (blocks or []):
                fully_shard(blk, mesh=mesh)
            fully_shard(tf, mesh=mesh)
        cond_mod = getattr(m, "conditioner", None)       # generator only; invoked via __call__
        if cond_mod is not None and any(p.requires_grad for p in cond_mod.parameters()):
            fully_shard(cond_mod, mesh=mesh)
        # adaln action mode adds a small action->time-embedding Linear directly on
        # the backbone (outside the transformer AND the conditioner). It is invoked
        # via __call__ (``self.action_to_temb(cond)``), so shard it too -- otherwise
        # it stays a plain Tensor while every other param is a DTensor, and
        # clip_grad_norm_ over model.parameters() crashes on the mixed types.
        # ``backbone`` for the ARWorldModel generator; ``m`` itself for the bare
        # WanBackbone critic/teacher at L0.
        bb = getattr(m, "backbone", None) or m
        a2t = getattr(bb, "action_to_temb", None)
        if a2t is not None and any(p.requires_grad for p in a2t.parameters()):
            fully_shard(a2t, mesh=mesh)
    return True


def _gather_full_state_dict(model, distributed: bool):
    """Full (unsharded) cpu state dict for checkpointing. Under FSDP this is a
    COLLECTIVE -- call it on every rank, then write the file on the main process."""
    if not distributed:
        return model.state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


# ---------------------------------------------------------------------------
# Resume: a single full training-state bundle (generator + online critic + both
# AdamW optimizer states + step) so a cut run continues *exactly* where it left
# off -- not just the generator weights the inference checkpoints carry. It is
# overwritten in place each cadence, so only one (~30 GB for Wan-1.3B) file ever
# exists. FSDP-aware via torch.distributed.checkpoint get/set_state_dict.
# ---------------------------------------------------------------------------
_RESUME_NAME = "training_state.pt"


def _gather_train_state(gen, critic, opt_g, opt_c, step, distributed: bool) -> dict:
    """Collective on every rank under FSDP -> a full (cpu) resume bundle."""
    if not distributed:
        return {
            "gen": gen.state_dict(), "critic": critic.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_c": opt_c.state_dict(), "global_step": step,
        }
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    g_m, g_o = get_state_dict(gen, opt_g, options=opts)
    c_m, c_o = get_state_dict(critic, opt_c, options=opts)
    return {"gen": g_m, "critic": c_m, "opt_g": g_o, "opt_c": c_o, "global_step": step}


def _save_resume_atomic(state: dict, out_dir) -> Path:
    """Write the resume bundle to a temp file then os.replace -> never leaves a
    half-written file if the job dies mid-save (rank 0 only)."""
    out = Path(out_dir) / _RESUME_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, out)            # atomic on the same filesystem
    return out


def _restore_train_state(path, gen, critic, opt_g, opt_c, distributed: bool) -> int:
    """Load a resume bundle into the (already FSDP-sharded) models + optimizers on
    every rank. Returns the global step to continue from."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not distributed:
        gen.load_state_dict(state["gen"]); critic.load_state_dict(state["critic"])
        opt_g.load_state_dict(state["opt_g"]); opt_c.load_state_dict(state["opt_c"])
        return int(state["global_step"])
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_state_dict(gen, opt_g, model_state_dict=state["gen"],
                   optim_state_dict=state["opt_g"], options=opts)
    set_state_dict(critic, opt_c, model_state_dict=state["critic"],
                   optim_state_dict=state["opt_c"], options=opts)
    return int(state["global_step"])


def _backbone_state_from_ckpt(sd: dict) -> dict:
    """Pure-backbone weights from a checkpoint that may be a full ARWorldModel
    state dict. The mid-training stages save ARWorldModel checkpoints (keys
    ``backbone.*`` + ``conditioner.*``), but the L0 teacher/critic are bare
    backbones (keys ``transformer.*``). Strip the ``backbone.`` prefix and drop
    the conditioner; pass through unchanged if the dict is already bare."""
    if any(k.startswith("backbone.") for k in sd):
        return {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    return sd


def _load_config(config_arg: str | None) -> ARWMArgs:
    if config_arg is None:
        return ARWMArgs()
    path = Path(config_arg)
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Config not found: {config_arg}")
    spec = importlib.util.spec_from_file_location("user_ar_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_args"):
        raise AttributeError(f"{path} must define get_args() -> ARWMArgs")
    return mod.get_args()


def _latent_block_shape(cfg, batch: int) -> tuple:
    return (batch, cfg.frames_per_block, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)


def _side_by_side(gt: np.ndarray, pred: np.ndarray, gap: int = 4) -> np.ndarray:
    """Two ``[T, H, W, 3]`` uint8 clips -> one ``[T, H, W_gt+gap+W_pred, 3]`` clip."""
    T = min(gt.shape[0], pred.shape[0])
    gt, pred = gt[:T], pred[:T]
    sep = np.full((T, gt.shape[1], gap, 3), 128, dtype=np.uint8)
    return np.concatenate([gt, sep, pred], axis=2)


class _SamplePreviewer:
    """Qualitative wandb video previews via open-loop AR replay on ``val`` episodes.

    The student is FSDP-sharded, so the rollout forward MUST run on *every* rank
    (the unshard all-gathers are collective). Episode selection is seeded by the
    step so all ranks pick the same episodes and their collectives line up; only
    the main rank holds the (un-sharded) VAE and logs the decoded videos. Decoding
    is lazy/rank-0-only so non-main ranks never load the VAE.
    """

    def __init__(self, args: ARWMArgs, *, is_main: bool):
        self.args = args
        self.is_main = is_main
        self.enabled = bool(args.log_samples)
        self._decoder = None
        self._p01 = self._p99 = None
        self._ep_ids: list[str] = []
        # Mid-training backbones are not distilled, so previewing them at the
        # few-step ``denoising_step_list`` yields garbage. Sample with a many-step
        # uniform schedule instead; the self_forcing stage keeps its real list
        # (scheduler=None -> model.rollout builds it from cfg.denoising_step_list).
        self._preview_sched = None
        if getattr(args, "stage", "self_forcing") != "self_forcing":
            n = max(2, int(args.preview_denoising_steps))
            ts = args.num_train_timestep
            steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))  # ~ts..ts/n
            # warp=False -> uniform sigma grid reaching ~ts/n near the data end
            # (faithful to the uniform-sigma mid-training; the shift only matters
            # for the few-step distilled schedule).
            self._preview_sched = FlowMatchScheduler(
                steps, num_train_timestep=ts, warp=False)
        if not self.enabled:
            return
        # action stats are needed on ALL ranks (they feed the rollout conditioning)
        from .infer import load_action_stats
        try:
            self._p01, self._p99 = load_action_stats(args.latent_root)
        except FileNotFoundError:
            self.enabled = False
            return
        val_dir = os.path.join(args.latent_root, "val")
        try:
            self._ep_ids = sorted(f[:-3] for f in os.listdir(val_dir) if f.endswith(".pt"))
        except FileNotFoundError:
            self._ep_ids = []
        if not self._ep_ids:
            self.enabled = False

    def _ensure_decoder(self):
        if self._decoder is not None:
            return
        from diffusers import AutoencoderKLWan

        from .data.decode import VaeLatentDecoder
        vae = AutoencoderKLWan.from_pretrained(
            self.args.vae_dir, subfolder="vae", torch_dtype=torch.float32)
        self._decoder = VaeLatentDecoder(vae, device="cuda", dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, model, step: int, *, accelerator, logger) -> None:
        if not self.enabled:
            return
        from .data.decode import decode_stacked
        from .infer import load_full_episode, normalize_actions, replay_episode_latents
        a = self.args
        # rank-synchronized random episode pick (same seed -> matched FSDP collectives)
        rng = random.Random(a.seed * 1_000_003 + step)
        picks = rng.sample(self._ep_ids, min(a.num_sample_videos, len(self._ep_ids)))
        max_blocks = a.sample_max_blocks or None
        hist_min = (a.sample_history_blocks + 1) * a.frames_per_block

        if self.is_main:
            self._ensure_decoder()
            out_dir = Path(a.output_dir) / "samples" / f"step-{step}"
            out_dir.mkdir(parents=True, exist_ok=True)
        was_training = model.training
        model.eval()
        logged = {}
        try:
            for ep_id in picks:
                latent_gt, action_raw, text = load_full_episode(
                    a.latent_root, "val", ep_id, a.num_cams)
                # identical guard on every rank -> consistent control flow / collectives
                if latent_gt.shape[0] < hist_min:
                    continue
                action_norm = normalize_actions(action_raw, self._p01, self._p99)
                gt_lat, pred_lat, _ = replay_episode_latents(
                    model, latent_gt, action_norm,
                    frames_per_block=a.frames_per_block,
                    num_history_blocks=a.sample_history_blocks,
                    in_channels=a.in_channels, device=accelerator.device,
                    dtype=a.dtype, max_blocks=max_blocks,
                    scheduler=self._preview_sched,
                )
                if not self.is_main:
                    continue
                gt_vid = decode_stacked(self._decoder, gt_lat, a.num_cams)     # [T, V*H, W, 3]
                pred_vid = decode_stacked(self._decoder, pred_lat, a.num_cams)
                vid = _side_by_side(gt_vid, pred_vid)                          # [T, H, W2, 3]
                # write the mp4 to disk (mediapy needs no moviepy) AND log the file to
                # wandb -- passing a path avoids wandb's raw-array moviepy dependency.
                import mediapy
                import wandb
                path = out_dir / f"{ep_id}.mp4"
                mediapy.write_video(str(path), vid, fps=a.sample_video_fps, codec="h264",
                                    ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")
                logged[f"samples/{ep_id}"] = wandb.Video(
                    str(path), caption=f"step {step} | GT | PRED | {text}"[:200])
        except Exception as e:                          # never let a preview kill training
            if self.is_main:
                logger.warning(f"sample preview failed at step {step}: {e!r}")
        finally:
            if was_training:
                model.train()
        if self.is_main and logged:
            accelerator.log(logged, step=step)
            logger.info(f"logged {len(logged)} sample video(s) at step {step} -> {out_dir}")


# ---------------------------------------------------------------------------
# Smoke mode: synthetic, weightless, CPU-fast. Validates the full train loop.
# ---------------------------------------------------------------------------
def run_smoke(steps: int = 1, backbone: str = "dummy") -> dict:
    torch.set_num_threads(2)  # keep the weightless CPU smoke light for CI / shared nodes
    cfg = ARWMArgs(
        backbone=backbone, random_init_backbone=True, multiview_layout="height_stack",
        num_cams=1, frames_per_block=2, rollout_blocks=2, num_history_blocks=1,
        denoising_step_list=(1000, 500), critic_steps_per_gen_step=1,
        width=32, height=32, action_dim=7, dtype=torch.float32,
    )
    gen, critic, teacher = build_training_stack(cfg)
    sched = FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)
    trainer = SelfForcingTrainer(
        gen, critic, teacher, sched, frames_per_block=cfg.frames_per_block,
        gen_lr=cfg.learning_rate, critic_lr=cfg.critic_learning_rate,
        critic_steps=cfg.critic_steps_per_gen_step, real_cfg=cfg.real_guidance_scale,
        dmd_lo=cfg.dmd_min_step_ratio, dmd_hi=cfg.dmd_max_step_ratio,
    )
    B = 1
    shape = _latent_block_shape(cfg, B)
    T = cfg.frames_per_block * (cfg.rollout_blocks + cfg.num_history_blocks)
    logs = {}
    for _ in range(steps):
        actions = torch.randn(B, T, cfg.action_dim)
        cond = gen.encode_cond(actions, cfg_drop=True)            # [B, T, cross]
        null = gen.null_cond_like(cond)
        logs = trainer.train_step(cond, null, num_blocks=cfg.rollout_blocks, latent_block_shape=shape)
    assert all(math.isfinite(v) for v in logs.values()), logs
    return logs


def main(args: ARWMArgs) -> None:
    from accelerate import Accelerator
    import logging

    from accelerate.logging import get_logger

    # default stdlib level is WARNING, which silently drops every logger.info below
    # (checkpoint saves, RESUMED, throughput, sample logging). Surface them.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = get_logger(__name__)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
    )
    gen, critic, teacher = build_training_stack(args)
    # Stage init: generator <- L2a student-init (full ARWorldModel), teacher +
    # critic <- L1b teacher (backbone weights only; they are bare backbones).
    if args.student_init_ckpt:
        sd = torch.load(args.student_init_ckpt, map_location="cpu", weights_only=False)
        m, u = gen.load_state_dict(sd, strict=False)
        logger.info(f"loaded generator from L2a {args.student_init_ckpt} "
                    f"(missing {len(m)}, unexpected {len(u)})")
    if args.teacher_ckpt:
        sd = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        bb = _backbone_state_from_ckpt(sd)
        mt, ut = teacher.load_state_dict(bb, strict=False)
        critic.load_state_dict(bb, strict=False)  # critic init from teacher (L1b)
        logger.info(f"loaded teacher+critic backbone from L1b {args.teacher_ckpt} "
                    f"(missing {len(mt)}, unexpected {len(ut)})")

    # Params/grads/optimizer stay in args.dtype (fp32 master for real runs); the
    # backbone's bf16 autocast (cfg.autocast_dtype, driven by mixed_precision) does
    # the heavy compute. Move to the device, then shard across ranks with FSDP2
    # BEFORE building the optimizers so they bind to the sharded params.
    gen, critic, teacher = (m.to(accelerator.device, args.dtype) for m in (gen, critic, teacher))
    distributed = _fsdp_shard_models([gen, critic, teacher], enabled=args.use_fsdp)
    if accelerator.is_main_process:
        logger.info(f"FSDP sharding: {'ON' if distributed else 'off (single process)'}")

    sched = FlowMatchScheduler(args.denoising_step_list, num_train_timestep=args.num_train_timestep,
                               warp=args.warp_denoising_step)
    trainer = SelfForcingTrainer(
        gen, critic, teacher, sched, frames_per_block=args.frames_per_block,
        gen_lr=args.learning_rate, critic_lr=args.critic_learning_rate,
        critic_steps=args.critic_steps_per_gen_step, real_cfg=args.real_guidance_scale,
        dmd_lo=args.dmd_min_step_ratio, dmd_hi=args.dmd_max_step_ratio,
        betas=args.adam_betas, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        score_whole_clip=args.dmd_score_whole_clip,
        random_exit=args.dmd_random_exit,
    )

    # --- data ---------------------------------------------------------------
    from openworld.autoregressive.data import ARLatentDataset
    train_dataset = ARLatentDataset(args, split="train")
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    loader = accelerator.prepare(loader)
    if accelerator.is_main_process:
        accelerator.init_trackers(args.wandb_project_name, config={},
                                  init_kwargs={"wandb": {"name": args.wandb_run_name}})

    hist_frames = args.num_history_blocks * args.frames_per_block
    global_step = 0
    # Auto-resume: continue exactly from the last full training-state bundle.
    resume_path = args.resume_from or (Path(args.output_dir) / _RESUME_NAME)
    if args.save_resume_state and Path(resume_path).exists():
        global_step = _restore_train_state(
            resume_path, gen, critic, trainer.opt_g, trainer.opt_c, distributed)
        logger.info(f"RESUMED from {resume_path} at step {global_step}")
    t_start = time.monotonic()
    t_warm = None
    start_step = global_step          # throughput baseline is relative to this run
    keeper = _CheckpointKeeper(args.output_dir, args.checkpointing_steps,
                              args.permanent_checkpoint_steps)
    previewer = _SamplePreviewer(args, is_main=accelerator.is_main_process)
    logger.info(f"Starting AR self-forcing training: {args.backbone}, "
                f"{args.rollout_blocks} blocks x {args.frames_per_block} frames/block; "
                f"checkpoints: rolling every {args.checkpointing_steps} steps, "
                f"permanent every {args.permanent_checkpoint_steps} steps; "
                f"resume {'on' if args.save_resume_state else 'off'}")
    while global_step < args.max_train_steps:
        for batch in loader:
            latent = batch["latent"].to(accelerator.device, args.dtype)   # [B, F, C, H, W]
            actions = batch["action"].to(accelerator.device, args.dtype)
            texts = batch.get("text")
            B = latent.shape[0]
            # clean history blocks prime the KV-cache.
            hist = latent[:, :hist_frames]
            history_blocks = [hist[:, i:i + args.frames_per_block]
                              for i in range(0, hist_frames, args.frames_per_block)] or None
            cond = gen.encode_cond(actions, texts=texts, cfg_drop=True)
            null = gen.null_cond_like(cond)
            shape = _latent_block_shape(args, B)
            logs = trainer.train_step(
                cond, null, num_blocks=args.rollout_blocks,
                latent_block_shape=shape, history_blocks=history_blocks,
            )
            global_step += 1
            if global_step == start_step + _WARMUP_STEPS:   # baseline after warmup/compile
                t_warm = time.monotonic()
            if accelerator.is_main_process and global_step % 50 == 0:
                accelerator.log(logs, step=global_step)
                thru = ""
                if t_warm is not None and global_step > start_step + _WARMUP_STEPS:
                    sps = (time.monotonic() - t_warm) / (global_step - start_step - _WARMUP_STEPS)
                    thru = f" | {sps:.2f}s/step (~{round(3600 / sps)} steps/h)"
                logger.info(f"step {global_step}: {logs}{thru}")
            if keeper.due(global_step):
                sd = _gather_full_state_dict(gen, distributed)   # collective on all ranks
                if accelerator.is_main_process:
                    out, perm = keeper.save(sd, global_step)
                    logger.info(f"saved {'PERMANENT' if perm else 'rolling'} {out} "
                                f"(step {global_step}, {(time.monotonic() - t_start) / 3600:.2f}h elapsed)")
                # full resume bundle (overwrites the single training_state.pt)
                if args.save_resume_state:
                    rs = _gather_train_state(gen, critic, trainer.opt_g, trainer.opt_c,
                                             global_step, distributed)  # collective
                    if accelerator.is_main_process:
                        _save_resume_atomic(rs, args.output_dir)
                # qualitative previews share the checkpoint cadence; the rollout is a
                # collective under FSDP, so this runs on every rank (decode/log on main).
                previewer(gen, global_step, accelerator=accelerator, logger=logger)
            if global_step >= args.max_train_steps:
                break

    # always end on a permanent checkpoint (and reclaim any pending rolling file)
    sd = _gather_full_state_dict(gen, distributed)       # collective on all ranks
    if accelerator.is_main_process:
        out, _ = keeper.save(sd, global_step, force_permanent=True)
        logger.info(f"saved final PERMANENT {out} (step {global_step})")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="run weightless synthetic smoke steps")
    a = parser.parse_args()
    if a.smoke:
        print("smoke logs:", run_smoke())
        print("SMOKE OK")
        return
    main(_load_config(a.config))


if __name__ == "__main__":
    cli()
