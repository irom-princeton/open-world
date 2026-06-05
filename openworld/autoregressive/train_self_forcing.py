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
from pathlib import Path

import torch

from .config import ARWMArgs
from .distill.scheduler import FlowMatchScheduler
from .distill.self_forcing import SelfForcingTrainer
from .model import build_training_stack


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


# ---------------------------------------------------------------------------
# Smoke mode: synthetic, weightless, CPU-fast. Validates the full train loop.
# ---------------------------------------------------------------------------
def run_smoke(steps: int = 1) -> dict:
    torch.set_num_threads(2)  # keep the weightless CPU smoke light for CI / shared nodes
    cfg = ARWMArgs(
        backbone="dummy", random_init_backbone=True, multiview_layout="height_stack",
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
    from accelerate.logging import get_logger

    logger = get_logger(__name__)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
    )
    gen, critic, teacher = build_training_stack(args)
    # student-init (E1) / teacher (E2) weights, if provided.
    if args.student_init_ckpt:
        gen.load_state_dict(torch.load(args.student_init_ckpt, map_location="cpu"), strict=False)
    if args.teacher_ckpt:
        sd = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(sd, strict=False)
        critic.load_state_dict(sd, strict=False)  # critic init from teacher

    sched = FlowMatchScheduler(args.denoising_step_list, num_train_timestep=args.num_train_timestep,
                               warp=args.warp_denoising_step)
    trainer = SelfForcingTrainer(
        gen, critic, teacher, sched, frames_per_block=args.frames_per_block,
        gen_lr=args.learning_rate, critic_lr=args.critic_learning_rate,
        critic_steps=args.critic_steps_per_gen_step, real_cfg=args.real_guidance_scale,
        dmd_lo=args.dmd_min_step_ratio, dmd_hi=args.dmd_max_step_ratio,
    )
    gen, critic, teacher = (m.to(accelerator.device) for m in (gen, critic, teacher))

    # --- data ---------------------------------------------------------------
    from openworld.training.world_model.dataset import LiberoLatentDataset
    train_dataset = LiberoLatentDataset(args, mode="train")
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
    logger.info(f"Starting AR self-forcing training: {args.backbone}, "
                f"{args.rollout_blocks} blocks x {args.frames_per_block} frames/block")
    while global_step < args.max_train_steps:
        for batch in loader:
            latent = batch["latent"].to(accelerator.device)      # [B, F, C, H, W]
            actions = batch["action"].to(accelerator.device)
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
            if accelerator.is_main_process and global_step % 50 == 0:
                accelerator.log(logs, step=global_step)
                logger.info(f"step {global_step}: {logs}")
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                out = Path(args.output_dir) / f"checkpoint-{global_step}.pt"
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(accelerator.unwrap_model(gen).state_dict(), out)
                logger.info(f"saved {out}")
            if global_step >= args.max_train_steps:
                break


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
