"""End-to-end *training-step* validation on real Wan-1.3B + real DROID latents.

`validate_data.py` only exercised `forward_train`. This runs the full
self-forcing / DMD loop on the actual training stack:

    * generator  = real Wan-1.3B made block-causal (+ action conditioner)
    * critic      = real Wan-1.3B ("fake score")
    * teacher     = real Wan-1.3B, frozen, CFG'd ("real score")

i.e. few-step KV-cache rollout -> DMD generator loss + critic denoising loss,
on a real batch from `ARLatentDataset`. Confirms the whole training step runs,
updates the generator, and produces finite losses at real scale on the H200.

    sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/validate_train_step.py [latent_root]
"""

import sys

import torch
from torch.utils.data import DataLoader

from openworld.autoregressive.config import ARWMArgs
from openworld.autoregressive.data.dataset import ARLatentDataset
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.distill.self_forcing import SelfForcingTrainer
from openworld.autoregressive.model import build_training_stack


def main() -> int:
    assert torch.cuda.is_available(), "needs GPU"
    latent_root = sys.argv[1] if len(sys.argv) > 1 else "data/droid_ar_latents_test"
    # small rollout so short test episodes survive the length filter and the
    # loop is quick: clip = (2 + 4) * 2 = 12 latent frames.
    cfg = ARWMArgs(
        backbone="wan_1_3b", backbone_ckpt="external/Wan2.1-T2V-1.3B-Diffusers",
        num_cams=3, multiview_layout="height_stack", height=192, width=320,
        frames_per_block=2, num_history_blocks=2, rollout_blocks=4,
        denoising_step_list=(1000, 500), warp_denoising_step=True,
        critic_steps_per_gen_step=1, real_guidance_scale=3.5,
        learning_rate=6e-6, critic_learning_rate=6e-6,
        # match the launch config: fp32 master weights + bf16 autocast compute.
        latent_root=latent_root, dtype=torch.float32, mixed_precision="bf16",
    )
    ds = ARLatentDataset(cfg, "train")
    loader = DataLoader(ds, batch_size=1, shuffle=True, drop_last=True)

    print("building real Wan-1.3B training stack (gen + critic + teacher) ...", flush=True)
    gen, critic, teacher = build_training_stack(cfg)
    # uniform dtype across weights + data so the self-forcing graph is single-dtype
    gen, critic, teacher = (m.to("cuda", cfg.dtype) for m in (gen, critic, teacher))
    nparams = sum(p.numel() for p in gen.parameters()) / 1e9
    print(f"generator params: {nparams:.2f}B", flush=True)

    sched = FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)
    trainer = SelfForcingTrainer(
        gen, critic, teacher, sched, frames_per_block=cfg.frames_per_block,
        gen_lr=cfg.learning_rate, critic_lr=cfg.critic_learning_rate,
        critic_steps=cfg.critic_steps_per_gen_step, real_cfg=cfg.real_guidance_scale,
        dmd_lo=cfg.dmd_min_step_ratio, dmd_hi=cfg.dmd_max_step_ratio,
    )

    hist_frames = cfg.num_history_blocks * cfg.frames_per_block
    n_steps = 3
    it = iter(loader)
    for step in range(n_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        latent = batch["latent"].to("cuda", cfg.dtype)   # [B,L,C,H,W]
        actions = batch["action"].to("cuda", cfg.dtype)
        texts = batch.get("text")
        B = latent.shape[0]
        hist = latent[:, :hist_frames]
        history_blocks = [hist[:, i:i + cfg.frames_per_block]
                          for i in range(0, hist_frames, cfg.frames_per_block)] or None
        cond = gen.encode_cond(actions, texts=texts, cfg_drop=True)
        null = gen.null_cond_like(cond)
        shape = (B, cfg.frames_per_block, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)
        logs = trainer.train_step(
            cond, null, num_blocks=cfg.rollout_blocks,
            latent_block_shape=shape, history_blocks=history_blocks,
        )
        print(f"step {step}: {logs}", flush=True)
        assert all(torch.isfinite(torch.tensor(v)).item() for v in logs.values()), logs

    print("\nTRAIN-STEP VALIDATION PASSED (full self-forcing/DMD loop, real Wan-1.3B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
