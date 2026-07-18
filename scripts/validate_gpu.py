"""Weightless GPU validation for the autoregressive world model.

Runs on a compute node (sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/validate_gpu.py).
Confirms the device/dtype paths the CPU unit tests don't exercise:

  1. DummyDiT KV-cache == masked forward, on CUDA (fp32, exact).
  2. Real diffusers Wan transformer (random-init) train == cached rollout, on CUDA.
  3. Wan forward in bf16 is finite (the training dtype).
  4. Real Cosmos transformer (random-init) block-causal forward is finite on CUDA.
  5. Self-forcing/DMD train_step runs on CUDA and updates the generator.

No weights are downloaded.
"""

import math
import sys

import torch


def section(name):
    print(f"\n=== {name} ===", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available on this node")
        return 1
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)

    # 1. Dummy KV-cache equivalence on GPU (fp32) ---------------------------
    section("DummyDiT KV-cache == masked forward (CUDA, fp32)")
    from openworld.autoregressive.backbones.dummy import DummyDiT
    torch.manual_seed(0)
    C, H, W, fpb, nb = 16, 8, 8, 2, 4
    dit = DummyDiT(in_channels=C, dim=48, heads=4, layers=3, cross_attn_dim=32).to(dev).eval()
    x = torch.randn(1, fpb * nb, C, H, W, device=dev)
    cond = torch.randn(1, 5, 32, device=dev)
    t = torch.zeros(1, device=dev)
    with torch.no_grad():
        full = dit.forward_train(x, t, cond, frames_per_block=fpb)
        cache = dit.make_kv_cache()
        rolled = torch.cat([dit.forward_cached(x[:, b*fpb:(b+1)*fpb], t, cond, kv_cache=cache,
                                               start_frame=b*fpb) for b in range(nb)], dim=1)
    err = (full - rolled).abs().max().item()
    print(f"max|full - rollout| = {err:.3e}")
    assert err < 1e-4, err

    # 2 + 3. Real Wan transformer on GPU -----------------------------------
    section("Wan2.1 (random-init) train == cached rollout (CUDA, fp32) + bf16 forward")
    from openworld.autoregressive.backbones.wan import WanBackbone
    wan = WanBackbone.random_init(cross_attn_dim=64, small=True).to(dev).eval()
    xw = torch.randn(1, fpb * 2, 16, 16, 16, device=dev)
    cw = torch.randn(1, 8, 64, device=dev)
    tw = torch.zeros(1, device=dev)
    with torch.no_grad():
        full_w = wan.forward_train(xw, tw, cw, frames_per_block=fpb)
        cache = wan.make_kv_cache()
        rolled_w = torch.cat([wan.forward_cached(xw[:, b*fpb:(b+1)*fpb], tw, cw, kv_cache=cache,
                                                 start_frame=b*fpb) for b in range(2)], dim=1)
    errw = (full_w - rolled_w).abs().max().item()
    print(f"Wan fp32 max|train - rollout| = {errw:.3e}")
    assert errw < 1e-2, errw
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = wan.forward_train(xw, tw, cw, frames_per_block=fpb)
    print("Wan bf16 forward finite:", torch.isfinite(out_bf16).all().item())
    assert torch.isfinite(out_bf16).all()

    # 4. Real Cosmos transformer on GPU ------------------------------------
    section("Cosmos-Predict2 (random-init) block-causal forward (CUDA)")
    from openworld.autoregressive.backbones.cosmos_predict2 import CosmosBackbone
    cos = CosmosBackbone.random_init(cross_attn_dim=64, small=True).to(dev).eval()
    with torch.no_grad():
        out_c = cos.forward_train(xw, tw, cw, frames_per_block=fpb)
    print("Cosmos forward finite:", torch.isfinite(out_c).all().item(), "shape", tuple(out_c.shape))
    assert torch.isfinite(out_c).all()

    # 5. Self-forcing train_step on GPU ------------------------------------
    section("Self-forcing / DMD train_step on CUDA (DummyDiT)")
    from openworld.autoregressive.config import ARWMArgs
    from openworld.autoregressive.model import build_training_stack
    from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
    from openworld.autoregressive.distill.self_forcing import SelfForcingTrainer
    cfg = ARWMArgs(backbone="dummy", random_init_backbone=True, multiview_layout="height_stack",
                   num_cams=1, frames_per_block=2, rollout_blocks=2, num_history_blocks=1,
                   denoising_step_list=(1000, 500), critic_steps_per_gen_step=1,
                   width=32, height=32, dtype=torch.float32)
    gen, critic, teacher = (m.to(dev) for m in build_training_stack(cfg))
    sched = FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)
    trainer = SelfForcingTrainer(gen, critic, teacher, sched, frames_per_block=cfg.frames_per_block,
                                 critic_steps=cfg.critic_steps_per_gen_step)
    p0 = [p.detach().clone() for p in gen.parameters()]
    T = cfg.frames_per_block * (cfg.rollout_blocks + cfg.num_history_blocks)
    actions = torch.randn(1, T, cfg.action_dim, device=dev)
    cond = gen.encode_cond(actions, cfg_drop=True)
    logs = trainer.train_step(cond, gen.null_cond_like(cond), num_blocks=cfg.rollout_blocks,
                              latent_block_shape=(1, cfg.frames_per_block, cfg.in_channels,
                                                  cfg.latent_h_total, cfg.latent_w))
    moved = any((p.detach() - q).abs().max().item() > 0 for p, q in zip(gen.parameters(), p0))
    print("train_step logs:", {k: round(v, 5) for k, v in logs.items()}, "| gen updated:", moved)
    assert all(math.isfinite(v) for v in logs.values()) and moved

    print("\nALL GPU VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
