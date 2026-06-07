"""Quick generation-throughput benchmark for the AR world-model student.

Times the KV-cached autoregressive rollout (``model.rollout``) with REAL Wan-1.3B
weights on one GPU, at a couple of denoising-step settings, and reports both
latent-frame and decoded-RGB fps. This is the number that matters for the
distilled student (the deliverable); the undistilled teacher would run the same
backbone at ~50 full bidirectional steps with no KV cache (see --teacher-steps
for a same-backbone, no-cache reference timing).

    sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/bench_ar_rollout.py \
        --config configs/training/ar_wan_droid.py --blocks 8 --decode
"""

from __future__ import annotations

import argparse
import time

import torch

from openworld.autoregressive.config import BACKBONE_PRESETS
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_rollout(model, cfg, *, steps, blocks, hist_blocks, device, dtype, reps):
    fpb = cfg.frames_per_block
    Hs = cfg.latent_h_total
    W = cfg.latent_w
    sched = FlowMatchScheduler(tuple(s for s in cfg.denoising_step_list[:steps]),
                               num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)
    N = (hist_blocks + blocks) * fpb
    hist = torch.randn(hist_blocks, fpb, cfg.in_channels, Hs, W, device=device, dtype=dtype)
    history_blocks = [hist[i:i + 1].reshape(1, fpb, cfg.in_channels, Hs, W) for i in range(hist_blocks)]
    actions = torch.randn(1, N, cfg.action_dim, device=device)
    ac = torch.autocast(device.type, dtype=dtype) if device.type == "cuda" else _null()
    with ac:
        cond = model.encode_cond(actions, cfg_drop=False)
        shape = (1, fpb, cfg.in_channels, Hs, W)
        # warmup (CUDA graphs / cudnn autotune / alloc)
        model.rollout(history_blocks, cond, num_blocks=blocks, latent_block_shape=shape, scheduler=sched)
        _sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = model.rollout(history_blocks, cond, num_blocks=blocks,
                                latent_block_shape=shape, scheduler=sched)
        _sync()
        dt = (time.perf_counter() - t0) / reps
    gen_lat = blocks * fpb
    rgb = (gen_lat) * BACKBONE_PRESETS[cfg.backbone]["vae_temporal_factor"]  # ~4x temporal
    return dt, gen_lat, rgb, out


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--blocks", type=int, default=8, help="generated blocks per rollout")
    p.add_argument("--hist-blocks", type=int, default=1)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--steps-list", type=int, nargs="+", default=[2, 4])
    p.add_argument("--decode", action="store_true", help="also time VAE decode -> RGB")
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(a.config)
    dtype = cfg.autocast_dtype or cfg.dtype
    print(f"[bench] device={device} backbone={cfg.backbone} "
          f"geom={cfg.num_cams}cam {cfg.height}x{cfg.width} fpb={cfg.frames_per_block} "
          f"latent={cfg.latent_h_total}x{cfg.latent_w} autocast={dtype}", flush=True)

    print(f"[bench] building ARWorldModel (real weights: {cfg.resolved_backbone_ckpt}) ...", flush=True)
    model = ARWorldModel(cfg).to(device, cfg.dtype).eval()

    if len(cfg.denoising_step_list) < max(a.steps_list):
        print(f"[bench] NOTE: cfg.denoising_step_list has {len(cfg.denoising_step_list)} steps; "
              f"capping requested step counts to it where needed.")

    for steps in a.steps_list:
        s = min(steps, len(cfg.denoising_step_list))
        dt, gen_lat, rgb, out = time_rollout(
            model, cfg, steps=s, blocks=a.blocks, hist_blocks=a.hist_blocks,
            device=device, dtype=dtype, reps=a.reps)
        lat_fps = gen_lat / dt
        rgb_fps = rgb / dt
        print(f"[bench] steps={s:>2}  blocks={a.blocks}  "
              f"rollout={dt*1000:8.1f} ms  | gen {gen_lat} latent frames ({rgb} RGB)  "
              f"-> {lat_fps:6.1f} latent-fps  {rgb_fps:6.1f} RGB-fps", flush=True)

        if a.decode:
            from diffusers import AutoencoderKLWan

            from openworld.autoregressive.data.decode import VaeLatentDecoder, decode_stacked
            vae = AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder="vae", torch_dtype=torch.float32)
            dec = VaeLatentDecoder(vae, device=device, dtype=torch.float32)
            lat = out[0].float()                       # [gen_lat, C, Hs, W]
            _sync(); t0 = time.perf_counter()
            vid = decode_stacked(dec, lat, cfg.num_cams)
            _sync(); ddt = time.perf_counter() - t0
            print(f"[bench]          decode {ddt*1000:8.1f} ms for {vid.shape[0]} RGB frames "
                  f"-> {vid.shape[0]/ddt:6.1f} decode-fps  (frame {vid.shape[1]}x{vid.shape[2]})", flush=True)
            e2e = dt + ddt
            print(f"[bench]          end-to-end (rollout+decode) -> {vid.shape[0]/e2e:6.1f} RGB-fps", flush=True)
            del vae, dec

    print("[bench] DONE", flush=True)


if __name__ == "__main__":
    main()
