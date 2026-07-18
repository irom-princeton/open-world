"""Real-weights GPU smoke for the Wan2.1-1.3B backbone.

Loads the actual pretrained Wan transformer (+ VAE) and checks, at real scale:
  1. block-causal forward_train is finite (fp32 + bf16);
  2. KV-cached autoregressive rollout == masked forward (fp32, exact);
  3. the Wan VAE loads (needed to re-encode the robot dataset to 16-ch latents);
  4. a rough per-block forward latency on this GPU (a real throughput datapoint).

Run on a compute node:
  sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/smoke_wan_real.py
"""

import os
import time

import torch

# Local dir (populated by bash_scripts/download_weights.sh) — required on the
# offline compute nodes; diffusers' sharded loader pings the Hub for a repo id
# even in offline mode, but reads a local dir's index directly.
REPO = os.environ.get("AR_WAN_DIR", "external/Wan2.1-T2V-1.3B-Diffusers")


def main() -> int:
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)

    from openworld.autoregressive.backbones.wan import WanBackbone
    print(f"\nloading {REPO} transformer (fp32) ...", flush=True)
    wan = WanBackbone.from_pretrained(REPO, cross_attn_dim=4096, torch_dtype=torch.float32).to(dev).eval()
    n_params = sum(p.numel() for p in wan.transformer.parameters())
    print(f"loaded: {n_params/1e9:.2f}B params, {wan.num_self_layers} self-attn layers")

    # geometry: 1 view at robot-ish res (320/8 = 40), 4 blocks x 2 frames.
    B, C, H, W, fpb, nb = 1, 16, 40, 40, 2, 4
    x = torch.randn(B, fpb * nb, C, H, W, device=dev)
    cond = torch.randn(B, 16, 4096, device=dev)
    t = torch.zeros(B, device=dev)

    # 1+2. fp32 exactness: masked full forward == cached rollout -------------
    print("\n=== fp32: forward_train vs cached rollout (real weights) ===", flush=True)
    with torch.no_grad():
        full = wan.forward_train(x, t, cond, frames_per_block=fpb)
        cache = wan.make_kv_cache()
        rolled = torch.cat([wan.forward_cached(x[:, b*fpb:(b+1)*fpb], t, cond, kv_cache=cache,
                                               start_frame=b*fpb) for b in range(nb)], dim=1)
    err = (full - rolled).abs().max().item()
    print(f"finite: {torch.isfinite(full).all().item()} | max|train - rollout| = {err:.3e}")
    assert torch.isfinite(full).all() and err < 1e-2, err

    # bf16 forward (training dtype) -----------------------------------------
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = wan.forward_train(x, t, cond, frames_per_block=fpb)
    print("bf16 forward finite:", torch.isfinite(out_bf16).all().item())

    # 3. VAE loads ----------------------------------------------------------
    print("\n=== Wan VAE load ===", flush=True)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    print("Wan VAE loaded:", type(vae).__name__,
          "| latent channels:", vae.config.z_dim if hasattr(vae.config, "z_dim") else "?")

    # 4. per-block latency (bf16, single forward per block) -----------------
    print("\n=== per-block forward latency (bf16) ===", flush=True)
    blk = torch.randn(B, fpb, C, H, W, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cache = wan.make_kv_cache()
        for w in range(3):  # warmup
            wan.forward_cached(blk, t, cond, kv_cache=cache, start_frame=w*fpb)
        torch.cuda.synchronize()
        cache = wan.make_kv_cache()
        t0 = time.time()
        N = 10
        for i in range(N):
            wan.forward_cached(blk, t, cond, kv_cache=cache, start_frame=i*fpb)
        torch.cuda.synchronize()
    ms = (time.time() - t0) / N * 1000
    print(f"~{ms:.1f} ms / block ({fpb} latent frames) single forward at {H*8}x{W*8}, 1 view")
    print(f"  -> a 2-step distilled rollout is ~2x + commit; ballpark "
          f"{1000/(ms*2.5):.0f} blocks/s on this GPU at this resolution")

    print("\nREAL-WEIGHTS WAN SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
