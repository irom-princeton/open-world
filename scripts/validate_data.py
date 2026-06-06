"""End-to-end data validation: precomputed DROID latents -> real Wan forward.

Assumes scripts/preprocess_ar_latents.py already wrote a (small) latent set.
Loads it via ARLatentDataset, batches it, builds the real ARWorldModel
(Wan-1.3B + action conditioner), and runs one block-causal forward_train — i.e.
confirms the data pipeline feeds the model with matching shapes/dtypes on GPU.

    sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/validate_data.py [latent_root]
"""

import sys

import torch
from torch.utils.data import DataLoader

from openworld.autoregressive.config import ARWMArgs
from openworld.autoregressive.data.dataset import ARLatentDataset
from openworld.autoregressive.model import ARWorldModel


def main() -> int:
    assert torch.cuda.is_available(), "needs GPU"
    latent_root = sys.argv[1] if len(sys.argv) > 1 else "data/droid_ar_latents_test"
    # small clip so short episodes survive the length filter
    cfg = ARWMArgs(
        backbone="wan_1_3b", backbone_ckpt="external/Wan2.1-T2V-1.3B-Diffusers",
        num_cams=3, multiview_layout="height_stack", height=192, width=320,
        frames_per_block=2, num_history_blocks=2, rollout_blocks=6,
        latent_root=latent_root,
    )
    ds = ARLatentDataset(cfg, "train")
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    latent, action, text = batch["latent"], batch["action"], batch["text"]
    print(f"batch: latent {tuple(latent.shape)} {latent.dtype}, action {tuple(action.shape)}, text[0]={text[0]!r}")
    L = (cfg.num_history_blocks + cfg.rollout_blocks) * cfg.frames_per_block
    exp = (2, L, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)
    assert tuple(latent.shape) == exp, (tuple(latent.shape), exp)
    assert action.abs().max() <= 1.0 + 1e-6

    print("building ARWorldModel (Wan-1.3B) ...", flush=True)
    model = ARWorldModel(cfg).to("cuda").eval()
    latent = latent.to("cuda")
    action = action.to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cond = model.encode_cond(action, cfg_drop=False)            # [B,L,4096]
        t = torch.zeros(latent.shape[0], device="cuda")
        out = model.forward_train(latent, t, cond, frames_per_block=cfg.frames_per_block)
    print(f"cond {tuple(cond.shape)} | forward_train out {tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
    assert out.shape == latent.shape and torch.isfinite(out).all()
    print("\nDATA PIPELINE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
