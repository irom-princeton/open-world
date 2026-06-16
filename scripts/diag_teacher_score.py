"""Teacher/critic score sanity probe for the DMD (self-forcing) stage.

Hypothesis (suspect A): the DMD score path scores the *generated* clip (the
``rollout_blocks`` frames, which occupy absolute positions ``hist..hist+N`` in the
self-forcing rollout) but passes the *full-window* action cond unsliced
(``_cond_for(cond, 0)``). With ``action_cond_mode="cross_attn_aligned"`` the
per-frame cross-mask then aligns generated frame ``j`` to action ``j`` instead of
action ``hist+j`` -- a constant off-by-``hist`` action misalignment (and the last
``hist`` actions are never attended). A misaligned *real score* yields a wrong DMD
target -> the collapse we see.

This probe is decisive without training: feed the frozen teacher a REAL GT clip
(which it must denoise well) using the EXACT call path DMD uses, two ways --

  * ``trainer`` : full-window cond, unsliced (Lkv = hist+N)  -> reproduces the bug
  * ``aligned`` : cond sliced to the clip's own action frames (Lkv = N)

and reports, per (sigma, cfg_scale), the x0-reconstruction MSE to the GT clip.
If ``aligned`` reconstructs the GT and ``trainer`` does not, the misalignment is
confirmed. Decoded frames are written for eyeballing.

    python scripts/diag_teacher_score.py --config configs/training/ar_wan_droid_aligned_v2.py
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch

from openworld.autoregressive.train_self_forcing import _load_config, _backbone_state_from_ckpt
from openworld.autoregressive.model import build_training_stack
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.distill.dmd import make_cfg_score_fn
from openworld.autoregressive.infer.replay import (
    load_full_episode, load_action_stats, normalize_actions,
)
from openworld.autoregressive.data.decode import VaeLatentDecoder, decode_stacked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--episode-id", default=None, help="val episode; default: first available.")
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--sigmas", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--cfg-scales", default="1.0,3.0")
    p.add_argument("--decode-sigma", type=float, default=0.5, help="sigma at which to decode frames.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(a.config)
    fpb = cfg.frames_per_block
    hist = cfg.num_history_blocks * fpb
    n_gen = cfg.rollout_blocks * fpb
    win = hist + n_gen
    autocast_dtype = getattr(cfg, "autocast_dtype", torch.bfloat16)
    print(f"[probe] stage={cfg.stage} mode={cfg.action_cond_mode} fpb={fpb} "
          f"hist={hist} n_gen={n_gen} window={win}")

    # --- models: generator (for encode_cond) + frozen teacher (real score) ---
    gen, critic, teacher = build_training_stack(cfg)
    if cfg.student_init_ckpt:
        sd = torch.load(cfg.student_init_ckpt, map_location="cpu", weights_only=False)
        m, u = gen.load_state_dict(sd, strict=False)
        print(f"[probe] generator <- {cfg.student_init_ckpt} (missing {len(m)}, unexpected {len(u)})")
    sd = torch.load(cfg.teacher_ckpt, map_location="cpu", weights_only=False)
    bb = _backbone_state_from_ckpt(sd)
    mt, ut = teacher.load_state_dict(bb, strict=False)
    print(f"[probe] teacher <- {cfg.teacher_ckpt} (missing {len(mt)}, unexpected {len(ut)})")
    gen = gen.to(device, torch.float32).eval()
    teacher = teacher.to(device, torch.float32).eval()
    for m_ in (gen, teacher):
        for prm in m_.parameters():
            prm.requires_grad_(False)

    sched = FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)

    # --- data: one real episode, take the generated-equivalent clip ----------
    ep = a.episode_id
    if ep is None:
        import os
        ep = sorted(f[:-3] for f in os.listdir(Path(cfg.latent_root) / "val") if f.endswith(".pt"))[0]
    latent_gt, action_raw, text = load_full_episode(cfg.latent_root, "val", ep, cfg.num_cams)
    p01, p99 = load_action_stats(cfg.latent_root)
    act = normalize_actions(action_raw, p01, p99)
    assert latent_gt.shape[0] >= win, f"episode {ep} too short ({latent_gt.shape[0]} < {win})"
    act_win = torch.from_numpy(act[:win]).float()                       # [win, A]
    # generated-equivalent clip: rollout frames live at absolute positions hist..hist+n_gen
    clip = latent_gt[hist:hist + n_gen].unsqueeze(0).to(device).float()  # [1, n_gen, C, Hs, W]
    print(f"[probe] episode={ep!r}  text={text!r}  clip={tuple(clip.shape)} "
          f"std={clip.std().item():.4f}")

    # conds (encode the *sliced* actions so the aligned variant is genuinely aligned)
    cond_full = gen.encode_cond(act_win.unsqueeze(0).to(device), cfg_drop=False)        # [1, win, cross]
    cond_aln = gen.encode_cond(act_win[hist:hist + n_gen].unsqueeze(0).to(device), cfg_drop=False)  # [1, n_gen, cross]

    sigmas = [float(s) for s in a.sigmas.split(",")]
    cfg_scales = [float(s) for s in a.cfg_scales.split(",")]
    variants = {"trainer(full,misaligned)": cond_full, "aligned(sliced)": cond_aln}

    def x0_mse(cond, sigma_val, scale):
        null = torch.zeros_like(cond)
        score = make_cfg_score_fn(teacher, frames_per_block=fpb, null_cond=null,
                                  scale=scale, causal=False)
        g = torch.Generator(device=device).manual_seed(a.seed)
        eps = torch.randn(clip.shape, device=device, generator=g)
        sigma = torch.full((1,), sigma_val, device=device)
        x_sigma = sched.add_noise(clip, eps, sigma)
        t = sched.to_timestep(sigma)
        ac = (torch.autocast(device.type, dtype=autocast_dtype) if device.type == "cuda"
              else contextlib.nullcontext())
        with torch.no_grad(), ac:
            v = score(x_sigma, t, cond)
        x0r = sched.x0_from_velocity(x_sigma, v, sigma).float()
        mse = torch.mean((x0r - clip) ** 2).item()
        return mse, x0r

    # --- table ---------------------------------------------------------------
    print("\n[probe] x0-reconstruction MSE to GT clip (lower=better; clip var="
          f"{clip.var().item():.4f}):")
    header = f"{'sigma':>6} {'cfg':>4} | " + " | ".join(f"{k:>26}" for k in variants)
    print(header); print("-" * len(header))
    for s in sigmas:
        for sc in cfg_scales:
            cells = []
            for cond in variants.values():
                mse, _ = x0_mse(cond, s, sc)
                cells.append(f"{mse:26.4f}")
            print(f"{s:6.2f} {sc:4.1f} | " + " | ".join(cells))

    # --- decode frames for eyeballing at one (sigma, cfg) --------------------
    out = Path(a.output_dir) if a.output_dir else Path(cfg.output_dir) / "replay_diag" / "teacher_probe"
    out.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder="vae", torch_dtype=torch.float32)
    dec = VaeLatentDecoder(vae, device=device, dtype=torch.float32)
    sc = cfg_scales[-1]
    import mediapy
    gt_rgb = decode_stacked(dec, clip[0], cfg.num_cams)
    mediapy.write_image(str(out / f"{ep}_GT.png"), gt_rgb[len(gt_rgb) // 2])
    for name, cond in variants.items():
        _, x0r = x0_mse(cond, a.decode_sigma, sc)
        rgb = decode_stacked(dec, x0r[0], cfg.num_cams)
        tag = name.split("(")[0]
        mediapy.write_image(str(out / f"{ep}_{tag}_sigma{a.decode_sigma}_cfg{sc}.png"),
                            rgb[len(rgb) // 2])
    print(f"\n[probe] wrote decoded frames -> {out}/")


if __name__ == "__main__":
    main()
