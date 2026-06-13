"""Bidirectional (teacher-stage) evaluation for the AR world model.

The L1b "teacher" (``stage=teacher``) is trained with FULL bidirectional
attention and plain flow-matching: a clean clip is noised at ONE uniform sigma
and the whole clip is denoised jointly -- no causal mask, no KV-cache, and no
image/history frame (the only conditioning is the recorded action sequence +
text). Autoregressive open-loop replay (``scripts/replay_ar.py``) is therefore
*off-distribution* for the teacher. This script evaluates it in its NATIVE
bidirectional mode, two complementary ways:

  --mode recon  : the training objective itself, on held-out val clips. For a
                  sweep of noise levels sigma, noise the GT clip, run ONE
                  bidirectional forward, recover x0_hat = x_sigma - sigma*v_pred,
                  and report latent MSE / PSNR vs GT. This is the rigorous "did
                  the teacher actually learn to denoise?" check -- a trained model
                  reconstructs well at low/mid sigma; an untrained backbone does
                  not. (Matches DiffusionTrainer.train_step, causal=False.)

  --mode replay : Ctrl-World-style trajectory replay. Multi-step bidirectional
                  denoising of the WHOLE clip conditioned on the recorded actions,
                  with the first ``--history-blocks`` frames pinned to the GT
                  latent (RePaint-style replacement) so the generation is anchored
                  to the initial frame(s) and can be compared to GT side-by-side.
                  This is inference-time inpainting -- the teacher was not
                  explicitly trained for it, but its bidirectional attention
                  supports conditioning on a clean sub-window.

Both modes decode with the Wan VAE and write side-by-side GT|PRED videos plus a
summary.json. Everything runs in the teacher's native full-clip bidirectional
forward (``model.forward_train(..., causal=False)``); there is no AR rollout.

    .venv/bin/python scripts/eval_teacher_bidir.py \
        --config configs/training/ar_wan_teacher_droid_aligned.py \
        --checkpoint checkpoints/ar_wm/ar_wan_teacher_aligned/checkpoint-13000-rolling.pt \
        --latent-root data/droid_ar_latents --split val --num-episodes 10 \
        --output-dir checkpoints/ar_wm/ar_wan_teacher_aligned/eval_bidir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np
import torch

from openworld.autoregressive.data.decode import VaeLatentDecoder, decode_stacked
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.infer import (
    load_action_stats,
    load_full_episode,
    normalize_actions,
)
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config


def list_episode_ids(latent_root: str, split: str) -> list[str]:
    with open(Path(latent_root) / f"{split}_sample.json") as f:
        return [s["ep_id"] for s in json.load(f)]


def _side_by_side(gt: np.ndarray, pred: np.ndarray, gap: int = 4) -> np.ndarray:
    T = min(gt.shape[0], pred.shape[0])
    gt, pred = gt[:T], pred[:T]
    sep = np.full((T, gt.shape[1], gap, 3), 128, dtype=np.uint8)
    return np.concatenate([gt, sep, pred], axis=2)


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps, codec="h264",
                        ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")


def _psnr(mse: float) -> float:
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


@torch.no_grad()
def recon_sweep(model, latent_gt, action_norm, sigmas, *, fpb, in_ch, device, dtype):
    """Teacher training objective on a held-out clip, at several noise levels.

    Returns (per_sigma: {sigma: latent_mse}, x0_hats: {sigma: [N,C,Hs,W]}, gt [N,C,Hs,W]).
    Linear sigma (no warp) -- matches DiffusionTrainer.train_step's add_noise.
    """
    L, C, Hs, W = latent_gt.shape
    assert C == in_ch, f"latent channels {C} != backbone in_channels {in_ch}"
    N = (L // fpb) * fpb
    x0 = latent_gt[:N].unsqueeze(0).to(device, dtype)                 # [1,N,C,Hs,W]
    actions = torch.from_numpy(action_norm[:N]).float().unsqueeze(0).to(device)
    sched = FlowMatchScheduler(num_train_timestep=1000, warp=False)   # only need add_noise / to_timestep
    cond = model.encode_cond(actions, cfg_drop=False)
    per_sigma, x0_hats = {}, {}
    for s in sigmas:
        sigma = torch.full((1,), float(s), device=device)
        eps = torch.randn_like(x0)
        x_sigma = sched.add_noise(x0, eps, sigma)
        t = sched.to_timestep(sigma)
        v = model.forward_train(x_sigma, t, cond, frames_per_block=fpb, causal=False)
        x0_hat = sched.x0_from_velocity(x_sigma, v, sigma)
        per_sigma[float(s)] = float(((x0_hat.float() - x0.float()) ** 2).mean())
        x0_hats[float(s)] = x0_hat[0].float().cpu()
    return per_sigma, x0_hats, x0[0].float().cpu()


@torch.no_grad()
def bidir_replay(model, latent_gt, action_norm, *, fpb, hist_frames, n_steps,
                 in_ch, device, dtype):
    """Multi-step bidirectional denoising of the whole clip, conditioned on the
    recorded actions, with the first ``hist_frames`` pinned to the GT latent
    (RePaint replacement). Returns (gt [N,C,Hs,W], pred [N,C,Hs,W], hist_frames).
    """
    L, C, Hs, W = latent_gt.shape
    assert C == in_ch, f"latent channels {C} != backbone in_channels {in_ch}"
    N = (L // fpb) * fpb
    gt = latent_gt[:N].unsqueeze(0).to(device, dtype)                 # [1,N,C,Hs,W]
    actions = torch.from_numpy(action_norm[:N]).float().unsqueeze(0).to(device)

    ts = 1000
    n = max(2, n_steps)
    steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))   # ts .. ts/n
    sched = FlowMatchScheduler(steps, num_train_timestep=ts, warp=False)
    sigmas = sched.sigmas                                              # descending, sigmas[0]=1.0

    cond = model.encode_cond(actions, cfg_drop=False)
    x = torch.randn_like(gt)                                          # sigma_0 == 1.0 -> pure noise
    for i in range(n):
        sigma_i = sigmas[i].to(device).expand(1)
        if hist_frames > 0:                                          # pin history to noised GT
            eps_h = torch.randn_like(gt[:, :hist_frames])
            x[:, :hist_frames] = sched.add_noise(gt[:, :hist_frames], eps_h, sigma_i)
        t_i = sched.to_timestep(sigma_i)
        v = model.forward_train(x, t_i, cond, frames_per_block=fpb, causal=False)
        x0_hat = sched.x0_from_velocity(x, v, sigma_i)
        if i < n - 1:
            sigma_next = sigmas[i + 1].to(device).expand(1)
            x = sched.add_noise(x0_hat, torch.randn_like(x0_hat), sigma_next)
        else:
            x_clean = x0_hat
    if hist_frames > 0:
        x_clean[:, :hist_frames] = gt[:, :hist_frames]               # exact GT history
    return gt[0].float().cpu(), x_clean[0].float().cpu(), hist_frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--latent-root", default=None)
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--mode", choices=["recon", "replay", "both"], default="both")
    p.add_argument("--clip-frames", type=int, default=0,
                   help="Latent frames per clip (0 -> training clip = (hist+rollout)*fpb).")
    p.add_argument("--history-blocks", type=int, default=1,
                   help="GT blocks pinned in --mode replay (RePaint anchor).")
    p.add_argument("--denoising-steps", type=int, default=0,
                   help="Multi-step count for --mode replay (0 -> cfg.preview_denoising_steps).")
    p.add_argument("--recon-sigmas", default="0.2,0.4,0.6,0.8,1.0",
                   help="Comma list of noise levels for --mode recon.")
    p.add_argument("--recon-vis-sigma", type=float, default=0.6,
                   help="Which recon sigma to decode for the side-by-side video.")
    p.add_argument("--video-fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(a.config)
    if a.latent_root:
        cfg.latent_root = a.latent_root
    if getattr(cfg, "stage", None) != "teacher":
        print(f"[eval] WARNING: cfg.stage={getattr(cfg,'stage',None)!r} (expected 'teacher'); "
              "bidirectional eval still runs but is only faithful for the teacher.")

    fpb = cfg.frames_per_block
    clip = a.clip_frames or (cfg.num_history_blocks + cfg.rollout_blocks) * fpb
    hist_frames = a.history_blocks * fpb
    n_steps = a.denoising_steps or cfg.preview_denoising_steps
    sigmas = [float(s) for s in a.recon_sigmas.split(",")]
    dtype = cfg.dtype

    out_root = (Path(a.output_dir) if a.output_dir
                else (Path(a.checkpoint).resolve().parent / "eval_bidir" if a.checkpoint
                      else Path("eval_bidir") / cfg.tag))

    print(f"[eval] building ARWorldModel ({cfg.backbone}); clip={clip} frames, fpb={fpb}, "
          f"hist={hist_frames}, replay-steps={n_steps}, mode={a.mode}", flush=True)
    model = ARWorldModel(cfg).to(device).eval()
    if a.checkpoint:
        print(f"[eval] loading checkpoint {a.checkpoint}")
        missing, unexpected = model.load_state_dict(
            torch.load(a.checkpoint, map_location="cpu"), strict=False)
        if missing:
            print(f"[eval] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[eval] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    else:
        print("[eval] WARNING: no --checkpoint; untrained weights (sanity baseline only).")

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder="vae", torch_dtype=torch.float32)
    decoder = VaeLatentDecoder(vae, device=device, dtype=torch.float32)

    ep_ids = list_episode_ids(cfg.latent_root, a.split)
    if a.num_episodes > 0:
        ep_ids = ep_ids[: a.num_episodes]
    p01, p99 = load_action_stats(cfg.latent_root)

    summary = []
    for ep_id in ep_ids:
        latent_gt, action_raw, text = load_full_episode(cfg.latent_root, a.split, ep_id, cfg.num_cams)
        action_norm = normalize_actions(action_raw, p01, p99)
        if latent_gt.shape[0] < clip:
            print(f"[eval] {ep_id}: skipped (only {latent_gt.shape[0]} < {clip} latent frames)")
            continue
        latent_gt = latent_gt[:clip]
        rec = {"episode": ep_id, "text": text, "frames": int((clip // fpb) * fpb)}

        if a.mode in ("recon", "both"):
            per_sigma, x0_hats, gt_lat = recon_sweep(
                model, latent_gt, action_norm, sigmas,
                fpb=fpb, in_ch=cfg.in_channels, device=device, dtype=dtype)
            rec["recon_latent_mse"] = per_sigma
            rec["recon_psnr_db"] = {s: _psnr(m) for s, m in per_sigma.items()}
            vs = min(sigmas, key=lambda s: abs(s - a.recon_vis_sigma))
            gt_vid = decode_stacked(decoder, gt_lat, cfg.num_cams)
            rc_vid = decode_stacked(decoder, x0_hats[vs], cfg.num_cams)
            write_video(_side_by_side(gt_vid, rc_vid), out_root / "recon" / f"{ep_id}_sigma{vs}.mp4", a.video_fps)
            tbl = "  ".join(f"s{s}:{_psnr(m):.1f}dB" for s, m in per_sigma.items())
            print(f"[eval][recon] {ep_id}: {tbl}  {text!r}")

        if a.mode in ("replay", "both"):
            gt_lat, pred_lat, nh = bidir_replay(
                model, latent_gt, action_norm,
                fpb=fpb, hist_frames=hist_frames, n_steps=n_steps,
                in_ch=cfg.in_channels, device=device, dtype=dtype)
            gt_vid = decode_stacked(decoder, gt_lat, cfg.num_cams)
            pred_vid = decode_stacked(decoder, pred_lat, cfg.num_cams)
            write_video(_side_by_side(gt_vid, pred_vid), out_root / "replay" / f"{ep_id}.mp4", a.video_fps)
            gen_gt, gen_pred = gt_lat[nh:], pred_lat[nh:]
            lat_mse = float(((gen_gt - gen_pred) ** 2).mean())
            T = min(gt_vid.shape[0], pred_vid.shape[0])
            pix_mse = float(np.mean((gt_vid[:T] / 255.0 - pred_vid[:T] / 255.0) ** 2))
            rec.update({"replay_history_frames": int(nh), "replay_latent_mse": lat_mse,
                        "replay_pixel_mse": pix_mse, "replay_psnr_db": _psnr(pix_mse)})
            print(f"[eval][replay] {ep_id}: latent-MSE={lat_mse:.4f} PSNR={_psnr(pix_mse):.2f}dB (hist {nh})")

        summary.append(rec)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "eval_summary.json").write_text(json.dumps(summary, indent=2))

    # aggregate
    if summary and a.mode in ("recon", "both"):
        print("\n[eval] === recon PSNR (dB) averaged over %d episodes ===" % len(summary))
        for s in sigmas:
            vals = [r["recon_psnr_db"][float(s)] for r in summary if "recon_psnr_db" in r]
            if vals:
                print(f"   sigma={s:<4}  mean PSNR = {np.mean(vals):6.2f} dB")
    if summary and a.mode in ("replay", "both"):
        vals = [r["replay_psnr_db"] for r in summary if "replay_psnr_db" in r]
        if vals:
            print("[eval] === replay PSNR mean = %.2f dB over %d episodes ===" % (np.mean(vals), len(vals)))
    print(f"[eval] wrote {len(summary)} episodes -> {out_root}/ (videos + eval_summary.json)")


if __name__ == "__main__":
    main()
