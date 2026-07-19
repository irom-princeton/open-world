"""Open-loop trajectory replay for the autoregressive world model.

Loads a trained AR student (Wan / Cosmos backbone + action conditioner), primes
it with the ground-truth first frame(s) of a recorded trajectory, feeds the
recorded action sequence open-loop, and lets the model autoregressively generate
the rest of the clip. Decodes both the ground-truth and the predicted latents
with the Wan VAE and writes them **side by side** (GT | PRED) for comparison.

    sbatch bash_scripts/ar_gpu.slurm .venv/bin/python scripts/replay_ar.py \
        --config configs/training/ar_wan_droid.py \
        --checkpoint checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
        --latent-root data/droid_ar_latents --split val \
        --output-dir checkpoints/ar_wm/ar_wan_droid/replay

Without ``--checkpoint`` the (untrained) backbone weights are used — useful only
to smoke-test the inference plumbing end to end, not for meaningful video.

Output:
    <output-dir>/<ep_id>.mp4          # side-by-side GT | PRED (decoded)
    <output-dir>/gt/<ep_id>.mp4       # ground truth only          (--separate)
    <output-dir>/pred/<ep_id>.mp4     # prediction only            (--separate)
    <output-dir>/replay_summary.json  # per-episode latent/pixel MSE, PSNR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mediapy
import numpy as np
import torch

from openworld.autoregressive.data.decode import VaeLatentDecoder, decode_stacked
from openworld.autoregressive.infer import (
    load_action_stats,
    load_full_episode,
    normalize_actions,
    replay_episode_latents,
)
from openworld.autoregressive.model import ARWorldModel
from openworld.autoregressive.train_self_forcing import _load_config


def list_episode_ids(latent_root: str, split: str) -> list[str]:
    with open(Path(latent_root) / f"{split}_sample.json") as f:
        return [s["ep_id"] for s in json.load(f)]


def _side_by_side(gt: np.ndarray, pred: np.ndarray, gap: int = 4) -> np.ndarray:
    """Two ``[T, H, W, 3]`` clips -> one ``[T, H, W_gt + gap + W_pred, 3]`` clip."""
    T = min(gt.shape[0], pred.shape[0])
    gt, pred = gt[:T], pred[:T]
    sep = np.full((T, gt.shape[1], gap, 3), 128, dtype=np.uint8)
    return np.concatenate([gt, sep, pred], axis=2)


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps, codec="h264",
                        ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="ARWMArgs config (configs/training/ar_*.py).")
    p.add_argument("--checkpoint", default=None, help="Trained student .pt (gen state_dict). Omit for untrained smoke.")
    p.add_argument("--latent-root", default=None, help="Override cfg.latent_root.")
    p.add_argument("--vae-dir", default="external/Wan2.1-T2V-1.3B-Diffusers", help="Wan VAE for decoding.")
    p.add_argument("--output-dir", default=None, help="Default: <checkpoint dir>/replay or replay_out/<tag>.")
    p.add_argument("--split", default="val", help="Dataset split to replay.")
    p.add_argument("--episode-id", default=None, help="Single episode; default: all in split.")
    p.add_argument("--num-episodes", type=int, default=0, help="Cap episodes (0=all).")
    p.add_argument("--history-blocks", type=int, default=1,
                   help="Ground-truth blocks used to prime the model ('first frame' = 1).")
    p.add_argument("--max-blocks", type=int, default=0, help="Cap generated blocks (0=to episode end).")
    p.add_argument("--denoising-steps", type=int, default=0,
                   help="Many-step preview sampler steps for non-distilled (mid-training) "
                        "checkpoints. 0 -> use cfg.preview_denoising_steps. Matches the "
                        "training-time _SamplePreviewer; avoids the few-step distilled list "
                        "that yields garbage on a student_init/teacher backbone. "
                        "On a self_forcing (DMD) config this is normally a no-op (the real "
                        "few-step list is used); pass a value WITH --force-many-step to render "
                        "a distilled checkpoint at many steps (collapse-vs-undistilled diagnostic).")
    p.add_argument("--force-many-step", action="store_true",
                   help="Use the N-step (--denoising-steps) sampler even on a self_forcing "
                        "config, instead of the few-step distilled cfg.denoising_step_list.")
    p.add_argument("--action-override", choices=["none", "zero", "const"], default="none",
                   help="Diagnostic: 'zero' feeds neutral actions, 'const' holds the first "
                        "action for the whole rollout. If spurious camera motion vanishes, it "
                        "was being driven by the action conditioning.")
    p.add_argument("--conditioning", choices=["episode", "action"], default="episode",
                   help="camera_cond source (camera_cond checkpoints only): 'episode' renders "
                        "band+raymap from the RECORDED sidecar geometry (training parity); "
                        "'action' synthesises them CLOSED-LOOP from the commanded joint chunk via "
                        "FK (deploy/policy-in-the-loop path). 'action' requires a joint sidecar.")
    p.add_argument("--video-fps", type=int, default=8)
    p.add_argument("--separate", action="store_true", help="Also write gt/ and pred/ videos separately.")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(a.config)
    if a.latent_root:
        cfg.latent_root = a.latent_root
    max_blocks = a.max_blocks if a.max_blocks > 0 else None

    # Build the SAME many-step sampler the training-time _SamplePreviewer uses for
    # non-distilled (mid-training) checkpoints. With scheduler=None, model.rollout
    # falls back to the few-step distilled cfg.denoising_step_list, which produces
    # garbage on a student_init/teacher backbone.
    sched = None
    use_many_step = (getattr(cfg, "stage", "self_forcing") != "self_forcing") or a.force_many_step
    if use_many_step:
        from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
        n = max(2, int(a.denoising_steps) or int(cfg.preview_denoising_steps))
        ts = cfg.num_train_timestep
        steps = tuple(int(round(ts * (i + 1) / n)) for i in reversed(range(n)))  # ~ts..ts/n
        sched = FlowMatchScheduler(steps, num_train_timestep=ts, warp=False)
        forced = " (forced on self_forcing)" if a.force_many_step and cfg.stage == "self_forcing" else ""
        print(f"[replay] preview sampler: {n}-step FlowMatchScheduler (warp=False){forced}, stage={cfg.stage}")

    out_root = (Path(a.output_dir) if a.output_dir
                else (Path(a.checkpoint).resolve().parent / "replay" if a.checkpoint
                      else Path("replay_out") / cfg.tag))

    # --- model ---
    print(f"[replay] building ARWorldModel ({cfg.backbone}) ...", flush=True)
    model = ARWorldModel(cfg).to(device).eval()
    if a.checkpoint:
        print(f"[replay] loading checkpoint {a.checkpoint}")
        missing, unexpected = model.load_state_dict(
            torch.load(a.checkpoint, map_location="cpu"), strict=False)
        if missing:
            print(f"[replay] {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[replay] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    else:
        print("[replay] WARNING: no --checkpoint; using untrained weights (plumbing smoke only).")

    # --- VAE decoder ---
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder="vae", torch_dtype=torch.float32)
    decoder = VaeLatentDecoder(vae, device=device, dtype=torch.float32)

    # --- episodes ---
    ep_ids = [a.episode_id] if a.episode_id else list_episode_ids(cfg.latent_root, a.split)
    if a.num_episodes > 0:
        ep_ids = ep_ids[: a.num_episodes]
    p01, p99 = load_action_stats(cfg.latent_root)

    # geometric (camera_cond) conditioning: recorded-sidecar (episode) or FK-closed-loop (action).
    # These K extra channels ride the widened patch-embed slot; the ckpt is otherwise unloadable.
    cam_data, cam_sel, joint_data = None, None, None
    if getattr(cfg, "camera_cond", False):
        cam_data = np.load(os.path.join(cfg.latent_root, f"{a.split}_camera_cond.npy"),
                           allow_pickle=True).item()
        cam_sel = list(cfg.view_indices) if getattr(cfg, "view_indices", None) else None
        if a.conditioning == "action":
            # closed-loop needs the commanded joint chunk [Lf,16] to FK the band+raymap.
            joint_data = np.load(os.path.join(cfg.latent_root, f"{a.split}_joint_actions.npy"),
                                 allow_pickle=True).item()
            print("[replay] camera_cond=CLOSED-LOOP (FK from commanded joints)")
        else:
            print("[replay] camera_cond=EPISODE (recorded sidecar geometry)")

    summary = []
    for ep_id in ep_ids:
        latent_gt, action_raw, text = load_full_episode(cfg.latent_root, a.split, ep_id, cfg.num_cams, view_indices=getattr(cfg, 'view_indices', None))
        action_norm = normalize_actions(action_raw, p01, p99)
        # Diagnostic action overrides: kill action-driven variation to test whether
        # spurious (static-camera) motion is being driven by the action conditioning.
        if a.action_override == "zero":
            action_norm = np.zeros_like(action_norm)                     # neutral action everywhere
        elif a.action_override == "const":
            action_norm = np.repeat(action_norm[:1], action_norm.shape[0], axis=0)  # hold the first (no commanded change)
        # camera_cond: render the K extra geometric channels for this episode (the widened
        # patch-embed slot). Same signal the trainer fed; None -> plain 16-channel model.
        pix = None
        if cam_data is not None and ep_id in cam_data:
            # latent_gt is [L, C, V*h, w]; per-view latent height = (V*h)//V
            h = latent_gt.shape[2] // cfg.num_cams; w = latent_gt.shape[3]
            sel = cam_sel if cam_sel is not None else list(range(cfg.num_cams))
            rec = cam_data[ep_id]
            if a.conditioning == "action" and joint_data is not None and ep_id in joint_data:
                # CLOSED-LOOP: anchor from the episode's initial geometry, then FK the
                # COMMANDED joint chunk into band+evolving-wrist-c2w -> the same signal a
                # policy rollout would feed. (Here the "command" is the recorded joint
                # trajectory, so this also validates FK-parity vs the episode path.)
                from openworld.autoregressive.conditioning.closed_loop_camera_cond import (
                    ClosedLoopCameraCond)
                clc = ClosedLoopCameraCond(
                    np.asarray(rec["pose"])[0], np.asarray(rec["c2w"])[0],
                    np.asarray(joint_data[ep_id])[0], np.asarray(rec["K"]),
                    np.asarray(rec["band_valid"]))
                pix = clc.render(np.asarray(joint_data[ep_id]), sel, h, w,
                                 wrist_band=getattr(cfg, "camera_cond_wrist_band", False),
                                 draw_sticks=getattr(cfg, "camera_cond_sticks", False))
            else:
                from openworld.autoregressive.conditioning.camera_cond import render_camera_cond_full
                pix = render_camera_cond_full(
                    rec, sel, h, w, band_scale=True,
                    draw_band=getattr(cfg, "camera_cond_band", True),
                    draw_sticks=getattr(cfg, "camera_cond_sticks", False),
                    wrist_band=getattr(cfg, "camera_cond_wrist_band", False))
            pix = pix.float()                                        # [L, K, V*h, w] torch.float32
        try:
            gt_lat, pred_lat, n_hist = replay_episode_latents(
                model, latent_gt, action_norm,
                frames_per_block=cfg.frames_per_block, num_history_blocks=a.history_blocks,
                in_channels=cfg.in_channels, device=device, dtype=cfg.dtype, max_blocks=max_blocks,
                scheduler=sched, pixel_cond=pix,
            )
        except RuntimeError as e:
            print(f"[replay] {ep_id}: skipped ({e})")
            continue

        gt_vid = decode_stacked(decoder, gt_lat, cfg.num_cams)       # [T, V*H, W, 3]
        pred_vid = decode_stacked(decoder, pred_lat, cfg.num_cams)
        write_video(_side_by_side(gt_vid, pred_vid), out_root / f"{ep_id}.mp4", a.video_fps)
        if a.separate:
            write_video(gt_vid, out_root / "gt" / f"{ep_id}.mp4", a.video_fps)
            write_video(pred_vid, out_root / "pred" / f"{ep_id}.mp4", a.video_fps)

        # metrics on the *generated* region only (history is ground truth)
        gen_gt, gen_pred = gt_lat[n_hist:], pred_lat[n_hist:]
        latent_mse = float(((gen_gt - gen_pred) ** 2).mean())
        T = min(gt_vid.shape[0], pred_vid.shape[0])
        pixel_mse = float(np.mean((gt_vid[:T] / 255.0 - pred_vid[:T] / 255.0) ** 2))
        psnr = float(10 * np.log10(1.0 / max(pixel_mse, 1e-12)))
        print(f"[replay] {ep_id}: {pred_lat.shape[0]} frames (hist {n_hist}) "
              f"latent-MSE={latent_mse:.4f} PSNR={psnr:.2f}dB  {text!r}")
        summary.append({
            "episode": ep_id, "frames": int(pred_lat.shape[0]), "history_frames": int(n_hist),
            "latent_mse": latent_mse, "pixel_mse": pixel_mse, "psnr_db": psnr, "text": text,
        })

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[replay] wrote {len(summary)} episodes -> {out_root}/ (side-by-side + replay_summary.json)")


if __name__ == "__main__":
    main()
