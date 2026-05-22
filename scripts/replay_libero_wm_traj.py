"""Auto-regressive trajectory replay for the LIBERO world model.

Replays recorded LIBERO trajectories through a trained ``CrtlWorld`` checkpoint
and writes ground-truth vs world-model-predicted videos side by side as two
separate sets.

The checkpoint is trained at 5 Hz (``data/wm_training/libero_processed_5hz``)
while collected rollouts are stored at 20 Hz, so each trajectory is strided
down to 5 Hz before replay (``--target_hz``). Actions are normalized with the
*training* stats (``--stat_root``), not the eval data.

Rollout (per chunk, at 5 Hz, mirroring LiberoLatentDataset._build_frame_ids
with deterministic skip=1, skip_his=4):

    frame_now -> history = rolled[frame_now - 6*4 .. -4]   (mostly past predictions)
                 current = rolled[frame_now]
                 action  = state[frame_now - 24 .. +4] (normalized)
                 pred[5] = pipeline(image=current, history=history, text=action_latent)
    overwrite rolled[frame_now .. frame_now+4] with pred so the next chunk
    conditions on its own predictions (closed loop).

Output:
    <output_dir>/gt/<suite>_<episode>.mp4     # decoded ground-truth latents
    <output_dir>/pred/<suite>_<episode>.mp4   # decoded predictions (same frames)
    <output_dir>/replay_summary.json          # per-episode latent/pixel MSE, PSNR

Usage:
    uv run scripts/replay_libero_wm_traj.py \\
        --checkpoint checkpoints/wm/libero/checkpoint-30000.pt \\
        --data_root  data/libero_collected \\
        --output_dir checkpoints/wm/libero/replay
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import einops
import mediapy
import numpy as np
import torch

from openworld.training.world_model.config import LiberoWMArgs
from openworld.training.world_model.dataset import _load_stat
from openworld.world_models.ctrl_world import CrtlWorld, CtrlWorldDiffusionPipeline


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def list_suites(data_root: str) -> list[str]:
    return sorted(p.name for p in Path(data_root).iterdir() if (p / "annotation").is_dir())


def list_episode_ids(data_root: str, suite: str, split: str) -> list[str]:
    ann_dir = Path(data_root) / suite / "annotation" / split
    return sorted(p.stem for p in ann_dir.glob("*.json"))


def load_full_episode(
    data_root: str, suite: str, episode_id: str, args: LiberoWMArgs, split: str
) -> tuple[torch.Tensor, np.ndarray, str, int]:
    """Return (latent[T,4,total_h,latent_w] fp32, action[T,7] fp32, text, native_fps)."""
    suite_dir = os.path.join(data_root, suite)
    with open(os.path.join(suite_dir, args.annotation_name, split, f"{episode_id}.json")) as f:
        label = json.load(f)

    cam_specs = label["latent_videos"]
    if len(cam_specs) < args.num_cams:
        raise ValueError(f"{episode_id}: {len(cam_specs)} cameras < num_cams={args.num_cams}")

    cam_tensors = []
    for cam_idx in range(args.num_cams):
        with open(os.path.join(suite_dir, cam_specs[cam_idx]["latent_video_path"]), "rb") as fh:
            cam_tensors.append(torch.load(fh, map_location="cpu").float())

    # Stack the two camera latents vertically along H, exactly as the dataset does.
    per_cam_h, latent_w = args.height // 8, args.width // 8
    T = min(int(t.shape[0]) for t in cam_tensors)
    latent = torch.zeros((T, 4, args.num_cams * per_cam_h, latent_w), dtype=torch.float32)
    for cam_idx, t in enumerate(cam_tensors):
        latent[:, :, cam_idx * per_cam_h : (cam_idx + 1) * per_cam_h] = t[:T]

    cart = np.asarray(label["observation.state.cartesian_position"], dtype=np.float32)
    grip = np.asarray(label["observation.state.gripper_position"], dtype=np.float32)
    if grip.ndim == 1:
        grip = grip[:, None]
    action_native = np.concatenate([cart, grip], axis=-1)[:T]  # (T, 7)

    text = label["texts"][0] if label.get("texts") else label.get("language_instruction", "")
    return latent, action_native, text, int(label.get("fps", 20))


def normalize_actions(action: np.ndarray, p01: np.ndarray, p99: np.ndarray) -> np.ndarray:
    return np.clip(2 * (action - p01) / (p99 - p01 + 1e-8) - 1, -1, 1)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def build_frame_ids(frame_now: int, num_history: int, num_frames: int,
                     skip: int, skip_his: int) -> list[int]:
    """History going back by skip_his, then num_frames future at stride skip."""
    rgb_id = [int(frame_now - i * skip_his) for i in range(num_history, 0, -1)]
    rgb_id.append(int(frame_now))
    rgb_id += [int(frame_now + i * skip) for i in range(1, num_frames)]
    return rgb_id


@torch.no_grad()
def replay_episode(
    model: CrtlWorld, pipeline, pipeline_cls, args: LiberoWMArgs,
    latent_gt: torch.Tensor,        # (T, 4, total_h, latent_w) fp32, at WM (5 Hz) rate
    action_norm: np.ndarray,        # (T, 7) normalized, at WM rate, 1:1 with latents
    text: str, device: torch.device,
    num_inference_steps: int, skip: int, max_chunks: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Closed-loop rollout. Returns (gt_stack, pred_stack, predicted_indices)."""
    T = int(latent_gt.shape[0])
    num_history, num_frames = args.num_history, args.num_frames
    skip_his = skip * 4  # matches dataset._build_frame_ids (history is 4x coarser)

    # rolled[t] = latent we believe represents frame t; seed with GT, overwrite with preds.
    rolled = [latent_gt[t].clone() for t in range(T)]

    first_anchor = num_history * skip_his  # so history indices stay >= 0
    step = (num_frames - 1) * skip         # chain: last pred frame -> next anchor
    if first_anchor + (num_frames - 1) * skip >= T:
        raise RuntimeError(f"episode too short: T={T}")

    predicted_indices: list[int] = []
    chunk_idx = 0
    while True:
        frame_now = first_anchor + chunk_idx * step
        if frame_now + (num_frames - 1) * skip >= T:
            break
        if max_chunks is not None and chunk_idx >= max_chunks:
            break

        rgb_id = build_frame_ids(frame_now, num_history, num_frames, skip, skip_his)
        state_id = [min(max(r, 0), T - 1) for r in rgb_id]  # action at same frame as latent

        history = torch.stack([rolled[rgb_id[i]] for i in range(num_history)], 0).unsqueeze(0).to(device)
        current = rolled[rgb_id[num_history]].unsqueeze(0).to(device)
        action = torch.tensor(action_norm[state_id], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            action_latent = model.action_encoder(
                action, [text], model.tokenizer, model.text_encoder, args.frame_level_cond)
            _, pred_latents = pipeline_cls.__call__(
                pipeline, image=current, text=action_latent,
                width=args.width, height=int(args.num_cams * args.height),
                num_frames=num_frames, history=history,
                num_inference_steps=num_inference_steps, decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale, fps=args.fps,
                motion_bucket_id=args.motion_bucket_id, mask=None,
                output_type="latent", return_dict=False,
                frame_level_cond=args.frame_level_cond, his_cond_zero=args.his_cond_zero,
                flow_map_type=args.flow_map_type, flow_map_loss_type=args.flow_map_loss_type,
            )
        pred = pred_latents[0].float().cpu()  # (num_frames, 4, total_h, latent_w)

        for k in range(num_frames):
            t_native = rgb_id[num_history + k]
            if t_native < T:
                rolled[t_native] = pred[k]
                predicted_indices.append(t_native)
        chunk_idx += 1

    if chunk_idx == 0:
        raise RuntimeError("no chunks produced")
    predicted_indices = sorted(set(predicted_indices))
    pred_stack = torch.stack([rolled[t] for t in predicted_indices], 0)
    gt_stack = torch.stack([latent_gt[t] for t in predicted_indices], 0)
    return gt_stack, pred_stack, predicted_indices


# ---------------------------------------------------------------------------
# Decode / IO
# ---------------------------------------------------------------------------


def decode_per_cam(latents: torch.Tensor, pipeline, args: LiberoWMArgs) -> np.ndarray:
    """(T, 4, num_cams*per_cam_h, latent_w) -> (T, num_cams*H, W, 3) uint8.

    Decode each camera view separately (avoids cross-view conv bleed), stack vertically."""
    device, dtype = pipeline.unet.device, pipeline.unet.dtype
    per_cam = einops.rearrange(latents, "t c (m h) w -> m t c h w", m=args.num_cams)
    decoded = []
    for cam_idx in range(args.num_cams):
        flat = per_cam[cam_idx].to(device=device, dtype=dtype)
        chunks = []
        for i in range(0, flat.shape[0], args.decode_chunk_size):
            chunk = flat[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            chunks.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        out = torch.cat(chunks, 0)  # (T, 3, H, W)
        out = ((out / 2.0 + 0.5).clamp(0, 1) * 255).float().cpu().numpy().astype(np.uint8)
        decoded.append(out.transpose(0, 2, 3, 1))  # (T, H, W, 3)
    return np.concatenate(decoded, axis=1)


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps, codec="h264",
                        ffmpeg_args="-movflags +faststart -pix_fmt yuv420p")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/wm/libero/checkpoint-30000.pt")
    p.add_argument("--data_root", default="data/libero_collected")
    p.add_argument("--stat_root", default="data/wm_training/libero_processed_5hz",
                   help="Where stat.json lives (the TRAINING stats; not the eval data).")
    p.add_argument("--output_dir", default=None, help="Default: <checkpoint dir>/replay.")
    p.add_argument("--suites", default=None, help="Comma-separated; default: all under data_root.")
    p.add_argument("--episode_id", default=None, help="Single episode; default: all in suite.")
    p.add_argument("--num_episodes", type=int, default=0, help="Cap episodes per suite (0=all).")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--target_hz", type=int, default=5, help="WM rate; data is strided down to this.")
    p.add_argument("--skip", type=int, default=1, help="Frame stride at WM rate (training used 1 or 2).")
    p.add_argument("--max_chunks", type=int, default=0, help="Cap autoregressive chunks (0=to episode end).")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--video_fps", type=int, default=5, help="5 = real time at the 5 Hz WM rate.")
    p.add_argument("--svd_model_path", default="external/stable-video-diffusion-img2vid")
    p.add_argument("--clip_model_path", default="external/clip-vit-base-patch32")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suites = ([s.strip() for s in a.suites.split(",") if s.strip()]
              if a.suites else list_suites(a.data_root))
    out_root = Path(a.output_dir) if a.output_dir else Path(a.checkpoint).resolve().parent / "replay"

    args = LiberoWMArgs(
        svd_model_path=a.svd_model_path, clip_model_path=a.clip_model_path,
        dataset_root_path=a.data_root, dataset_meta_info_path=a.stat_root,
        dataset_names="+".join(suites), dataset_cfgs="+".join(suites),
        prob=tuple([1.0 / len(suites)] * len(suites)),
        num_cams=2, num_frames=5, num_history=6, action_dim=7, down_sample=1,
        flow_map_type="flow_matching", distance_conditioning=False, tag="libero_replay",
    )

    print(f"[replay] loading checkpoint {a.checkpoint}")
    model = CrtlWorld(args)
    missing, unexpected = model.load_state_dict(
        torch.load(a.checkpoint, map_location="cpu"), strict=False)
    if missing:
        print(f"[replay] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[replay] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    model.to(device).eval()
    pipeline, pipeline_cls = model.pipeline, CtrlWorldDiffusionPipeline

    max_chunks = a.max_chunks if a.max_chunks > 0 else None
    summary = []
    for suite in suites:
        ep_ids = [a.episode_id] if a.episode_id else list_episode_ids(a.data_root, suite, a.split)
        if a.num_episodes > 0:
            ep_ids = ep_ids[: a.num_episodes]
        p01, p99 = _load_stat(a.stat_root, suite)

        for ep_id in ep_ids:
            latent_gt, action_native, text, native_fps = load_full_episode(
                a.data_root, suite, ep_id, args, a.split)

            # Stride 20 Hz -> target_hz (5 Hz) so spacing matches how the WM was trained.
            stride = max(1, round(native_fps / a.target_hz))
            latent5 = latent_gt[::stride]
            action5 = normalize_actions(action_native[::stride], p01, p99)
            print(f"[replay] {suite}/{ep_id}: {latent_gt.shape[0]}@{native_fps}Hz "
                  f"-> {latent5.shape[0]}@{a.target_hz}Hz  {text!r}")

            try:
                gt_stack, pred_stack, idxs = replay_episode(
                    model, pipeline, pipeline_cls, args, latent5, action5, text, device,
                    a.num_inference_steps, a.skip, max_chunks)
            except RuntimeError as e:
                print(f"[replay]   skipped: {e}")
                continue

            gt_vid = decode_per_cam(gt_stack, pipeline, args)
            pred_vid = decode_per_cam(pred_stack, pipeline, args)
            write_video(gt_vid, out_root / "gt" / f"{suite}_{ep_id}.mp4", a.video_fps)
            write_video(pred_vid, out_root / "pred" / f"{suite}_{ep_id}.mp4", a.video_fps)

            latent_mse = float(((gt_stack - pred_stack) ** 2).mean())
            pixel_mse = float(np.mean((gt_vid / 255.0 - pred_vid / 255.0) ** 2))
            psnr = float(10 * np.log10(1.0 / max(pixel_mse, 1e-12)))
            print(f"[replay]   {len(idxs)} frames  latent-MSE={latent_mse:.4f}  PSNR={psnr:.2f}dB")
            summary.append({
                "suite": suite, "episode": ep_id, "frames": len(idxs),
                "latent_mse": latent_mse, "pixel_mse": pixel_mse, "psnr_db": psnr, "text": text,
            })

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[replay] wrote {len(summary)} episodes -> {out_root}/{{gt,pred}}/  (+ replay_summary.json)")


if __name__ == "__main__":
    main()
