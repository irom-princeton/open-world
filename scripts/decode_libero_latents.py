"""Decode SVD VAE latents back to RGB video for a single LIBERO trajectory.

Example:
    uv run python scripts/decode_libero_latents.py \\
        --suite_dir data/libero_processed/libero_spatial \\
        --trajectory_id 25 \\
        --svd_path external/stable-video-diffusion-img2vid \\
        --output_dir outputs/libero_decoded
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def find_annotation(suite_dir: Path, episode_id: str) -> dict | None:
    for split in ("train", "val"):
        ann_path = suite_dir / "annotation" / split / f"{episode_id}.json"
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)
            ann["_split"] = split
            return ann
    return None


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
    vae,
    scale: float,
    chunk: int,
    device: str,
) -> np.ndarray:
    """latents: (T, 4, h, w) float16 -> (T, H, W, 3) uint8."""
    T = latents.shape[0]
    out_frames = []
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        z = latents[start:end].to(device=device, dtype=torch.float16) / scale
        decoded = vae.decode(z, num_frames=end - start).sample  # (b,3,H,W) in [-1,1]
        decoded = (decoded.float().clamp(-1, 1) + 1.0) * 127.5
        decoded = decoded.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        out_frames.append(decoded)
    return np.concatenate(out_frames, axis=0)


def save_video(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps, codec="libx264")
        return
    except Exception as e:
        print(f"  imageio mp4 write failed ({e}); falling back to PNG sequence")
    png_dir = path.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(png_dir / f"{i:04d}.png")
    print(f"  wrote PNG sequence to {png_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite_dir", type=Path, required=True,
                    help="Processed suite dir, e.g. data/libero_processed/libero_spatial")
    ap.add_argument("--trajectory_id", type=int, required=True)
    ap.add_argument("--svd_path", type=str,
                    default="external/stable-video-diffusion-img2vid")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/libero_decoded"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cams", nargs="+", default=["agentview", "wrist"])
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()

    episode_id = f"{args.trajectory_id:06d}"
    suite_name = args.suite_dir.name

    ann = find_annotation(args.suite_dir, episode_id)
    fps = int(ann["fps"]) if ann and "fps" in ann else 20
    if ann is not None:
        print(f"[{suite_name}/{episode_id}] split={ann.get('_split')} "
              f"language={ann.get('language_instruction', '')!r} fps={fps}")
    else:
        print(f"[{suite_name}/{episode_id}] no annotation found; using fps={fps}")

    from diffusers import AutoencoderKLTemporalDecoder

    print(f"Loading SVD VAE from {args.svd_path}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.svd_path, subfolder="vae", torch_dtype=torch.float16
    ).to(args.device)
    vae.eval()
    scale = vae.config.scaling_factor

    for cam in args.cams:
        latent_path = args.suite_dir / "latent_videos" / cam / f"{episode_id}.pt"
        if not latent_path.exists():
            print(f"  skip {cam}: missing {latent_path}")
            continue
        latents = torch.load(latent_path, map_location="cpu")
        if latents.dim() != 4:
            print(f"  skip {cam}: unexpected latent shape {tuple(latents.shape)}")
            continue
        print(f"  decoding {cam}: latent shape {tuple(latents.shape)}")
        frames = decode_latents(latents, vae, scale, chunk=args.chunk, device=args.device)
        out = args.output_dir / f"{suite_name}_{episode_id}_{cam}.mp4"
        save_video(frames, out, fps=fps)
        print(f"  wrote {out} ({frames.shape[0]} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
