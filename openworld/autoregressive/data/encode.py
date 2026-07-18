"""RGB -> 16-channel Wan-VAE latent encoding.

Wraps ``AutoencoderKLWan``. Handles the pixel normalisation, the 4x temporal
compression (``T`` RGB frames -> ``(T-1)//4 + 1`` latent frames), and Wan's
per-channel latent normalisation (``z = (encode(x) - mean) / std``, the same
convention ``WanPipeline`` uses) so the saved latents are in the space the DiT
expects.
"""

from __future__ import annotations

import numpy as np
import torch


class VaeLatentEncoder:
    def __init__(self, vae, *, device="cuda", dtype=torch.float32):
        self.vae = vae.to(device).eval()
        self.device = device
        self.dtype = dtype
        z = vae.config.z_dim if hasattr(vae.config, "z_dim") else 16
        self.z_dim = z
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, z, 1, 1, 1)
        self.latents_mean = mean.to(device)
        self.latents_std = std.to(device)

    @property
    def temporal_factor(self) -> int:
        # Wan VAE temporal downsample (frames -> (T-1)//factor + 1)
        return 2 ** sum(self.vae.config.temperal_downsample) if hasattr(self.vae.config, "temperal_downsample") else 4

    @torch.no_grad()
    def encode_video(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """``[T, H, W, 3]`` uint8 -> normalized latent ``[z, Lf, H/8, W/8]`` (fp16, cpu)."""
        x = torch.from_numpy(rgb_uint8).to(self.device).float() / 127.5 - 1.0   # [T,H,W,3] in [-1,1]
        x = x.permute(3, 0, 1, 2).unsqueeze(0).to(self.dtype)                    # [1,3,T,H,W]
        dist = self.vae.encode(x).latent_dist
        z = dist.mode().float()                                                 # [1,z,Lf,h,w]
        z = (z - self.latents_mean) / self.latents_std
        return z[0].to(torch.float16).cpu()

    def latent_frames(self, num_rgb_frames: int) -> int:
        f = self.temporal_factor
        return (num_rgb_frames - 1) // f + 1


def align_actions_to_latent(actions: np.ndarray, num_latent_frames: int) -> np.ndarray:
    """Subsample a per-RGB-frame action sequence ``[T, A]`` to one action per
    latent frame ``[Lf, A]``.

    The Wan VAE compresses time 4x: latent frame ``i>0`` is decoded from the RGB
    group ``{4i-3 .. 4i}`` whose visual centroid sits at ``4i-1.5``. A naive
    ``round(linspace(0, T-1, Lf))`` maps latent ``i`` to RGB frame ``4i`` -- the
    *last* frame of that group -- so the conditioning pose *leads* the imagery it
    controls by ~1.4 frames (confirmed empirically). We instead sample the group
    *center* index (``4i -> 4i-1``), which removes the systematic lead. Latent frame 0 maps
    to RGB frame 0 (no preceding group). We pick the center *index* rather than
    averaging over the group because Euler-XYZ angles cannot be averaged safely
    across the +/-pi wrap.
    """
    T = actions.shape[0]
    if num_latent_frames <= 1:
        return actions[:1]
    last = np.round(np.linspace(0, T - 1, num_latent_frames)).astype(int)  # [0, ~4, ~8, ...]
    f = int(last[1])                                                       # temporal factor (~4)
    idx = last.copy()
    idx[1:] = np.clip(last[1:] - (f - 1) // 2, 0, T - 1)                   # 4i -> 4i-1 (group center)
    return actions[idx]
