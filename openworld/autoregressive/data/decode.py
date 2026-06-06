"""16-channel Wan-VAE latent -> RGB decoding (inverse of ``encode.py``).

The encoder stores Wan's per-channel normalized latents ``z = (mode - mean) / std``
(the convention ``WanPipeline`` uses). Decoding inverts that — ``mode = z * std + mean``
— then runs the VAE decoder, which reverses the 4x temporal compression
(``Lf`` latent frames -> ``(Lf - 1) * 4 + 1`` RGB frames). This mirrors
``WanPipeline.__call__``'s post-process exactly.

Multi-view latents are height-stacked (``[L, C, V*h, w]``); decode each view
*separately* (``decode_stacked``) so the VAE's spatial convolutions don't bleed
across the view boundaries, then re-stack the decoded frames along H.
"""

from __future__ import annotations

import numpy as np
import torch


class VaeLatentDecoder:
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

    @torch.no_grad()
    def decode_video(self, latent: torch.Tensor) -> np.ndarray:
        """Normalized latent ``[z, Lf, h, w]`` -> RGB ``[T, H, W, 3]`` uint8."""
        z = latent.to(self.device).float().unsqueeze(0)          # [1,z,Lf,h,w]
        z = z * self.latents_std + self.latents_mean             # un-normalize
        x = self.vae.decode(z.to(self.dtype), return_dict=False)[0]   # [1,3,T,H,W] in [-1,1]
        x = ((x / 2.0 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return x[0].permute(1, 2, 3, 0).cpu().numpy()            # [T,H,W,3]


def decode_stacked(decoder: VaeLatentDecoder, latents: torch.Tensor, num_cams: int) -> np.ndarray:
    """Height-stacked latents ``[L, C, V*h, w]`` -> RGB ``[T, V*H, W, 3]`` uint8.

    Decodes each view independently (no cross-view conv bleed) and stacks the
    decoded views back along the pixel height axis."""
    import einops

    per_view = einops.rearrange(latents, "t c (m h) w -> m c t h w", m=num_cams)
    views = [decoder.decode_video(per_view[v]) for v in range(num_cams)]  # each [T,H,W,3]
    return np.concatenate(views, axis=1)                                  # stack along H
