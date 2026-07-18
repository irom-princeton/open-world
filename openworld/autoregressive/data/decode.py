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
        # Cast the VAE to the compute dtype so a bf16 deploy decodes in bf16
        # (~1.5x faster than fp32). dtype=float32 (the default) is a no-op for the
        # fp32-loaded VAE, so existing callers are unaffected.
        self.vae = vae.to(device=device, dtype=dtype).eval()
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

    @torch.no_grad()
    def decode_batched(self, latents: torch.Tensor) -> np.ndarray:
        """Batched decode. Normalized ``[B, z, Lf, h, w]`` -> RGB ``[B, T, H, W, 3]`` uint8.

        Numerically identical to looping :meth:`decode_video` over the batch -- batch
        samples are independent through the VAE -- but issues one set of kernels
        instead of ``B``, which matters when ``B`` (the number of views) decodes are
        otherwise serialized on one stream."""
        z = latents.to(self.device).float()                      # [B,z,Lf,h,w]
        z = z * self.latents_std + self.latents_mean
        x = self.vae.decode(z.to(self.dtype), return_dict=False)[0]   # [B,3,T,H,W] in [-1,1]
        x = ((x / 2.0 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return x.permute(0, 2, 3, 4, 1).cpu().numpy()            # [B,T,H,W,3]


def decode_stacked(decoder: VaeLatentDecoder, latents: torch.Tensor, num_cams: int) -> np.ndarray:
    """Height-stacked latents ``[L, C, V*h, w]`` -> RGB ``[T, V*H, W, 3]`` uint8.

    Decodes each view independently (no cross-view conv bleed) -- as one batched VAE
    call rather than a per-view Python loop -- and stacks the decoded views back along
    the pixel height axis."""
    import einops

    per_view = einops.rearrange(latents, "t c (m h) w -> m c t h w", m=num_cams)  # [m,z,Lf,h,w]
    views = decoder.decode_batched(per_view)                              # [m,T,H,W,3]
    return np.concatenate(list(views), axis=1)                            # stack along H
