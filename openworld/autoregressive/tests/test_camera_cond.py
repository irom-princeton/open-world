"""camera_cond geometric conditioning: render correctness + the widened-patch-embed
teacher-forcing (clean_x) channel fix.

Covers the bugs found in the sbwr/cc runs:
  * non-band views must read as the grey BACKGROUND, not 0.0 (consistent "no band");
  * the band is rendered supersampled then downsampled (anti-aliased, sub-pixel), so
    a moved EEF shifts the mark continuously rather than snapping to a 1px cell;
  * ``clean_x`` (teacher-forcing / CD) must go through the widened conv WITH the extra
    pixel/camera-cond channels appended, else patch_embedding(clean_x) sees the wrong
    in_channels and crashes.
"""
import dataclasses
import importlib

import numpy as np
import torch

from openworld.autoregressive.conditioning.camera_cond import render_camera_cond, _BG
from openworld.autoregressive.model import ARWorldModel


def _rec(L=6, V=4):
    """Minimal synthetic sidecar: 2 scene views (band_valid) + 2 wrist (fisheye idx 2,3)."""
    rng = np.random.default_rng(0)
    pose = np.zeros((L, 20), dtype=np.float32)
    pose[:, 0:3] = np.array([0.3, 0.0, 0.4]) + rng.normal(0, 0.02, (L, 3))   # L arm xyz
    pose[:, 3:9] = np.array([1, 0, 0, 0, 1, 0])                              # identity rot6d
    pose[:, 9] = np.linspace(0.0, 0.1, L)                                    # L gripper open->..
    pose[:, 10:13] = np.array([-0.3, 0.0, 0.4]) + rng.normal(0, 0.02, (L, 3))
    pose[:, 13:19] = np.array([1, 0, 0, 0, 1, 0])
    pose[:, 19] = np.linspace(0.1, 0.0, L)
    c2w = np.tile(np.eye(4, dtype=np.float32), (L, V, 1, 1))
    c2w[:, :, 2, 3] = -1.2                                                    # camera behind scene
    K = np.tile(np.array([[20, 0, 20], [0, 20, 12], [0, 0, 1]], np.float32), (V, 1, 1))
    band_valid = np.array([True, True, False, False])
    return {"pose": pose, "c2w": c2w, "K": K, "band_valid": band_valid}


def test_no_band_views_fill_with_background_not_zero():
    rec = _rec(); h, w = 24, 40
    out = render_camera_cond(rec, sel=(1, 2, 3), h=h, w=w, band_scale=True,
                             draw_band=True, draw_sticks=False, wrist_band=False)
    assert tuple(out.shape) == (6, 9, 3 * h, w)
    bg = _BG * 2.0 - 1.0
    # views 2,3 are wrist (band_valid False, wrist_band off) -> band channels == bg exactly
    for vi in (1, 2):                                     # stacked positions of sel[1],sel[2]
        region = out[:, 0:3, vi * h:(vi + 1) * h]
        assert torch.allclose(region, torch.full_like(region, bg)), \
            f"no-band view fill {region.min():.3f}..{region.max():.3f} != bg {bg:.3f}"


def test_band_is_antialiased_not_hard_squares():
    rec = _rec(); h, w = 24, 40
    out = render_camera_cond(rec, sel=(0,), h=h, w=w, band_scale=True,
                             draw_band=True, draw_sticks=False)
    band = out[0, 0:3]                                    # scene view frame 0
    bg = _BG * 2.0 - 1.0
    diff = (band - bg).abs().amax(0)                      # per-pixel deviation from bg
    on = diff[diff > 1e-3]
    assert on.numel() > 0, "no band drawn"
    # supersample+downsample => a spread of intensities (edges), not a single hard value
    assert on.unique().numel() >= 4, "band looks like a hard-edged raster (no anti-aliasing)"


def test_wrist_band_draws_all_views():
    rec = _rec(); h, w = 24, 40
    off = render_camera_cond(rec, sel=(0, 2, 3), h=h, w=w, wrist_band=False, draw_sticks=False)
    on = render_camera_cond(rec, sel=(0, 2, 3), h=h, w=w, wrist_band=True, draw_sticks=False)
    bg = _BG * 2.0 - 1.0
    for vi in (1, 2):                                     # the two wrist views in the stack
        r_off = off[:, 0:3, vi * h:(vi + 1) * h]
        r_on = on[:, 0:3, vi * h:(vi + 1) * h]
        assert torch.allclose(r_off, torch.full_like(r_off, bg))       # off: pure bg
        assert not torch.allclose(r_on, torch.full_like(r_on, bg))     # on: band present


def test_frame_idx_gather_matches_contiguous_and_supports_sparse():
    # frame_idx gather must (a) match the contiguous t0:t1 path for a contiguous range,
    # and (b) render arbitrary/sparse frame sets (WEAVER sparse history) without error.
    rec = _rec(L=8); h, w = 24, 40
    contig = render_camera_cond(rec, sel=(0, 2), h=h, w=w, t0=2, t1=6,
                                wrist_band=True, draw_sticks=False)
    gathered = render_camera_cond(rec, sel=(0, 2), h=h, w=w,
                                  frame_idx=[2, 3, 4, 5], wrist_band=True, draw_sticks=False)
    assert torch.allclose(contig, gathered), "frame_idx gather != contiguous t0:t1 for a contiguous range"
    # sparse strided + clamped-repeat history (like frame_pos [0,0,4,8] -> idx [0,0,4,8])
    sparse = render_camera_cond(rec, sel=(0, 2, 3), h=h, w=w,
                                frame_idx=[0, 0, 4, 7], wrist_band=True, draw_sticks=False)
    assert tuple(sparse.shape) == (4, 9, 3 * h, w)
    assert torch.isfinite(sparse).all()
    # duplicate index -> identical rendered frames (clamped repeat is a true repeat)
    assert torch.allclose(sparse[0], sparse[1])


def _tiny_camcond_cfg():
    # The published camera_cond student's inference config (camera_cond=True, 9 extra
    # channels). random-init + global cross_attn keeps the test about the widened conv,
    # not action alignment.
    base = importlib.import_module(
        "configs.inference.ar_wan_student_3view_bimanual").get_args()
    return dataclasses.replace(base, random_init_backbone=True, action_cond_mode="cross_attn")


def test_clean_x_with_pixel_cond_forward_runs():
    """Regression: teacher-forcing clean_x through the widened patch-embed used to crash
    because clean_x never received the extra camera_cond channels."""
    cfg = _tiny_camcond_cfg()
    assert cfg.extra_in_channels == cfg.camera_cond_channels == 9
    torch.manual_seed(0)
    model = ARWorldModel(cfg)
    B, F, H, W = 1, 4, 8, 8
    lat = torch.randn(B, F, 16, H, W)
    clean = torch.randn(B, F, 16, H, W)
    pix = torch.randn(B, F, 9, H, W)                     # [B,F,K,H,W], H,W match the latent
    cond = model.encode_cond(torch.randn(B, F, cfg.action_dim), cfg_drop=False)
    # clean_x path (teacher forcing): must not raise on the widened conv.
    out = model.backbone.forward_train(lat, torch.zeros(B), cond, frames_per_block=1,
                                       causal=True, clean_x=clean, pixel_cond=pix)
    assert out.shape == lat.shape
    # and the plain (no clean_x) pixel_cond path still works.
    out2 = model.backbone.forward_train(lat, torch.zeros(B), cond, frames_per_block=1,
                                        causal=True, pixel_cond=pix)
    assert out2.shape == lat.shape


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
