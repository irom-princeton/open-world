"""Unit tests for the autoregressive world model core.

These run on CPU with the DummyDiT — no downloaded weights — and assert the
properties the whole design rests on:

* the KV-cache rollout reproduces the block-causal masked forward exactly
  (unbounded and sliding-window), i.e. the autoregressive memory is correct;
* the few-step self-forcing / DMD training step runs and updates the generator;
* multi-view packing round-trips; config geometry is consistent.

Run: ``.venv/bin/python -m pytest openworld/autoregressive/tests/test_ar.py -q``
"""

from __future__ import annotations

import math

import torch


def _dummy(layers=3, dim=48, C=16, cross=32):
    from openworld.autoregressive.backbones.dummy import DummyDiT
    return DummyDiT(in_channels=C, dim=dim, heads=4, layers=layers, cross_attn_dim=cross).eval()


def test_kv_cache_matches_masked_forward():
    torch.manual_seed(0)
    C, H, W, fpb, nblocks = 16, 8, 8, 2, 4
    dit = _dummy(C=C)
    x = torch.randn(1, fpb * nblocks, C, H, W)
    cond = torch.randn(1, 5, 32)
    t = torch.zeros(1)
    with torch.no_grad():
        full = dit.forward_train(x, t, cond, frames_per_block=fpb)
        cache = dit.make_kv_cache()
        rolled = torch.cat([
            dit.forward_cached(x[:, b*fpb:(b+1)*fpb], t, cond, kv_cache=cache, start_frame=b*fpb)
            for b in range(nblocks)
        ], dim=1)
    assert (full - rolled).abs().max().item() < 1e-4


def test_sliding_window_rollout():
    torch.manual_seed(1)
    C, H, W, fpb, nblocks, win = 16, 8, 8, 2, 5, 2
    dit = _dummy(C=C)
    x = torch.randn(1, fpb * nblocks, C, H, W)
    cond = torch.randn(1, 4, 32)
    t = torch.zeros(1)
    with torch.no_grad():
        full = dit.forward_train(x, t, cond, frames_per_block=fpb, window=win)
        cache = dit.make_kv_cache(max_blocks=win)
        rolled = torch.cat([
            dit.forward_cached(x[:, b*fpb:(b+1)*fpb], t, cond, kv_cache=cache, start_frame=b*fpb)
            for b in range(nblocks)
        ], dim=1)
    assert (full - rolled).abs().max().item() < 1e-4


def test_block_causal_mask_shape():
    from openworld.autoregressive.causal.mask import block_ids_for_video, dense_block_causal_mask
    bids = block_ids_for_video(num_frames=6, tokens_per_frame=4, frames_per_block=2)
    assert bids.tolist() == [0]*8 + [1]*8 + [2]*8
    m = dense_block_causal_mask(bids, bids)
    # block 0 query (token 0) attends only to block 0 keys (first 8)
    assert m[0, :8].all() and not m[0, 8:].any()
    # last query attends to everything
    assert m[-1].all()


def test_multiview_packer_roundtrip():
    from openworld.autoregressive.conditioning.multiview import ViewPacker
    B, V, F, C, h, w = 2, 3, 4, 16, 5, 6
    views = torch.randn(B, V, F, C, h, w)
    for layout in ("height_stack", "sequence_pack"):
        p = ViewPacker(layout, V)
        packed = p.pack(views)
        back = p.unpack(packed, F)
        assert back.shape == views.shape
        assert torch.allclose(back, views), layout


def test_self_forcing_smoke_updates_generator():
    from openworld.autoregressive.train_self_forcing import run_smoke
    logs = run_smoke(steps=1)
    assert set(logs) == {"gen_loss", "critic_loss"}
    assert all(math.isfinite(v) for v in logs.values())


def test_config_geometry():
    from openworld.autoregressive.config import ARWMArgs
    cfg = ARWMArgs(backbone="wan_1_3b", num_cams=3, height=320, width=320,
                   multiview_layout="height_stack")
    assert cfg.in_channels == 16 and cfg.cross_attn_dim == 4096
    assert cfg.latent_h_per_cam == 40 and cfg.latent_h_total == 120 and cfg.latent_w == 40
