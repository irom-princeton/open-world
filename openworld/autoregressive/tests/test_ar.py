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


def _cosmos(cross=32):
    """Tiny real Cosmos DiT (random init, 2 layers) — CPU/CI friendly."""
    from openworld.autoregressive.backbones.cosmos_predict2 import CosmosBackbone
    return CosmosBackbone.random_init(cross_attn_dim=cross, small=True).eval()


def test_cosmos_kv_cache_matches_masked_forward():
    """The Cosmos RoPE-offset cached rollout reproduces the masked forward exactly
    (the OmniDreams `start_frame_for_rope` slice, ported to diffusers' Cosmos)."""
    torch.manual_seed(0)
    C, H, W, fpb, nblocks = 16, 8, 8, 2, 4
    dit = _cosmos()
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


def test_cosmos_sliding_window_rollout():
    torch.manual_seed(1)
    C, H, W, fpb, nblocks, win = 16, 8, 8, 2, 5, 2
    dit = _cosmos()
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


def test_cosmos_self_forcing_smoke_updates_generator():
    """The full few-step self-forcing / DMD train step runs with the Cosmos
    backbone (previously blocked: cached rollout raised NotImplementedError)."""
    from openworld.autoregressive.train_self_forcing import run_smoke
    logs = run_smoke(steps=1, backbone="cosmos_predict2_2b")
    assert set(logs) == {"gen_loss", "critic_loss"}
    assert all(math.isfinite(v) for v in logs.values())


def _write_synthetic_latents(root, split, n_eps, *, V=3, C=16, Lf=10, h=3, w=4):
    import json, os
    os.makedirs(os.path.join(root, split), exist_ok=True)
    samples = []
    for e in range(n_eps):
        torch.save(
            {"latent": torch.randn(V, C, Lf, h, w).half(),
             "action": torch.randn(Lf, 7), "text": f"task {e}", "num_latent_frames": Lf},
            os.path.join(root, split, f"ep{e}.pt"),
        )
        samples.append({"ep_id": f"ep{e}", "num_latent_frames": Lf})
    json.dump(samples, open(os.path.join(root, f"{split}_sample.json"), "w"))
    json.dump({"state_01": [-1.0] * 7, "state_99": [1.0] * 7},
              open(os.path.join(root, "stats.json"), "w"))


def test_ar_latent_dataset_roundtrip(tmp_path):
    from openworld.autoregressive.config import ARWMArgs
    from openworld.autoregressive.data.dataset import ARLatentDataset
    V, C, h, w = 3, 16, 3, 4
    _write_synthetic_latents(str(tmp_path), "train", n_eps=4, V=V, C=C, Lf=10, h=h, w=w)
    cfg = ARWMArgs(backbone="dummy", num_cams=V, frames_per_block=2,
                   num_history_blocks=1, rollout_blocks=2, latent_root=str(tmp_path))
    ds = ARLatentDataset(cfg, "train")
    clip = (cfg.num_history_blocks + cfg.rollout_blocks) * cfg.frames_per_block  # 6
    assert len(ds) == 4
    s = ds[0]
    # height-stacked latent: [L, C, V*h, w]
    assert tuple(s["latent"].shape) == (clip, C, V * h, w)
    assert tuple(s["action"].shape) == (clip, 7)
    assert s["action"].abs().max() <= 1.0 + 1e-6   # normalized to [-1, 1]
    assert isinstance(s["text"], str)


def test_droid_format_adapter():
    """The droid_ctrl_world adapter reads the real annotation layout if present."""
    import os
    from openworld.autoregressive.data.formats import build_format
    root = "/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world"
    if not os.path.isdir(os.path.join(root, "annotation", "train")):
        import pytest
        pytest.skip("droid_ctrl_world dataset not present")
    fmt = build_format("droid_ctrl_world", root, num_views=3)
    eps = fmt.list_episodes("train")
    assert len(eps) > 0
    ep = fmt.load_episode(eps[0], "train")
    assert ep.actions.shape[1] == 7 and len(ep.video_paths) == 3
    assert ep.actions.shape[0] == ep.num_frames


def test_scheduler_preserves_dtype():
    """add_noise / x0_from_velocity must keep the latent's dtype: sigma is built
    in fp32, and if it promotes a bf16 latent back to fp32 that fp32 leaks into
    the loss graph and breaks the bf16 backward (regression guard)."""
    from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler((1000, 500), num_train_timestep=1000)
    x0 = torch.randn(2, 2, 16, 4, 4, dtype=torch.bfloat16)
    eps = torch.randn_like(x0)
    sigma = sched.random_sigma(2, x0.device, lo=0.02, hi=0.98)   # fp32
    x_sigma = sched.add_noise(x0, eps, sigma)
    assert x_sigma.dtype == torch.bfloat16
    v = torch.randn_like(x0)
    assert sched.x0_from_velocity(x_sigma, v, sigma).dtype == torch.bfloat16


def test_config_geometry():
    from openworld.autoregressive.config import ARWMArgs
    cfg = ARWMArgs(backbone="wan_1_3b", num_cams=3, height=320, width=320,
                   multiview_layout="height_stack")
    assert cfg.in_channels == 16 and cfg.cross_attn_dim == 4096
    assert cfg.latent_h_per_cam == 40 and cfg.latent_h_total == 120 and cfg.latent_w == 40


def test_autocast_dtype_derivation():
    """autocast only kicks in when it would change precision vs the param dtype:
    fp32 params + bf16 mixed_precision -> bf16 autocast; uniform dtypes -> None."""
    from openworld.autoregressive.config import ARWMArgs
    fp32_bf16 = ARWMArgs(backbone="dummy", dtype=torch.float32, mixed_precision="bf16")
    assert fp32_bf16.autocast_dtype == torch.bfloat16          # fp32 master + bf16 compute
    assert ARWMArgs(backbone="dummy", dtype=torch.bfloat16, mixed_precision="bf16").autocast_dtype is None
    assert ARWMArgs(backbone="dummy", dtype=torch.float32, mixed_precision="no").autocast_dtype is None
