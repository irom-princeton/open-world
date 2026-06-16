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

import pytest
import torch


def _dummy(layers=3, dim=48, C=16, cross=32):
    from openworld.autoregressive.backbones.dummy import DummyDiT
    return DummyDiT(in_channels=C, dim=dim, heads=4, layers=layers, cross_attn_dim=cross).eval()


def _wan(mode="cross_attn", cross=64):
    """Tiny real Wan DiT (random init, 2 layers) in a given action_cond_mode."""
    from openworld.autoregressive.backbones.wan import WanBackbone
    return WanBackbone.random_init(cross_attn_dim=cross, small=True, action_mode=mode).eval().float()


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
    assert set(logs) == {
        "gen_loss", "critic_loss", "gen_grad_norm", "critic_grad_norm",
        "dmd_grad_norm", "gen_x0_std", "gen_x0_absmean",
    }
    assert all(math.isfinite(v) for v in logs.values())


def test_bidirectional_forward_differs_from_causal():
    """forward_train(causal=False) gives full bidirectional attention -> a
    different result than the block-causal path (the L1b teacher mode)."""
    import torch

    from openworld.autoregressive.config import ARWMArgs
    from openworld.autoregressive.model import ARWorldModel
    torch.manual_seed(0)
    cfg = ARWMArgs(backbone="dummy", random_init_backbone=True, num_cams=1,
                   frames_per_block=2, width=32, height=32, action_dim=7, dtype=torch.float32)
    m = ARWorldModel(cfg).eval()
    F = 6
    x = torch.randn(1, F, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)
    t = torch.full((1,), 500.0)
    cond = m.encode_cond(torch.randn(1, F, cfg.action_dim), cfg_drop=False)
    with torch.no_grad():
        v_causal = m.forward_train(x, t, cond, frames_per_block=2, causal=True)
        v_bidir = m.forward_train(x, t, cond, frames_per_block=2, causal=False)
    assert v_causal.shape == x.shape == v_bidir.shape
    assert torch.isfinite(v_bidir).all()
    assert not torch.allclose(v_causal, v_bidir)   # masking actually changed attention


@pytest.mark.parametrize("causal", [True, False])
def test_midtrain_step_updates_model(causal):
    """A flow-matching mid-training step (student-init causal / teacher bidirectional)
    yields a finite loss and moves the weights."""
    import torch

    from openworld.autoregressive.config import ARWMArgs
    from openworld.autoregressive.distill.midtrain import DiffusionTrainer
    from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
    from openworld.autoregressive.model import ARWorldModel
    torch.manual_seed(0)
    cfg = ARWMArgs(backbone="dummy", random_init_backbone=True, num_cams=1,
                   frames_per_block=2, num_history_blocks=1, rollout_blocks=2,
                   width=32, height=32, action_dim=7, dtype=torch.float32)
    model = ARWorldModel(cfg)
    sched = FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                               warp=cfg.warp_denoising_step)
    tr = DiffusionTrainer(model, sched, frames_per_block=cfg.frames_per_block, causal=causal)
    F = (cfg.num_history_blocks + cfg.rollout_blocks) * cfg.frames_per_block
    before = next(model.parameters()).clone()
    latent = torch.randn(1, F, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)
    cond = model.encode_cond(torch.randn(1, F, cfg.action_dim), cfg_drop=True)
    logs = tr.train_step(latent, cond)
    assert set(logs) == {"loss", "grad_norm"}
    assert math.isfinite(logs["loss"]) and math.isfinite(logs["grad_norm"])
    assert not torch.allclose(next(model.parameters()), before)


def test_resume_roundtrip_restores_full_training_state(tmp_path):
    """A cut run resumes *exactly*: generator + online critic + both AdamW
    optimizer states + step round-trip through the resume bundle (CPU path)."""
    import torch

    from openworld.autoregressive.config import ARWMArgs
    from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
    from openworld.autoregressive.distill.self_forcing import SelfForcingTrainer
    from openworld.autoregressive.model import build_training_stack
    from openworld.autoregressive.train_self_forcing import (
        _gather_train_state, _latent_block_shape, _restore_train_state, _save_resume_atomic,
    )

    def _stack():
        cfg = ARWMArgs(backbone="dummy", random_init_backbone=True, num_cams=1,
                       frames_per_block=2, rollout_blocks=2, num_history_blocks=1,
                       denoising_step_list=(1000, 500), critic_steps_per_gen_step=1,
                       width=32, height=32, action_dim=7, dtype=torch.float32)
        gen, critic, teacher = build_training_stack(cfg)
        sched = FlowMatchScheduler(cfg.denoising_step_list,
                                   num_train_timestep=cfg.num_train_timestep, warp=cfg.warp_denoising_step)
        tr = SelfForcingTrainer(gen, critic, teacher, sched, frames_per_block=cfg.frames_per_block,
                                critic_steps=cfg.critic_steps_per_gen_step)
        return cfg, gen, critic, tr

    torch.manual_seed(0)
    cfg, gen, critic, tr = _stack()
    # take a couple of real steps so the optimizers have non-trivial moment buffers
    shape = _latent_block_shape(cfg, 1)
    T = cfg.frames_per_block * (cfg.rollout_blocks + cfg.num_history_blocks)
    for _ in range(2):
        cond = gen.encode_cond(torch.randn(1, T, cfg.action_dim), cfg_drop=True)
        tr.train_step(cond, gen.null_cond_like(cond), num_blocks=cfg.rollout_blocks, latent_block_shape=shape)
    state = _gather_train_state(gen, critic, tr.opt_g, tr.opt_c, step=2, distributed=False)
    path = _save_resume_atomic(state, tmp_path)
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()  # atomic: no temp left

    # fresh stack (different random init) then restore -> must match the saved one
    torch.manual_seed(123)
    cfg2, gen2, critic2, tr2 = _stack()
    g_before = next(gen2.parameters()).clone()
    step = _restore_train_state(path, gen2, critic2, tr2.opt_g, tr2.opt_c, distributed=False)
    assert step == 2
    assert not torch.allclose(next(gen2.parameters()), g_before)        # actually loaded
    for (k, a), (_, b) in zip(gen.state_dict().items(), gen2.state_dict().items()):
        assert torch.allclose(a, b), f"gen param {k} mismatch after resume"
    for (k, a), (_, b) in zip(critic.state_dict().items(), critic2.state_dict().items()):
        assert torch.allclose(a, b), f"critic param {k} mismatch after resume"
    # optimizer momentum buffers restored
    s1 = tr.opt_g.state_dict()["state"]; s2 = tr2.opt_g.state_dict()["state"]
    assert s1 and s2 and set(s1) == set(s2)
    for pid in s1:
        assert torch.allclose(s1[pid]["exp_avg"], s2[pid]["exp_avg"])


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
    assert set(logs) == {
        "gen_loss", "critic_loss", "gen_grad_norm", "critic_grad_norm",
        "dmd_grad_norm", "gen_x0_std", "gen_x0_absmean",
    }
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


# ---------------------------------------------------------------------------
# action_cond_mode: the four action-conditioning modes (see config.ARWMArgs and
# backbones/wan.py). Exercised on a tiny real Wan DiT on CPU.
# ---------------------------------------------------------------------------
_ACTION_MODES = ["cross_attn", "cross_attn_pe", "cross_attn_aligned", "adaln"]


@pytest.mark.parametrize("mode", _ACTION_MODES)
def test_wan_action_mode_rollout_matches_masked(mode):
    """The KV-cache rollout reproduces the masked forward for *every* action mode
    — i.e. the per-frame aligned mask / adaln modulation / per-block cond slice
    stay consistent between training and the autoregressive rollout."""
    torch.manual_seed(0)
    C, H, W, fpb, nblocks = 16, 8, 8, 2, 4
    bb = _wan(mode)
    x = torch.randn(1, fpb * nblocks, C, H, W)
    cond = torch.randn(1, fpb * nblocks, bb.cross_attn_dim)
    t = torch.zeros(1)
    with torch.no_grad():
        full = bb.forward_train(x, t, cond, frames_per_block=fpb, causal=True)
        cache = bb.make_kv_cache()
        rolled = torch.cat([
            bb.forward_cached(x[:, b*fpb:(b+1)*fpb], t, cond, kv_cache=cache, start_frame=b*fpb)
            for b in range(nblocks)
        ], dim=1)
    assert full.shape == x.shape
    assert (full - rolled).abs().max().item() < 1e-4


def test_wan_aligned_binds_action_to_its_own_frame():
    """cross_attn_aligned: swapping the actions of frames 2 and 3 changes ONLY
    those frames' outputs; the baseline cross_attn (a positionally-unordered bag)
    is unchanged by the same swap — the exact controllability failure this fixes."""
    torch.manual_seed(0)
    C, H, W, F, fpb = 16, 8, 8, 4, 2
    x = torch.randn(1, F, C, H, W)
    t = torch.full((1,), 500.0)
    cond = torch.randn(1, F, 64)
    swapped = cond.clone()
    swapped[:, [2, 3]] = cond[:, [3, 2]]

    aligned = _wan("cross_attn_aligned")
    with torch.no_grad():
        a = aligned.forward_train(x, t, cond, frames_per_block=fpb, causal=True)
        b = aligned.forward_train(x, t, swapped, frames_per_block=fpb, causal=True)
    per_frame = (a - b).abs().flatten(2).amax(-1)[0]
    assert per_frame[0] < 1e-5 and per_frame[1] < 1e-5      # frames 0,1 untouched
    assert per_frame[2] > 1e-3 and per_frame[3] > 1e-3      # frames 2,3 bound to swapped actions

    base = _wan("cross_attn")
    with torch.no_grad():
        a = base.forward_train(x, t, cond, frames_per_block=fpb, causal=True)
        b = base.forward_train(x, t, swapped, frames_per_block=fpb, causal=True)
    assert (a - b).abs().max().item() < 1e-5               # bag of actions: permutation-invariant


def test_wan_adaln_is_cfg_consistent_and_active():
    """adaln: a zero condition is the deterministic unconditional forward (so CFG's
    null branch is exact), and a non-zero action actually changes the output (the
    AdaLN modulation path is wired)."""
    torch.manual_seed(0)
    C, H, W, F = 16, 8, 8, 4
    bb = _wan("adaln")
    x = torch.randn(1, F, C, H, W)
    t = torch.full((1,), 500.0)
    cond = torch.randn(1, F, 64)
    zero = torch.zeros(1, F, 64)
    with torch.no_grad():
        out_action = bb.forward_train(x, t, cond, frames_per_block=2, causal=True)
        out_null_1 = bb.forward_train(x, t, zero, frames_per_block=2, causal=True)
        out_null_2 = bb.forward_train(x, t, zero, frames_per_block=2, causal=True)
    assert (out_null_1 - out_null_2).abs().max().item() < 1e-6   # deterministic null branch
    assert (out_action - out_null_1).abs().max().item() > 1e-3   # action changes the output


def test_conditioner_pe_makes_frames_positional():
    """cross_attn_pe: with a (trained, non-zero) temporal PE, the *same* action
    vector at two different frames yields different cond tokens — the positional
    signal the baseline lacks. The baseline conditioner gives identical tokens."""
    from openworld.autoregressive.conditioning.action import ActionConditioner
    torch.manual_seed(0)
    same = torch.zeros(1, 4, 7)                    # identical action every frame
    pe = ActionConditioner(action_dim=7, cross_attn_dim=64, mode="cross_attn_pe").eval()
    with torch.no_grad():
        pe.temporal_pe.normal_()                   # PE is zero-init; simulate a trained PE
        cond = pe(same, cfg_drop=False)
    assert (cond[:, 0] - cond[:, 1]).abs().max().item() > 1e-3   # frames are positionally distinct

    base = ActionConditioner(action_dim=7, cross_attn_dim=64, mode="cross_attn").eval()
    with torch.no_grad():
        cond_b = base(same, cfg_drop=False)
    assert (cond_b[:, 0] - cond_b[:, 1]).abs().max().item() < 1e-6   # identical action -> identical token


def test_non_wan_backbone_rejects_nonbaseline_mode():
    """dummy/cosmos/svd implement only the baseline cross-attn injection; asking
    for another mode must fail loudly rather than silently ignore the action."""
    from openworld.autoregressive.backbones import build_backbone
    from openworld.autoregressive.config import ARWMArgs
    cfg = ARWMArgs(backbone="dummy", random_init_backbone=True, action_cond_mode="adaln")
    with pytest.raises(NotImplementedError):
        build_backbone(cfg)


def test_slice_cond_to_frames_aligns_generated_window():
    """The DMD score path scores the GENERATED clip (frames [hist:hist+N]) but was
    handed the full-window cond, so cross_attn_aligned tied generated frame j to
    action j instead of action hist+j. ``slice_cond_to_frames`` must reproduce the
    cond of the truly-aligned actions, and differ from the unsliced (buggy) cond."""
    C, H, W, fpb = 16, 8, 8, 2
    hist, n_gen = 4, 8
    win = hist + n_gen
    aligned = _wan("cross_attn_aligned")
    torch.manual_seed(0)
    x = torch.randn(1, n_gen, C, H, W)          # the generated clip (positions hist..hist+n_gen)
    t = torch.full((1,), 500.0)
    cond_full = torch.randn(1, win, 64)         # full-window action cond

    sliced = aligned.slice_cond_to_frames(cond_full, hist, n_gen)
    assert sliced.shape[1] == n_gen
    # slicing == taking the generated window's own actions
    assert torch.equal(sliced, cond_full[:, hist:hist + n_gen])

    with torch.no_grad():
        v_sliced = aligned.forward_train(x, t, sliced, frames_per_block=fpb, causal=False)
        v_buggy = aligned.forward_train(x, t, cond_full, frames_per_block=fpb, causal=False)
    # the off-by-history misalignment genuinely changes the score
    assert (v_sliced - v_buggy).abs().max().item() > 1e-4

    # global cond modes must be unaffected (no-op slice)
    base = _wan("cross_attn")
    assert torch.equal(base.slice_cond_to_frames(cond_full, hist, n_gen), cond_full)
