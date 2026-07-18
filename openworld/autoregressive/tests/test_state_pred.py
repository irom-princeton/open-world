"""Auxiliary state-prediction head (WEAVER-style) — training path + the
backward-compatibility guarantee (off by default => no head, no behavior change).

Builds tiny random-init models from the real config dataclass so all cfg fields
are exercised the way a config file would set them.
"""
import dataclasses

import torch

from openworld.autoregressive.config import ARWMArgs
from openworld.autoregressive.distill.midtrain import DiffusionTrainer
from openworld.autoregressive.distill.scheduler import FlowMatchScheduler
from openworld.autoregressive.model import ARWorldModel


def _tiny(**over):
    return dataclasses.replace(
        ARWMArgs(random_init_backbone=True, action_cond_mode="cross_attn_aligned",
                 frames_per_block=1, num_history_blocks=4, rollout_blocks=4),
        **over)


def _sched(cfg):
    return FlowMatchScheduler(cfg.denoising_step_list, num_train_timestep=cfg.num_train_timestep,
                              warp=cfg.warp_denoising_step)


def test_state_head_trains_and_grads_flow():
    cfg = _tiny(state_pred=True, state_pred_dim=16)
    torch.manual_seed(0)
    model = ARWorldModel(cfg)
    assert hasattr(model.backbone, "state_head")
    assert getattr(model.backbone.transformer, "_stash_state_feat", False) is True
    tr = DiffusionTrainer(model, _sched(cfg), frames_per_block=1, causal=True, state_pred_weight=0.1)
    B, F = 1, 4
    lat = torch.randn(B, F, 16, 8, 8)
    cond = model.encode_cond(torch.randn(B, F, cfg.action_dim), cfg_drop=False)
    tr.forward_backward(lat, cond, state=torch.randn(B, F, cfg.state_pred_dim))
    ps = model.predict_state()
    assert ps.shape == (B, F, cfg.state_pred_dim)
    gsum = sum(p.grad.abs().sum().item() for p in model.backbone.state_head.parameters()
               if p.grad is not None)
    assert gsum > 0 and tr._last_state_loss > 0        # aux loss reaches the head
    tr.optimizer_step()


def test_predict_state_handles_bf16_feat():
    # The real backbone runs under bf16 autocast, so the stashed feature is bf16 while
    # state_head holds fp32 master weights. predict_state must cast, not crash.
    cfg = _tiny(state_pred=True, state_pred_dim=8)
    torch.manual_seed(0)
    model = ARWorldModel(cfg)
    dim = model.backbone.transformer.config.num_attention_heads * model.backbone.transformer.config.attention_head_dim
    model.backbone.transformer._state_feat = torch.randn(1, 4, dim, dtype=torch.bfloat16)
    out = model.predict_state()                         # must not raise (bf16 feat, fp32 head)
    assert out.shape == (1, 4, cfg.state_pred_dim)


def test_state_pred_off_is_noop():
    cfg = _tiny()                                       # state_pred defaults off
    torch.manual_seed(0)
    model = ARWorldModel(cfg)
    assert not hasattr(model.backbone, "state_head")
    assert getattr(model.backbone.transformer, "_stash_state_feat", False) is False
    tr = DiffusionTrainer(model, _sched(cfg), frames_per_block=1, causal=True, state_pred_weight=0.0)
    lat = torch.randn(1, 4, 16, 8, 8)
    cond = model.encode_cond(torch.randn(1, 4, cfg.action_dim), cfg_drop=False)
    loss = tr.forward_backward(lat, cond, state=None)   # no aux, no crash
    assert loss == loss


def test_history_noise_perturbs_only_history_and_trains():
    """WEAVER-style context noise: with history_noise_std>0 the causal forward runs and
    grads flow (the noise touches only the first `history_frames` frames)."""
    cfg = _tiny()
    torch.manual_seed(0)
    model = ARWorldModel(cfg)
    F, hf = 8, 4                                          # 4 history + 4 rollout frames
    lat = torch.randn(1, F, 16, 8, 8)
    cond = model.encode_cond(torch.randn(1, F, cfg.action_dim), cfg_drop=False)

    tr = DiffusionTrainer(model, _sched(cfg), frames_per_block=1, causal=True,
                          history_noise_std=0.1, history_frames=hf)
    assert tr.history_noise_std == 0.1 and tr.history_frames == hf
    loss = tr.forward_backward(lat, cond)
    assert loss == loss and loss > 0                     # finite, ran end-to-end
    gsum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert gsum > 0                                       # grads flow through the noised path


def test_history_noise_off_by_default():
    cfg = _tiny()
    assert cfg.history_noise_std == 0.0                   # default: off
    tr = DiffusionTrainer(ARWorldModel(cfg), _sched(cfg), frames_per_block=1, causal=True)
    assert tr.history_noise_std == 0.0 and tr.history_frames == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
