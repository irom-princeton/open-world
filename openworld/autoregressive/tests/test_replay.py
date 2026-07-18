"""Unit test for open-loop trajectory replay (latent space, CPU, dummy backbone).

Validates the inference plumbing in ``openworld.autoregressive.infer.replay``
without weights or a VAE: priming with ground-truth history, conditioning on the
full action sequence, and autoregressively generating the remaining blocks to a
clip aligned 1:1 with the ground truth.
"""

from __future__ import annotations

import numpy as np
import torch

from openworld.autoregressive.config import ARWMArgs
from openworld.autoregressive.infer.replay import replay_episode_latents
from openworld.autoregressive.model import ARWorldModel


def _cfg():
    return ARWMArgs(
        backbone="dummy", random_init_backbone=True, multiview_layout="height_stack",
        num_cams=1, frames_per_block=2, width=64, height=64, action_dim=7,
        denoising_step_list=(1000, 500), dtype=torch.float32,
    )


def test_replay_shapes_and_history_match():
    torch.manual_seed(0)
    cfg = _cfg()
    model = ARWorldModel(cfg).eval()

    L = 12
    C, Hs, W = cfg.in_channels, cfg.latent_h_total, cfg.latent_w
    latent_gt = torch.randn(L, C, Hs, W)
    action_norm = np.clip(np.random.randn(L, cfg.action_dim), -1, 1).astype(np.float32)

    hist_blocks = 1
    gt, pred, n_hist = replay_episode_latents(
        model, latent_gt, action_norm,
        frames_per_block=cfg.frames_per_block, num_history_blocks=hist_blocks,
        in_channels=cfg.in_channels, device=torch.device("cpu"), dtype=torch.float32,
    )

    fpb = cfg.frames_per_block
    num_blocks = (L - hist_blocks * fpb) // fpb
    N = hist_blocks * fpb + num_blocks * fpb
    assert n_hist == hist_blocks * fpb
    assert gt.shape == (N, C, Hs, W)
    assert pred.shape == (N, C, Hs, W)
    # history region of the prediction is ground truth (it primes the cache)
    assert torch.allclose(pred[:n_hist], gt[:n_hist], atol=1e-5)
    # the generated region is finite and (almost surely) differs from GT
    assert torch.isfinite(pred[n_hist:]).all()
    assert not torch.allclose(pred[n_hist:], gt[n_hist:])


def test_replay_respects_max_blocks():
    torch.manual_seed(1)
    cfg = _cfg()
    model = ARWorldModel(cfg).eval()
    L = 16
    latent_gt = torch.randn(L, cfg.in_channels, cfg.latent_h_total, cfg.latent_w)
    action_norm = np.zeros((L, cfg.action_dim), dtype=np.float32)

    _, pred, n_hist = replay_episode_latents(
        model, latent_gt, action_norm,
        frames_per_block=cfg.frames_per_block, num_history_blocks=1,
        in_channels=cfg.in_channels, device=torch.device("cpu"), dtype=torch.float32,
        max_blocks=2,
    )
    assert pred.shape[0] == n_hist + 2 * cfg.frames_per_block
