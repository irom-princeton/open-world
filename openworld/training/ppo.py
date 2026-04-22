"""PPO training for flow-matching policies in JAX.

Implements Proximal Policy Optimization (PPO) with clipped objective for
fine-tuning Pi0/Pi0.5 models using rewards from Robometer.

The key difference from standard PPO is that the "action" is a denoising
chain rather than a single vector.  Log-probabilities are computed at a
single stochastic denoising step (flow-SDE), while the rest of the chain
is deterministic (flow-ODE).

Uses optax for optimization and Flax NNX for parameter management.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openworld.training.openpi_trainable import TrainablePi0

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    # Learning rate
    lr: float = 1e-5

    # PPO clipping range
    clip_eps: float = 0.2

    # Value function loss coefficient
    vf_coef: float = 0.5

    # Entropy bonus coefficient
    entropy_coef: float = 0.01

    # Max gradient norm
    max_grad_norm: float = 1.0

    # Number of PPO epochs per rollout batch
    num_epochs: int = 4

    # Mini-batch size (number of chunks per gradient step)
    mini_batch_size: int = 8

    # Discount factor
    gamma: float = 0.99

    # GAE lambda
    gae_lambda: float = 0.95

    # Whether to normalize advantages
    normalize_advantages: bool = True

    # KL penalty coefficient (0 to disable)
    kl_coef: float = 0.0

    # Target KL for early stopping (0 to disable)
    target_kl: float = 0.0


class PPOTrainer:
    """Single-GPU PPO trainer for TrainablePi0 models.

    This trainer handles the gradient computation and parameter updates.
    Rollout collection and reward scoring are handled by the
    ``RLFineTuneRunner``.
    """

    # Keywords identifying trainable parameter groups.
    _TRAINABLE_KEYWORDS = (
        "lora", "value_head",
        "action_in_proj", "action_out_proj",
        "action_time_mlp", "state_proj",
        "time_mlp_in", "time_mlp_out",
    )

    def __init__(
        self,
        model: TrainablePi0,
        config: PPOConfig | None = None,
    ):
        self.model = model
        self.config = config or PPOConfig()

        # Build a trainable-param mask so the optimizer only updates LoRA
        # adapters and the value head, leaving frozen base weights unchanged.
        trainable_mask = self._build_trainable_mask(model)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.masked(optax.adam(self.config.lr), trainable_mask),
        )

        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self.opt_state = self.optimizer.init(state.to_pure_dict())

        self._step_count = 0

        # Pre-compile the gradient function so we don't retrace every call.
        self._grad_fn = jax.jit(
            jax.value_and_grad(self._loss_fn, has_aux=True)
        )

    @staticmethod
    def _build_trainable_mask(model: TrainablePi0) -> Any:
        """Build a pytree mask marking which params are trainable.

        Returns a pytree with the same structure as the model state,
        where ``True`` means the param should be updated and ``False``
        means it should be frozen.

        Trainable params:
          - LoRA adapter weights (path contains "lora")
          - Value head weights (path under "value_head")
          - Action projection layers (action_in_proj, action_out_proj, etc.)
        Frozen params:
          - PaliGemma VLM base weights
          - SigLIP vision encoder weights
          - Base Gemma LLM weights (non-LoRA)
        """
        from flax import traverse_util

        _, state = nnx.split(model)
        pure_dict = state.to_pure_dict()
        flat = traverse_util.flatten_dict(pure_dict, sep="/")

        trainable_keywords = [
            "lora", "value_head",
            "action_in_proj", "action_out_proj",
            "action_time_mlp", "state_proj",
            "time_mlp_in", "time_mlp_out",
        ]

        def _is_trainable(path: str) -> bool:
            path_lower = path.lower()
            return any(kw in path_lower for kw in trainable_keywords)

        mask_flat = {k: _is_trainable(k) for k in flat}
        mask_dict = traverse_util.unflatten_dict(mask_flat, sep="/")

        trainable_count = sum(1 for v in mask_flat.values() if v)
        frozen_count = sum(1 for v in mask_flat.values() if not v)
        logger.info(
            "Optimizer mask: %d trainable param groups, %d frozen",
            trainable_count, frozen_count,
        )

        return mask_dict

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one full PPO update on a batch of rollout data.

        Args:
            batch: Output of ``RolloutBuffer.get_batch()`` containing:
                observations, chains, denoise_inds, old_log_probs,
                old_values, advantages, returns.

        Returns:
            Dict of training metrics (loss, policy_loss, value_loss,
            entropy, approx_kl, clip_fraction).
        """
        cfg = self.config

        # Convert to JAX arrays
        jax_batch = jax.tree.map(jnp.asarray, batch)

        n = jax_batch["advantages"].shape[0]

        # Normalize advantages
        if cfg.normalize_advantages and n > 1:
            adv = jax_batch["advantages"]
            jax_batch["advantages"] = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

        # Config scalars packed separately — they are 0-d and must not
        # be indexed when slicing mini-batches.
        cfg_scalars = {
            "_clip_eps": jnp.float32(cfg.clip_eps),
            "_vf_coef": jnp.float32(cfg.vf_coef),
            "_entropy_coef": jnp.float32(cfg.entropy_coef),
        }

        # Split once — thread state through all mini-batch steps to avoid
        # repeated nnx.split / nnx.update per step.
        graphdef, state = nnx.split(self.model)

        all_metrics = []
        should_stop = False

        for epoch in range(cfg.num_epochs):
            # Shuffle and create mini-batches
            perm = np.random.permutation(n)
            for start in range(0, n, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, n)
                indices = perm[start:end]
                mb = jax.tree.map(lambda x: x[indices], jax_batch)
                mb.update(cfg_scalars)

                state, metrics = self._update_step(state, graphdef, mb)
                all_metrics.append(metrics)

                should_stop = (
                    cfg.target_kl > 0
                    and metrics["approx_kl"] > cfg.target_kl
                )
                if should_stop:
                    logger.info(
                        "PPO early stopping at epoch %d (KL=%.4f > target=%.4f)",
                        epoch, metrics["approx_kl"], cfg.target_kl,
                    )
                    break

            if should_stop:
                break

        # Write updated params back to the model once.
        nnx.update(self.model, state)

        self._step_count += 1
        self.model.update_noise_level(self._step_count)

        # Average metrics across all mini-batch steps
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        avg_metrics["num_epochs_actual"] = epoch + 1
        avg_metrics["num_updates"] = len(all_metrics)

        return avg_metrics

    def _update_step(
        self,
        state: nnx.State,
        graphdef: nnx.GraphDef,
        mini_batch: dict[str, jnp.ndarray],
    ) -> tuple[nnx.State, dict[str, float]]:
        """Single gradient update step.

        Takes and returns nnx.State so the caller can thread state
        through multiple mini-batch steps without split/merge overhead.
        """
        (loss, aux), grads = self._grad_fn(state, graphdef, mini_batch)

        grads_dict = grads.to_pure_dict()
        state_dict = state.to_pure_dict()

        updates, new_opt_state = self.optimizer.update(
            grads_dict, self.opt_state, state_dict
        )
        new_state_dict = optax.apply_updates(state_dict, updates)
        self.opt_state = new_opt_state

        # replace_by_pure_dict is in-place (returns None).
        state.replace_by_pure_dict(new_state_dict)

        metrics = {
            "loss": float(loss),
            "policy_loss": float(aux["policy_loss"]),
            "value_loss": float(aux["value_loss"]),
            "entropy": float(aux["entropy"]),
            "approx_kl": float(aux["approx_kl"]),
            "clip_fraction": float(aux["clip_fraction"]),
            "ratio_mean": float(aux["ratio_mean"]),
            "ratio_std": float(aux["ratio_std"]),
            "ratio_min": float(aux["ratio_min"]),
            "ratio_max": float(aux["ratio_max"]),
            "log_prob_new_mean": float(aux["log_prob_new_mean"]),
            "log_prob_old_mean": float(aux["log_prob_old_mean"]),
            "value_mean": float(aux["value_mean"]),
            "value_std": float(aux["value_std"]),
            "explained_variance": float(aux["explained_variance"]),
        }
        return state, metrics

    @staticmethod
    def _loss_fn(
        state: nnx.State,
        graphdef: nnx.GraphDef,
        mini_batch: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """PPO clipped objective + value loss + entropy bonus.

        This is a pure function suitable for ``jax.value_and_grad``.
        Frozen parameters are wrapped with ``stop_gradient`` so XLA
        skips their backward-pass computation — major speedup when
        only ~1% of params are trainable.

        Args:
            state: Current model parameters (Flax NNX state).
            graphdef: Model graph definition for reconstruction.
            mini_batch: Dict with observations, chains, denoise_inds,
                old_log_probs, advantages, returns.

        Returns:
            (total_loss, aux_dict) where aux_dict contains component losses
            and diagnostics.
        """
        # Apply stop_gradient to frozen params so XLA skips computing
        # their gradients (LoRA + value head + action projections are
        # trainable; everything else is frozen).
        keywords = PPOTrainer._TRAINABLE_KEYWORDS

        def _maybe_stop_grad(path, x):
            path_str = jax.tree_util.keystr(path).lower()
            if any(kw in path_str for kw in keywords):
                return x
            return jax.lax.stop_gradient(x)

        state = jax.tree_util.tree_map_with_path(_maybe_stop_grad, state)

        model = nnx.merge(graphdef, state)

        # Reconstruct observation from stored arrays
        obs_dict = mini_batch["observations"]
        observation = _model.Observation.from_dict(obs_dict)

        # Recompute log-probs and values under current policy
        result = model.get_log_prob_value(
            observation,
            mini_batch["chains"],
            mini_batch["denoise_inds"],
        )

        new_log_probs = result["log_probs"]
        new_values = result["values"]
        entropy = result["entropy"]

        old_log_probs = mini_batch["old_log_probs"]
        advantages = mini_batch["advantages"]
        returns = mini_batch["returns"]

        # Read config values from the batch
        clip_eps = mini_batch["_clip_eps"]
        vf_coef = mini_batch["_vf_coef"]
        entropy_coef = mini_batch["_entropy_coef"]

        # Per-element PPO: each (action_horizon, action_env_dim) cell is its
        # own PPO sample with its own ε trust region.  RLinf does the same
        # (rlinf/algorithms/losses.py:241-269).  Avoids the "sum 70 log-probs
        # into the exponent" failure mode documented in docs/PPO_NOTES.md.
        # log_ratio is still clipped before exp as a numerical safety net.
        log_ratio = new_log_probs - old_log_probs                  # [B, H, D]
        log_ratio = jnp.clip(log_ratio, -10.0, 10.0)
        ratio = jnp.exp(log_ratio)                                  # [B, H, D]

        # Broadcast scalar advantages [B] across action dims.
        adv_b = advantages[:, None, None]                           # [B, 1, 1]

        pg_loss1 = -adv_b * ratio
        pg_loss2 = -adv_b * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))     # mean over [B, H, D]

        # Value loss (simple MSE)
        value_loss = 0.5 * jnp.mean((new_values - returns) ** 2)

        # Total loss
        total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

        # Diagnostics — per-element, then averaged.
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        clip_fraction = jnp.mean(
            jnp.abs(ratio - 1.0) > clip_eps
        )

        # Explained variance of the value head: 1 - Var(returns - values)/Var(returns).
        # Near 0 = value fn no better than predicting the mean; near 1 = near-perfect.
        returns_var = jnp.var(returns)
        explained_var = 1.0 - jnp.var(returns - new_values) / (returns_var + 1e-8)

        aux = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            # Ratio distribution — shows how much the policy actually moved.
            "ratio_mean": jnp.mean(ratio),
            "ratio_std": jnp.std(ratio),
            "ratio_min": jnp.min(ratio),
            "ratio_max": jnp.max(ratio),
            # Log-prob magnitudes before/after the update.
            "log_prob_new_mean": jnp.mean(new_log_probs),
            "log_prob_old_mean": jnp.mean(old_log_probs),
            # Value head diagnostics.
            "value_mean": jnp.mean(new_values),
            "value_std": jnp.std(new_values),
            "explained_variance": explained_var,
        }

        return total_loss, aux

    def save_checkpoint(self, path: str) -> None:
        """Save trainable parameters to disk."""
        import orbax.checkpoint as ocp

        path = os.path.abspath(path)
        _, state = nnx.split(self.model)
        with ocp.PyTreeCheckpointer() as ckptr:
            ckptr.save(path, state.to_pure_dict())
        logger.info("Checkpoint saved to %s", path)
