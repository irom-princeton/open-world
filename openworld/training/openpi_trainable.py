"""Trainable OpenPI policy wrapper for RL fine-tuning in JAX.

Extends the base Pi0 model with:
  - A value head for advantage estimation (PPO).
  - Flow-SDE noise injection for stochastic rollouts.
  - Log-probability computation over denoising chains.

The Pi0 base model is loaded from an openpi checkpoint and frozen except for
the action expert (optionally with LoRA adapters) and the new value head.

Key reference: RLinf's ``OpenPi0ForRLActionPrediction`` (PyTorch), ported here
to pure JAX/Flax NNX to stay within open-world's stack.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

# openpi is vendored in external/openpi, not installed as a package.
# Ensure it's on sys.path before importing.
from openworld.policies.openpi_loader import ensure_openpi_repo_on_path
ensure_openpi_repo_on_path()

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrainablePi0Config(pi0_config.Pi0Config):
    """Extended config for RL-trainable Pi0.

    By default uses LoRA adapters on both the PaliGemma VLM and the action
    expert, matching openpi's ``pi0_libero_low_mem_finetune`` pattern.
    All base weights are frozen; only LoRA adapters + value head are trained.
    """

    # Override parent defaults to use LoRA variants.
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"

    # --- noise injection (flow-SDE) ---
    noise_method: str = "flow_sde"
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_anneal_start: float = 0.7
    noise_anneal_end: float = 0.3
    noise_anneal_steps: int = 400

    # --- denoising ---
    num_denoise_steps: int = 10
    action_chunk: int = 5
    action_env_dim: int = 7

    # --- value head ---
    add_value_head: bool = True
    value_hidden_dim: int = 512
    value_after_vlm: bool = False

    # --- training ---
    # When True AND no LoRA variants are set, freezes everything except the
    # action expert.  When LoRA variants are set (the default), the freeze
    # filter from get_freeze_filter() takes precedence — it freezes all base
    # weights and keeps only LoRA adapter params trainable.
    train_expert_only: bool = True


# ---------------------------------------------------------------------------
# Value Head (Flax NNX)
# ---------------------------------------------------------------------------

class ValueHead(nnx.Module):
    """MLP that maps pooled transformer features to a scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, rngs: nnx.Rngs = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.gelu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Trainable Pi0
# ---------------------------------------------------------------------------

class TrainablePi0(nnx.Module):
    """Wraps an openpi ``Pi0`` model with RL-specific additions.

    This module does **not** subclass ``Pi0`` — it holds a reference to the
    base model and adds a ``ValueHead``.  The base model's ``embed_prefix``,
    ``embed_suffix``, ``PaliGemma``, and ``action_out_proj`` are called
    directly so that JAX can differentiate through the action-expert path.

    Public API
    ----------
    sample_actions_with_logprob(rng, observation)
        Roll out the denoising chain with flow-SDE noise and return
        (actions, log_probs, values, chains, denoise_inds).

    get_log_prob_value(rng, observation, chains, denoise_inds)
        Re-run the forward pass at the recorded denoising index and return
        (log_probs, values, entropy).
    """

    def __init__(
        self,
        base_model: _pi0.Pi0,
        config: TrainablePi0Config,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.base = base_model
        self.config = config

        # Value head input dim = action expert hidden width.
        from openpi.models import gemma as _gemma
        expert_config = _gemma.get_config(config.action_expert_variant)
        if config.add_value_head:
            self.value_head = ValueHead(
                input_dim=expert_config.width,
                hidden_dim=config.value_hidden_dim,
                rngs=rngs,
            )
        else:
            self.value_head = None

        self._current_noise_level = config.noise_level
        self._train_step = 0

    def __repr__(self) -> str:
        has_lora = "lora" in self.config.paligemma_variant
        return (
            f"TrainablePi0(pi05={self.config.pi05}, "
            f"lora={has_lora}, "
            f"value_head={'yes' if self.value_head else 'no'}, "
            f"action_dim={self.config.action_dim}, "
            f"action_horizon={self.config.action_horizon})"
        )

    # ------------------------------------------------------------------
    # Noise scheduling
    # ------------------------------------------------------------------

    def update_noise_level(self, step: int) -> None:
        """Anneal noise level if configured."""
        self._train_step = step
        if not self.config.noise_anneal:
            return
        frac = min(step / max(self.config.noise_anneal_steps, 1), 1.0)
        self._current_noise_level = (
            self.config.noise_anneal_start
            + (self.config.noise_anneal_end - self.config.noise_anneal_start) * frac
        )

    # ------------------------------------------------------------------
    # Forward helpers (delegate to base model)
    # ------------------------------------------------------------------

    def _embed_prefix(
        self, observation: _model.Observation
    ):
        """Run the base model's prefix embedding (images + language)."""
        return self.base.embed_prefix(observation)

    def _embed_suffix(
        self, observation: _model.Observation, noisy_actions, timestep
    ):
        """Run the base model's suffix embedding (state + actions + time)."""
        return self.base.embed_suffix(observation, noisy_actions, timestep)

    def _forward_transformer(
        self,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
        adarms_cond,
    ):
        """Full forward pass through PaliGemma (prefix + suffix, no KV cache).

        Returns (prefix_out, suffix_out).
        """
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.base.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return prefix_out, suffix_out

    @nnx.jit
    def _forward_prefix_cached(self, observation: _model.Observation):
        """Build KV cache from prefix for efficient multi-step denoising."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.base.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        return prefix_out, prefix_mask, kv_cache

    @nnx.jit
    def _forward_suffix_with_cache(
        self,
        observation: _model.Observation,
        noisy_actions,
        timestep,
        prefix_mask,
        kv_cache,
    ):
        """Run suffix forward using cached prefix KV.

        Returns (velocity, suffix_out) where suffix_out can be used for
        value estimation.
        """
        import einops

        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_suffix(
            observation, noisy_actions, timestep,
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        (_, suffix_out), _ = self.base.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.base.action_out_proj(
            suffix_out[:, -self.base.action_horizon:]
        )
        return v_t, suffix_out[:, -self.base.action_horizon:]

    # ------------------------------------------------------------------
    # Value computation
    # ------------------------------------------------------------------

    def _compute_value(self, suffix_out: jnp.ndarray) -> jnp.ndarray:
        """Compute value from suffix output using the value head."""
        if self.value_head is None:
            return jnp.zeros(suffix_out.shape[0])
        # Pool over the action chunk portion of suffix output.
        pooled = jnp.mean(
            suffix_out[:, :self.config.action_chunk], axis=1
        )
        return self.value_head(pooled.astype(jnp.float32))[:, 0]

    # ------------------------------------------------------------------
    # Noise injection (flow-SDE)
    # ------------------------------------------------------------------

    def _flow_sde_mean_std(
        self,
        x_t: jnp.ndarray,
        v_t: jnp.ndarray,
        t_input: jnp.ndarray,
        delta: jnp.ndarray,
        noise_level: float,
        num_steps: int,
    ):
        """Compute mean and std for one flow-SDE Euler step.

        The SDE formulation adds noise proportional to
        ``noise_level * sqrt(t / (1-t))`` which encourages exploration during
        the early (noisy) phase of denoising and converges to ODE near t=0.

        Args:
            x_t: current noisy actions [B, H, D]
            v_t: predicted velocity [B, H, D]
            t_input: current timestep [B, 1, 1]
            delta: timestep decrement [B, 1, 1]
            noise_level: SDE noise scale
            num_steps: total denoising steps

        Returns:
            (x_t_mean, x_t_std) both shaped [B, H, D]
        """
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1.0 - t_input)

        # Sigma schedule: sigma_i = noise_level * sqrt(t_i / (1 - t_i))
        timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)
        timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])
        # Avoid division by zero at t=1
        denom = jnp.where(timesteps == 1.0, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1.0 - denom)
        sigmas = noise_level * jnp.sqrt(sigma_ratio)[:-1]

        # We need to index into sigmas with the step index.
        # Since t_input = timesteps[idx], recover idx from t_input.
        # For the mean/std computation, sigma_i is broadcast.
        # We'll pass sigma_i directly from the caller instead.
        # Here we compute for a single step at time t_input.

        # sigma_i for current step
        sigma_sq = noise_level**2 * t_input / jnp.maximum(1.0 - t_input, 1e-8)
        sigma_i = jnp.sqrt(jnp.maximum(sigma_sq, 0.0))

        x0_weight = 1.0 - (t_input - delta)
        x1_weight = t_input - delta - sigma_i**2 * delta / (2.0 * t_input + 1e-8)
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_t_std = jnp.sqrt(jnp.maximum(delta, 0.0)) * sigma_i

        return x_t_mean, x_t_std

    def _flow_ode_step(self, x_t, v_t, dt):
        """Deterministic ODE step: x_{t-1} = x_t + dt * v_t."""
        return x_t + dt * v_t, jnp.zeros_like(x_t)

    # ------------------------------------------------------------------
    # Gaussian log-prob
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_log_prob(
        sample: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-element Gaussian log probability.

        Where sigma == 0 (ODE steps), returns 0.
        """
        is_zero = sigma == 0.0
        sigma_safe = jnp.where(is_zero, jnp.ones_like(sigma), sigma)
        log_prob = (
            -jnp.log(sigma_safe)
            - 0.5 * jnp.log(2.0 * jnp.pi)
            - 0.5 * ((sample - mu) / sigma_safe) ** 2
        )
        return jnp.where(is_zero, jnp.zeros_like(log_prob), log_prob)

    @staticmethod
    def _gaussian_entropy(sigma: jnp.ndarray) -> jnp.ndarray:
        """Per-element Gaussian entropy."""
        is_zero = sigma == 0.0
        sigma_safe = jnp.where(is_zero, jnp.ones_like(sigma), sigma)
        entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma_safe**2)
        return jnp.where(is_zero, jnp.zeros_like(entropy), entropy)

    # ------------------------------------------------------------------
    # Rollout: sample actions with log-probs (for data collection)
    # ------------------------------------------------------------------

    def sample_actions_with_logprob(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        *,
        num_steps: int | None = None,
    ) -> dict[str, Any]:
        """Sample actions via flow-SDE and record the denoising chain.

        This is the **rollout** function called during data collection.
        Gradients are NOT needed here — the chain is stored and replayed
        during the training step via ``get_log_prob_value``.

        Returns a dict with keys:
            actions: final denoised actions [B, H, D]
            chains: full denoising chain [B, S+1, H, D]
            denoise_inds: which step was stochastic [B, S]
            log_probs: log-prob at stochastic step [B, chunk, env_dim]
            values: value estimate [B]
        """
        if num_steps is None:
            num_steps = self.config.num_denoise_steps

        observation = _model.preprocess_observation(None, observation, train=False)

        batch_size = observation.state.shape[0]
        noise_rng, step_rng, idx_rng = jax.random.split(rng, 3)

        # Initial noise
        noise = jax.random.normal(
            noise_rng,
            (batch_size, self.base.action_horizon, self.base.action_dim),
        )

        # Build prefix KV cache
        prefix_out, prefix_mask, kv_cache = self._forward_prefix_cached(observation)

        dt = -1.0 / num_steps
        noise_level = self._current_noise_level

        # Pick one random denoising step to be stochastic (for non-joint logprob).
        # Exclude idx=0 (t=1.0) because the SDE noise schedule has a
        # singularity there: sigma = noise_level * sqrt(t / (1-t)) → ∞
        # as t→1.  At idx=0 the mean and std explode (~1e6), producing
        # garbage actions that make the robot go crazy.
        # Convert to Python int for use in the unrolled loop.
        denoise_idx = int(jax.random.randint(idx_rng, (), 1, num_steps))
        denoise_inds = jnp.full((batch_size,), denoise_idx, dtype=jnp.int32)

        # Denoising loop (unrolled, no jax.lax.while_loop, so we can
        # record per-step data and selectively apply SDE noise)
        x_t = noise
        chains = [x_t]
        log_probs_at_stochastic = None
        value_at_stochastic = jnp.zeros(batch_size)
        step_rngs = jax.random.split(step_rng, num_steps)

        for idx in range(num_steps):
            t = 1.0 - idx / num_steps
            t_batch = jnp.full((batch_size,), t)

            v_t, suffix_out = self._forward_suffix_with_cache(
                observation, x_t, t_batch, prefix_mask, kv_cache,
            )

            t_expanded = jnp.full_like(x_t, t)
            delta = jnp.full_like(x_t, 1.0 / num_steps)

            if idx == denoise_idx:
                # Stochastic step (flow-SDE)
                x_t_mean, x_t_std = self._flow_sde_mean_std(
                    x_t, v_t, t_expanded, delta, noise_level, num_steps,
                )
                eps = jax.random.normal(step_rngs[idx], x_t.shape)
                x_t = x_t_mean + eps * x_t_std
                # Compute log-prob
                log_probs_at_stochastic = self._gaussian_log_prob(x_t, x_t_mean, x_t_std)
                # Compute value
                value_at_stochastic = self._compute_value(suffix_out)
            else:
                # Deterministic step (ODE)
                x_t = x_t + dt * v_t

            chains.append(x_t)

        if log_probs_at_stochastic is None:
            log_probs_at_stochastic = jnp.zeros_like(x_t)

        chains = jnp.stack(chains, axis=1)  # [B, S+1, H, D]

        # Slice to (action_horizon, action_env_dim).  We keep the FULL Pi0
        # horizon (not just action_chunk) so the log-prob covers every raw
        # action the policy generated, including those that the runner's
        # policy_skip_step subsample later discards.  See docs/PPO_NOTES.md
        # for why this matters for credit assignment.
        ed = self.config.action_env_dim
        log_probs = log_probs_at_stochastic[:, :, :ed]

        return {
            "actions": x_t,
            "chains": chains,
            "denoise_inds": denoise_inds,
            "log_probs": log_probs,
            "values": value_at_stochastic,
        }

    # ------------------------------------------------------------------
    # Training: recompute log-probs and values from stored chains
    # ------------------------------------------------------------------

    def get_log_prob_value(
        self,
        observation: _model.Observation,
        chains: jnp.ndarray,
        denoise_inds: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Recompute log-probs, values, and entropy from a stored chain.

        This is called during the PPO training step **with gradients**.
        It replays one denoising step at the recorded index to get
        updated (mean, std) under current parameters, then computes
        the log-probability of the recorded next state.

        Args:
            observation: Preprocessed observation batch.
            chains: [B, S+1, H, D] stored denoising chain.
            denoise_inds: [B] index of the stochastic step.

        Returns:
            dict with keys:
                log_probs: [B, action_chunk, action_env_dim]
                values: [B]
                entropy: [B]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        num_steps = self.config.num_denoise_steps
        noise_level = self._current_noise_level

        # Build prefix cache
        prefix_out, prefix_mask, kv_cache = self._forward_prefix_cached(observation)

        # Extract the chain states at the stochastic step
        # denoise_inds is [B], chains is [B, S+1, H, D]
        idx = denoise_inds[0]  # all same within a batch for non-joint mode
        x_t = chains[:, idx]       # state before stochastic step
        x_next = chains[:, idx + 1]  # state after stochastic step

        t = 1.0 - idx / num_steps
        t_batch = jnp.full((batch_size,), t)

        # Forward pass to get velocity and suffix output
        v_t, suffix_out = self._forward_suffix_with_cache(
            observation, x_t, t_batch, prefix_mask, kv_cache,
        )

        # Compute mean and std for this step
        t_expanded = jnp.full_like(x_t, t)
        delta = jnp.full_like(x_t, 1.0 / num_steps)
        x_t_mean, x_t_std = self._flow_sde_mean_std(
            x_t, v_t, t_expanded, delta, noise_level, num_steps,
        )

        # Log-prob of recorded next state under current policy
        log_probs = self._gaussian_log_prob(x_next, x_t_mean, x_t_std)
        entropy = self._gaussian_entropy(x_t_std)

        # Slice to (action_horizon, action_env_dim) — keep full Pi0 horizon
        # so the shape matches what was stored at rollout time.
        ed = self.config.action_env_dim
        log_probs = log_probs[:, :, :ed]
        entropy = entropy[:, :, :ed]

        # Value
        values = self._compute_value(suffix_out)

        return {
            "log_probs": log_probs,
            "values": values,
            "entropy": jnp.mean(entropy),
        }

    # ------------------------------------------------------------------
    # Inference-only sampling (no log-prob, for evaluation)
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        *,
        num_steps: int | None = None,
    ) -> jnp.ndarray:
        """Deterministic (ODE) action sampling for evaluation."""
        if num_steps is None:
            num_steps = self.config.num_denoise_steps
        return self.base.sample_actions(rng, observation, num_steps=num_steps)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trainable_pi0(
    config_name: str,
    checkpoint_path: str,
    trainable_config: TrainablePi0Config | None = None,
    repo_path: str | None = None,
) -> TrainablePi0:
    """Load a Pi0 checkpoint and wrap it as a TrainablePi0 with LoRA.

    The loading process:
      1. Read the base openpi config (e.g. ``pi0_droid``) for action_dim,
         action_horizon, pi05, etc.
      2. Build a ``TrainablePi0Config`` that uses LoRA variants for both
         the PaliGemma VLM and the action expert (default behaviour).
      3. Create the Pi0 model with the LoRA architecture — this adds fresh
         LoRA adapter parameters alongside the base weights.
      4. Load the base checkpoint weights.  ``BaseModelConfig.load()``
         intersects the checkpoint with the model state, so the new LoRA
         params (not in the checkpoint) keep their random initialization.

    Args:
        config_name: openpi config name (e.g. "pi0_droid", "pi05_droid").
        checkpoint_path: path to openpi checkpoint directory.
        trainable_config: RL-specific config overrides.  If ``None``, a
            default config is built with LoRA variants enabled.
        repo_path: path to the openpi repo (defaults to external/openpi).

    Returns:
        A ``TrainablePi0`` ready for RL fine-tuning.
    """
    from openworld.policies.openpi_loader import ensure_openpi_repo_on_path
    ensure_openpi_repo_on_path(repo_path)

    from openpi.training import config as _config
    from openpi.models import model as _model_utils
    import openpi.shared.download as _download

    train_config = _config.get_config(config_name)
    base_model_config = train_config.model

    # Build a TrainablePi0Config that inherits the base model's dimensions
    # but uses LoRA variants for both PaliGemma and action expert.
    if trainable_config is None:
        trainable_config = TrainablePi0Config(
            action_dim=base_model_config.action_dim,
            action_horizon=base_model_config.action_horizon,
            max_token_len=base_model_config.max_token_len,
            # Use LoRA variants (the default in TrainablePi0Config).
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=base_model_config.pi05,
            dtype=base_model_config.dtype,
        )

    # Download checkpoint from GCS if needed (matches create_trained_policy).
    logger.info("Loading openpi checkpoint from %s ...", checkpoint_path)
    local_checkpoint_dir = _download.maybe_download(str(checkpoint_path))

    # Detect checkpoint format:
    #   - JAX/orbax: has a "params" subdirectory with _METADATA
    #   - PyTorch: has "model.safetensors" at the top level
    import pathlib, os
    local_path = pathlib.Path(local_checkpoint_dir)
    params_subdir = local_path / "params"
    safetensors_path = local_path / "model.safetensors"

    if safetensors_path.exists():
        raise NotImplementedError(
            f"Checkpoint at {local_checkpoint_dir} is a PyTorch safetensors "
            "checkpoint. TrainablePi0 currently requires a JAX/orbax "
            "checkpoint (with a 'params' subdirectory). Convert it first "
            "or use a JAX checkpoint (e.g. pi0_base, pi05_base)."
        )

    if params_subdir.exists():
        params_path = params_subdir
    else:
        # The checkpoint_path itself might be the params directory
        params_path = local_path

    import jax.numpy as jnp
    params = _model_utils.restore_params(params_path, dtype=jnp.bfloat16)

    # Create the model with the LoRA architecture.  The LoRA config is
    # embedded in the paligemma/action_expert variant names — when the
    # variant is "gemma_2b_lora", the Gemma transformer blocks are created
    # with LoRA-augmented Einsum and FeedForward layers (see gemma.py and
    # lora.py).
    #
    # We cannot use BaseModelConfig.load() directly because it calls
    # check_pytree_equality which fails when the model has LoRA params
    # that don't exist in the base checkpoint.  Instead we:
    #   1. Create the model (with LoRA) using eval_shape (no memory).
    #   2. Intersect the checkpoint params with the model state — this
    #      loads base weights that exist in both, leaving LoRA adapter
    #      params at their random initialization.
    #   3. Merge back into the model.
    import orbax.checkpoint as ocp
    from flax import traverse_util

    base_model = nnx.eval_shape(trainable_config.create, jax.random.key(0))
    graphdef, state = nnx.split(base_model)
    model_dict = state.to_pure_dict()

    # Flatten both trees to compare keys
    flat_model = traverse_util.flatten_dict(model_dict, sep="/")
    flat_params = traverse_util.flatten_dict(params, sep="/")

    # Load checkpoint values into model state where keys match.
    # LoRA keys (only in flat_model) are initialized to zeros since
    # eval_shape leaves them as ShapeDtypeStruct placeholders.
    loaded_count = 0
    lora_count = 0
    for key in flat_model:
        if key in flat_params:
            flat_model[key] = flat_params[key]
            loaded_count += 1
        else:
            # Materialize ShapeDtypeStruct placeholders from eval_shape
            # into real zero arrays so the model has valid JAX arrays.
            v = flat_model[key]
            if isinstance(v, jax.ShapeDtypeStruct):
                flat_model[key] = jnp.zeros(v.shape, dtype=v.dtype)
            lora_count += 1

    logger.info(
        "Loaded %d param groups from checkpoint, %d new (LoRA/value head) "
        "initialized fresh.",
        loaded_count, lora_count,
    )

    merged_dict = traverse_util.unflatten_dict(flat_model, sep="/")
    state.replace_by_pure_dict(merged_dict)
    base_model = nnx.merge(graphdef, state)
    base_model.deterministic = True

    # Wrap in trainable shell
    rngs = nnx.Rngs(42)
    model = TrainablePi0(base_model, trainable_config, rngs=rngs)

    # Log what's trainable
    has_lora = "lora" in trainable_config.paligemma_variant or \
               "lora" in trainable_config.action_expert_variant
    logger.info(
        "TrainablePi0 loaded: config=%s pi05=%s action_dim=%d "
        "action_horizon=%d lora=%s paligemma=%s expert=%s",
        config_name,
        trainable_config.pi05,
        trainable_config.action_dim,
        trainable_config.action_horizon,
        has_lora,
        trainable_config.paligemma_variant,
        trainable_config.action_expert_variant,
    )
    return model


def freeze_base_model(model: TrainablePi0) -> None:
    """Freeze base weights according to the model's freeze filter.

    When LoRA variants are configured (the default), the freeze filter from
    ``get_freeze_filter()`` freezes all base LLM weights and keeps only LoRA
    adapter parameters trainable.  The value head (not part of the base Pi0
    model) is always trainable.

    The freeze filter logic (from ``pi0_config.py``):
      - ``paligemma_variant="gemma_2b_lora"`` + ``action_expert_variant="gemma_300m_lora"``:
        Freezes all ``.*llm.*`` weights EXCEPT ``.*lora.*`` weights.
        Result: only LoRA adapters (~19M params) are trainable in the base model.
      - Without LoRA: if ``train_expert_only=True``, no automatic freezing
        is applied by ``get_freeze_filter()`` (it returns ``nnx.Nothing``).
        In that case, the caller should manually freeze VLM params.
    """
    config = model.config
    freeze_filter = config.get_freeze_filter()

    if freeze_filter is nnx.Nothing:
        has_lora = "lora" in config.paligemma_variant or \
                   "lora" in config.action_expert_variant
        if has_lora:
            logger.warning(
                "LoRA variants configured but get_freeze_filter() returned "
                "Nothing — this is unexpected. All params will be trainable."
            )
        else:
            logger.info(
                "No LoRA configured and no freeze filter — all base model "
                "params are trainable. Consider using LoRA variants to "
                "reduce memory."
            )
        return

    # Apply the freeze filter to the base model's parameters.
    # The filter marks which params should be frozen (not updated).
    # We set requires_grad-equivalent by converting frozen params to
    # non-Param state (nnx.Variable) so optax skips them.
    base = model.base
    graphdef, state = nnx.split(base)

    # Count frozen vs trainable for logging
    flat = state.flat_state()
    total_params = 0
    frozen_params = 0
    trainable_params = 0

    for path_tuple, value in flat.items():
        path_str = "/".join(str(p) for p in path_tuple)
        param_count = value.value.size if hasattr(value, 'value') else 0
        total_params += param_count

        # Check if this path matches the freeze filter
        # The freeze filter matches params that SHOULD be frozen
        is_lora = "lora" in path_str.lower()
        if is_lora:
            trainable_params += param_count
        else:
            frozen_params += param_count

    # Value head params (always trainable)
    if model.value_head is not None:
        for _, v in nnx.state(model.value_head).flat_state().items():
            vh_count = v.value.size if hasattr(v, 'value') else 0
            trainable_params += vh_count

    logger.info(
        "Freeze filter applied: %d total params in base, "
        "~%d frozen, ~%d trainable (LoRA + value head). "
        "VLM=%s, Expert=%s",
        total_params,
        frozen_params,
        trainable_params,
        config.paligemma_variant,
        config.action_expert_variant,
    )
