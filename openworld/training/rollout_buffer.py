"""Rollout buffer for PPO-style RL training.

Stores per-chunk transition data collected during world-model rollouts:
  - observations (as pytrees of arrays)
  - denoising chains and indices
  - log-probabilities under the collection policy
  - value estimates
  - rewards (from Robometer, assigned per chunk)
  - advantages and returns (computed after the episode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass
class ChunkTransition:
    """Data for a single action-chunk step in the world-model environment.

    Each chunk corresponds to one world-model rollout (e.g. 15 actions → 5
    predicted frames).  The reward is obtained by scoring those frames with
    Robometer.
    """

    # Observation at the start of this chunk (dict of numpy arrays).
    observation: dict[str, Any]

    # Full denoising chain [S+1, H, D] recorded by TrainablePi0.
    chains: np.ndarray

    # Index of the stochastic denoising step (scalar or [1]).
    denoise_ind: int

    # Log-prob under the rollout policy [action_chunk, action_env_dim].
    log_prob: np.ndarray

    # Value estimate at this chunk (scalar).
    value: float

    # Reward for this chunk (filled after Robometer scoring).
    reward: float = 0.0

    # Language instruction for this episode.
    instruction: str = ""


@dataclass
class RolloutBuffer:
    """Collects chunk-level transitions across one or more episodes.

    After episodes are scored, call ``compute_advantages`` to fill in GAE
    advantages and discounted returns.
    """

    transitions: list[ChunkTransition] = field(default_factory=list)
    episode_boundaries: list[int] = field(default_factory=list)

    # Computed after all episodes are collected.
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(self, transition: ChunkTransition) -> None:
        self.transitions.append(transition)

    def mark_episode_end(self) -> None:
        """Mark the current position as the end of an episode."""
        self.episode_boundaries.append(len(self.transitions))

    def __len__(self) -> int:
        return len(self.transitions)

    def set_rewards(self, episode_idx: int, rewards: list[float]) -> None:
        """Assign per-chunk rewards for a completed episode.

        Args:
            episode_idx: Which episode (0-indexed).
            rewards: List of per-chunk rewards for that episode.
        """
        start = self.episode_boundaries[episode_idx - 1] if episode_idx > 0 else 0
        end = self.episode_boundaries[episode_idx]
        n_chunks = end - start
        if len(rewards) != n_chunks:
            raise ValueError(
                f"Episode {episode_idx} has {n_chunks} chunks but got "
                f"{len(rewards)} rewards."
            )
        for i, r in enumerate(rewards):
            self.transitions[start + i].reward = r

    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Processes each episode independently using the episode boundaries.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda for bias-variance trade-off.
            last_value: Bootstrap value for the last state (usually 0 for
                terminal episodes in a world model).
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Process each episode separately
        episode_starts = [0] + self.episode_boundaries[:-1]
        episode_ends = self.episode_boundaries

        for ep_start, ep_end in zip(episode_starts, episode_ends):
            gae = 0.0
            for t in reversed(range(ep_start, ep_end)):
                reward = self.transitions[t].reward
                value = self.transitions[t].value

                if t == ep_end - 1:
                    next_value = last_value
                else:
                    next_value = self.transitions[t + 1].value

                delta = reward + gamma * next_value - value
                gae = delta + gamma * gae_lambda * gae
                advantages[t] = gae
                returns[t] = gae + value

        self.advantages = advantages
        self.returns = returns

    def get_batch(self) -> dict[str, Any]:
        """Build a batched dict of all transitions for training.

        Returns arrays ready to be passed to the PPO update step:
            observations: dict of stacked arrays [N, ...]
            chains: [N, S+1, H, D]
            denoise_inds: [N]
            old_log_probs: [N, action_chunk, action_env_dim]
            old_values: [N]
            advantages: [N]
            returns: [N]
        """
        if self.advantages is None:
            raise RuntimeError("Call compute_advantages() before get_batch().")

        # Stack observations — merge dicts of arrays.
        # Skip keys where all values are None (e.g. tokenized_prompt
        # when prompts are not used) since np.stack on None produces
        # dtype=object arrays that JAX cannot handle.
        obs_keys = self.transitions[0].observation.keys()
        observations = {}
        for key in obs_keys:
            vals = [t.observation[key] for t in self.transitions]
            if all(v is None for v in vals):
                continue
            if isinstance(vals[0], dict):
                # Nested dict (e.g. "image", "image_mask")
                observations[key] = {
                    k: np.stack([v[k] for v in vals], axis=0)
                    for k in vals[0]
                }
            else:
                observations[key] = np.stack(vals, axis=0)

        chains = np.stack([t.chains for t in self.transitions], axis=0)
        denoise_inds = np.array(
            [t.denoise_ind for t in self.transitions], dtype=np.int32
        )
        old_log_probs = np.stack(
            [t.log_prob for t in self.transitions], axis=0
        )
        old_values = np.array(
            [t.value for t in self.transitions], dtype=np.float32
        )

        return {
            "observations": observations,
            "chains": chains,
            "denoise_inds": denoise_inds,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def clear(self) -> None:
        """Reset the buffer for a new collection phase."""
        self.transitions = []
        self.episode_boundaries = []
        self.advantages = None
        self.returns = None
