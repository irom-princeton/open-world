"""RL fine-tuning runner for policies in a world-model environment.

Orchestrates the full training loop:
  1. Collect rollouts in the world-model environment using TrainablePi0.
  2. Score generated frames with Robometer to obtain per-chunk rewards.
  3. Save rollout videos with reward annotations.
  4. Compute GAE advantages.
  5. Update the policy using PPO.
  6. Log metrics and videos to wandb.

The runner is designed for single-GPU training without FSDP.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from openworld.datasets.initialization_dataset import InitializationDataset
from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.openpi_action_adapter import AdaptedActionChunk, OpenPIActionAdapter
from openworld.training.openpi_trainable import TrainablePi0
from openworld.training.ppo import PPOConfig, PPOTrainer
from openworld.training.reward_scorer import (
    RobometerRewardScorer,
    frames_to_chunk_rewards,
)
from openworld.training.rollout_buffer import ChunkTransition, RolloutBuffer

logger = logging.getLogger(__name__)


class RLFineTuneRunner:
    """RL fine-tuning of policies inside a world-model environment."""

    def __init__(
        self,
        env: WorldModelEnv,
        policy: TrainablePi0,
        reward_scorer: RobometerRewardScorer,
        init_dataset: InitializationDataset,
        *,
        ppo_config: PPOConfig | None = None,
        max_chunks_per_episode: int = 10,
        episodes_per_iter: int = 4,
        num_iterations: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 10,
        log_every: int = 1,
        video_dir: str | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        rng_seed: int = 42,
        action_adapter_checkpoint_path: str | None = None,
        action_adapter_gripper_max: float = 1.0,
        action_adapter_device: str | None = None,
        policy_skip_step: int = 1,
        num_action_steps: int | None = None,
    ):
        """
        Args:
            env: World-model environment for rollouts.
            policy: Trainable Pi0 model with value head.
            reward_scorer: Robometer subprocess wrapper.
            init_dataset: Dataset of episode initializations.
            ppo_config: PPO hyperparameters.
            max_chunks_per_episode: Maximum action chunks per episode.
            episodes_per_iter: Episodes to collect per training iteration.
            num_iterations: Total training iterations.
            checkpoint_dir: Where to save checkpoints.
            checkpoint_every: Save checkpoint every N iterations.
            log_every: Log metrics every N iterations.
            video_dir: Directory to save all rollout videos. Each video is
                saved with reward metadata in a JSON sidecar.
            wandb_project: W&B project name.  If set, enables wandb logging.
            wandb_entity: W&B entity (team or user).
            wandb_run_name: W&B run name.  Defaults to auto-generated.
            wandb_config: Extra config dict to log to W&B.
            rng_seed: Random seed.
        """
        self.env = env
        self.policy = policy
        self.reward_scorer = reward_scorer
        self.init_dataset = init_dataset
        self.max_chunks_per_episode = max_chunks_per_episode
        self.episodes_per_iter = episodes_per_iter
        self.num_iterations = num_iterations
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.video_dir = video_dir
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name

        self.trainer = PPOTrainer(policy, ppo_config)
        self.rng = jax.random.key(rng_seed)

        # Action denormalization stats (Pi0 DROID z-score normalization).
        # Pi0 outputs actions in a normalized space: (x - mean) / (std + eps).
        # We must denormalize before passing to the action adapter.
        self._action_norm_mean: np.ndarray | None = None
        self._action_norm_std: np.ndarray | None = None

        # State normalization stats.  The eval path normalizes state via the
        # Normalize input transform before it reaches the model.  We must do
        # the same during training.
        self._state_norm_mean: np.ndarray | None = None
        self._state_norm_std: np.ndarray | None = None

        # Action adapter: converts raw Pi0 outputs (joint velocities)
        # to Cartesian actions via a learned dynamics model + FK,
        # matching the evaluation path (OpenPIPolicy._adapt_action_chunk).
        self._action_adapter_checkpoint_path = action_adapter_checkpoint_path
        self._action_adapter_gripper_max = action_adapter_gripper_max
        self._action_adapter_device = action_adapter_device
        self._policy_skip_step = policy_skip_step
        self._num_action_steps = num_action_steps
        self._action_adapter: OpenPIActionAdapter | None = None

        # wandb setup
        self._wandb_run = None
        if wandb_project:
            self._init_wandb(wandb_config or {})

    # ------------------------------------------------------------------
    # Action denormalization
    # ------------------------------------------------------------------

    def set_action_norm_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Set the z-score normalization statistics for Pi0 actions.

        Pi0 outputs actions in normalized space: ``(x - mean) / (std + eps)``.
        These stats are used to convert back to physical actions before
        passing them to the world model.

        Args:
            mean: Per-dimension mean, shape ``(action_dim,)`` (e.g. 32).
            std: Per-dimension std, shape ``(action_dim,)`` (e.g. 32).
        """
        self._action_norm_mean = np.asarray(mean, dtype=np.float32)
        self._action_norm_std = np.asarray(std, dtype=np.float32)
        logger.info(
            "Action norm stats set: mean[:8]=%s, std[:8]=%s",
            self._action_norm_mean[:8], self._action_norm_std[:8],
        )

    def set_state_norm_stats(
        self, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Set z-score normalization statistics for the state input.

        The eval path normalizes the state via the ``Normalize`` transform
        before it reaches the model.  The training path must do the same.

        Args:
            mean: Per-dimension mean, shape ``(state_dim,)`` (e.g. 32).
            std: Per-dimension std, shape ``(state_dim,)`` (e.g. 32).
        """
        self._state_norm_mean = np.asarray(mean, dtype=np.float32)
        self._state_norm_std = np.asarray(std, dtype=np.float32)
        logger.info(
            "State norm stats set: mean[:8]=%s, std[:8]=%s",
            self._state_norm_mean[:8], self._state_norm_std[:8],
        )

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state from physical space to model space.

        Applies ``(x - mean) / (std + eps)`` to match the eval path's
        ``Normalize`` input transform.
        """
        if self._state_norm_mean is None or self._state_norm_std is None:
            return state
        eps = 1e-6
        mean = self._state_norm_mean[: state.shape[-1]]
        std = self._state_norm_std[: state.shape[-1]]
        return (state - mean) / (std + eps)

    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize Pi0 actions from model space to physical space.

        Applies ``x * (std + eps) + mean``.  Returns the full
        denormalized array — the action adapter will extract the
        dimensions it needs (joint velocities + gripper).

        Args:
            actions: Normalized actions, shape ``(horizon, model_action_dim)``.

        Returns:
            Physical actions, shape ``(horizon, model_action_dim)``.
        """
        if self._action_norm_mean is None or self._action_norm_std is None:
            logger.warning(
                "Action norm stats not set — passing raw actions to adapter. "
                "Call set_action_norm_stats() to fix this."
            )
            return actions

        eps = 1e-6
        mean = self._action_norm_mean[: actions.shape[-1]]
        std = self._action_norm_std[: actions.shape[-1]]
        denormed = actions * (std + eps) + mean
        return denormed

    def _get_action_adapter(self) -> OpenPIActionAdapter:
        """Lazily create the action adapter (matches eval path)."""
        if self._action_adapter is None:
            if self._action_adapter_checkpoint_path is None:
                raise RuntimeError(
                    "Action adapter checkpoint path not set. "
                    "Set action_adapter.checkpoint_path in the config."
                )
            self._action_adapter = OpenPIActionAdapter(
                checkpoint_path=self._action_adapter_checkpoint_path,
                gripper_max=self._action_adapter_gripper_max,
                device=self._action_adapter_device,
            )
        return self._action_adapter

    def _extract_joint_state(
        self, env_info: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract joint position and gripper position from env state.

        Matches the eval path (``OpenPIPolicy._extract_joint_state``).
        Always reads ``joint_position`` / ``gripper_position``, never
        ``robot["state"]`` (which may hold Cartesian positions after
        the first step).
        """
        state = env_info["state"]
        if isinstance(state, dict):
            robot = state.get("robot", {})
            if isinstance(robot, dict):
                joint_pos = robot.get(
                    "joint_position", robot.get("joint_positions")
                )
                gripper_pos = robot.get("gripper_position")
                if joint_pos is not None and gripper_pos is not None:
                    return (
                        np.asarray(joint_pos, dtype=np.float32).reshape(-1),
                        np.asarray(gripper_pos, dtype=np.float32).reshape(-1),
                    )
        raise RuntimeError(
            "Cannot extract joint/gripper state from env info. "
            "Ensure the initialization provides joint_position and "
            "gripper_position in state['robot']."
        )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    def _init_wandb(self, extra_config: dict[str, Any]) -> None:
        """Initialize a wandb run."""
        import wandb
        from datetime import datetime

        config = {
            "ppo": vars(self.trainer.config),
            "policy": {
                "config_name": getattr(self.policy.config, "name", "unknown"),
                "paligemma_variant": self.policy.config.paligemma_variant,
                "action_expert_variant": self.policy.config.action_expert_variant,
                "noise_method": self.policy.config.noise_method,
                "noise_level": self.policy.config.noise_level,
                "num_denoise_steps": self.policy.config.num_denoise_steps,
                "action_chunk": self.policy.config.action_chunk,
                "action_env_dim": self.policy.config.action_env_dim,
                "add_value_head": self.policy.config.add_value_head,
            },
            "training": {
                "max_chunks_per_episode": self.max_chunks_per_episode,
                "episodes_per_iter": self.episodes_per_iter,
                "num_iterations": self.num_iterations,
            },
            **extra_config,
        }

        # Append MMDD_HHMMSS timestamp to run name
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        run_name = self.wandb_run_name or "rl_finetune"
        run_name = f"{run_name}_{timestamp}"

        self._wandb_run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name,
            config=config,
        )
        logger.info("wandb run initialized: %s", self._wandb_run.url)

    def _log_wandb(
        self,
        metrics: dict[str, float],
        iteration: int,
        videos: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log metrics and optionally videos to wandb."""
        if self._wandb_run is None:
            return

        import wandb

        log_dict = {f"train/{k}": v for k, v in metrics.items()}
        log_dict["train/iteration"] = iteration

        # Log videos as wandb.Video objects
        if videos:
            for vid_info in videos:
                # vid_info: {"id", "frames", "reward", "instruction"}
                frames = vid_info["frames"]
                if frames is not None and len(frames) > 0:
                    # wandb.Video expects (T, C, H, W) or a file path
                    vid_key = f"rollouts/{vid_info['id']}"
                    try:
                        video_np = np.stack(frames) if isinstance(frames, list) else frames
                        # (T, H, W, C) → (T, C, H, W)
                        video_np = np.transpose(video_np, (0, 3, 1, 2))
                        log_dict[vid_key] = wandb.Video(
                            video_np,
                            fps=5,
                            caption=f"reward={vid_info.get('reward', 0):.3f} | {vid_info.get('instruction', '')}",
                        )
                    except Exception as e:
                        logger.warning("Failed to log video %s to wandb: %s", vid_key, e)

        wandb.log(log_dict, step=iteration)

    # ------------------------------------------------------------------
    # Video saving
    # ------------------------------------------------------------------

    def _save_episode_video(
        self,
        iteration: int,
        ep_idx: int,
        frames: np.ndarray,
        instruction: str,
        chunk_rewards: list[float],
        per_frame_progress: list[float],
    ) -> None:
        """Save an episode video with reward annotations burned into frames."""
        if not self.video_dir:
            return

        from openworld.utils.video import save_rollout_video

        video_subdir = Path(self.video_dir) / f"iter_{iteration:06d}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        total_reward = sum(chunk_rewards)
        annotated = self._annotate_frames(
            frames, instruction, chunk_rewards, per_frame_progress,
            iteration, ep_idx, total_reward,
        )

        video_path = video_subdir / f"ep_{ep_idx:03d}_r{total_reward:.3f}.mp4"
        save_rollout_video(annotated, str(video_path), fps=5)

    @staticmethod
    def _annotate_frames(
        frames: np.ndarray,
        instruction: str,
        chunk_rewards: list[float],
        per_frame_progress: list[float],
        iteration: int,
        ep_idx: int,
        total_reward: float,
    ) -> list[np.ndarray]:
        """Burn reward and progress annotations onto video frames.

        Each frame gets:
          - Top: iteration, episode, instruction, total reward
          - Bottom: per-frame progress value and which chunk it belongs to
        """
        from PIL import Image, ImageDraw, ImageFont

        n_frames = len(frames)
        n_chunks = len(chunk_rewards)
        frames_per_chunk = max(n_frames // n_chunks, 1) if n_chunks > 0 else n_frames

        # Try to load a monospace font; fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

        annotated = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(np.asarray(frame))
            draw = ImageDraw.Draw(img)

            # Which chunk does this frame belong to?
            chunk_idx = min(i // frames_per_chunk, n_chunks - 1) if n_chunks > 0 else 0
            chunk_r = chunk_rewards[chunk_idx] if chunk_idx < len(chunk_rewards) else 0.0
            progress = per_frame_progress[i] if i < len(per_frame_progress) else 0.0

            # Top line: header
            header = f"iter={iteration} ep={ep_idx} R={total_reward:.3f}"
            draw.text((4, 2), header, fill=(255, 255, 0), font=font)

            # Second line: instruction (truncated)
            instr_trunc = instruction[:60] + "..." if len(instruction) > 60 else instruction
            if instr_trunc:
                draw.text((4, 16), instr_trunc, fill=(200, 200, 200), font=font_small)

            # Bottom: progress and chunk reward
            h = img.height
            bottom = f"frame={i}/{n_frames} chunk={chunk_idx} chunkR={chunk_r:.3f} prog={progress:.3f}"
            draw.text((4, h - 14), bottom, fill=(0, 255, 0), font=font_small)

            # Progress bar at the very bottom
            bar_y = h - 3
            bar_width = int(progress * img.width)
            draw.rectangle([(0, bar_y), (bar_width, h)], fill=(0, 200, 0))

            annotated.append(np.asarray(img))

        return annotated

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full RL fine-tuning loop."""
        logger.info(
            "Starting RL fine-tuning: %d iterations, %d episodes/iter, "
            "%d max chunks/episode",
            self.num_iterations,
            self.episodes_per_iter,
            self.max_chunks_per_episode,
        )

        if self.checkpoint_dir:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self.video_dir:
            Path(self.video_dir).mkdir(parents=True, exist_ok=True)

        for iteration in range(self.num_iterations):
            self.rng, iter_rng = jax.random.split(self.rng)

            # Phase 1: Collect rollouts
            buffer, episode_frames = self._collect_rollouts(iter_rng)

            # Phase 2: Score episodes with Robometer
            logger.info("Iter %d/%d — scoring %d episodes with reward model",
                        iteration + 1, self.num_iterations, len(episode_frames))
            reward_results = self._score_episodes(buffer, episode_frames)

            # Phase 2.5: Save videos with reward labels
            wandb_videos = self._save_all_videos(
                iteration, buffer, episode_frames, reward_results,
            )

            # Phase 3: Compute advantages
            buffer.compute_advantages(
                gamma=self.trainer.config.gamma,
                gae_lambda=self.trainer.config.gae_lambda,
            )

            # Phase 4: PPO update
            logger.info("Iter %d/%d — running PPO update (%d transitions)",
                        iteration + 1, self.num_iterations, len(buffer.transitions))
            batch = buffer.get_batch()
            metrics = self.trainer.train_step(batch)

            # Reward metrics
            ep_rewards = self._compute_episode_rewards(buffer)
            metrics["reward/mean"] = float(np.mean(ep_rewards))
            metrics["reward/min"] = float(np.min(ep_rewards))
            metrics["reward/max"] = float(np.max(ep_rewards))
            metrics["reward/std"] = float(np.std(ep_rewards))
            chunk_rewards = np.array([t.reward for t in buffer.transitions])
            metrics["reward/per_chunk_mean"] = float(np.mean(chunk_rewards))
            metrics["reward/per_chunk_std"] = float(np.std(chunk_rewards))

            # Return / advantage / value distributions — diagnose PPO signal.
            returns = buffer.returns
            advantages = buffer.advantages  # raw (pre-normalization)
            old_values = np.array(
                [t.value for t in buffer.transitions], dtype=np.float32
            )
            metrics["return/mean"] = float(np.mean(returns))
            metrics["return/std"] = float(np.std(returns))
            metrics["return/min"] = float(np.min(returns))
            metrics["return/max"] = float(np.max(returns))
            metrics["advantage/mean"] = float(np.mean(advantages))
            metrics["advantage/std"] = float(np.std(advantages))
            metrics["advantage/min"] = float(np.min(advantages))
            metrics["advantage/max"] = float(np.max(advantages))
            metrics["advantage/abs_mean"] = float(np.mean(np.abs(advantages)))
            metrics["value/old_mean"] = float(np.mean(old_values))
            metrics["value/old_std"] = float(np.std(old_values))

            # Exploration noise level (tracks noise_anneal schedule).
            metrics["noise_level"] = float(
                getattr(self.policy, "_current_noise_level", 0.0)
            )

            # Console logging
            if iteration % self.log_every == 0:
                logger.info(
                    "Iter %d/%d | reward=%.4f (%.4f..%.4f) return=%.4f | "
                    "loss=%.4f pg=%.4f vf=%.4f ent=%.4f kl=%.4f clip=%.3f",
                    iteration + 1,
                    self.num_iterations,
                    metrics["reward/mean"],
                    metrics["reward/min"],
                    metrics["reward/max"],
                    metrics["return/mean"],
                    metrics["loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy"],
                    metrics["approx_kl"],
                    metrics["clip_fraction"],
                )

            # wandb logging
            self._log_wandb(metrics, iteration, videos=wandb_videos)

            # Checkpointing
            if (
                self.checkpoint_dir
                and (iteration + 1) % self.checkpoint_every == 0
            ):
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"iter_{iteration + 1:06d}"
                )
                self.trainer.save_checkpoint(ckpt_path)

        if self._wandb_run is not None:
            self._wandb_run.finish()

        logger.info("RL fine-tuning complete.")

    # ------------------------------------------------------------------
    # Phase 1: Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollouts(
        self, rng: jax.Array
    ) -> tuple[RolloutBuffer, list[dict[str, Any]]]:
        """Collect episodes in the world-model environment.

        Returns:
            buffer: RolloutBuffer with chunk transitions (rewards unfilled).
            episode_frames: List of dicts with "id", "frames", "instruction"
                for Robometer scoring.
        """
        buffer = RolloutBuffer()
        episode_frames = []

        for ep_idx in range(self.episodes_per_iter):
            rng, ep_rng = jax.random.split(rng)
            init = self.init_dataset[ep_idx % len(self.init_dataset)]
            logger.info(
                "Collecting episode %d/%d (init=%s)",
                ep_idx + 1, self.episodes_per_iter,
                getattr(init, "id", "?"),
            )
            ep_data = self._run_episode(ep_rng, init)

            for chunk in ep_data["chunks"]:
                buffer.add(chunk)
            buffer.mark_episode_end()

            episode_frames.append({
                "id": f"ep_{ep_idx}",
                "frames": ep_data["all_frames"],
                "instruction": ep_data["instruction"],
            })

        return buffer, episode_frames

    def _run_episode(
        self, rng: jax.Array, initialization: Any
    ) -> dict[str, Any]:
        """Run one episode in the world-model environment.

        Returns dict with:
            chunks: list of ChunkTransition
            all_frames: np.ndarray (T, H, W, C) of all predicted frames
            instruction: str
        """
        from openpi.models import model as _model

        info = self.env.reset(initialization)
        instruction = getattr(initialization, "instruction", "") or ""
        chunks = []
        all_frames = []

        for chunk_idx in tqdm(
            range(self.max_chunks_per_episode),
            desc="  Chunks",
            leave=False,
        ):
            rng, act_rng = jax.random.split(rng)

            # Build observation for the policy
            observation = self._build_observation(info, initialization)

            # Sample action chunk with log-probs
            result = self.policy.sample_actions_with_logprob(
                act_rng, observation,
            )

            # Store transition data (reward filled later).
            # Model outputs have batch dim (B=1); squeeze it for storage.
            chunk_transition = ChunkTransition(
                observation=self._observation_to_numpy(observation),
                chains=np.asarray(result["chains"][0]),      # [S+1, H, D]
                denoise_ind=int(result["denoise_inds"][0]),
                log_prob=np.asarray(result["log_probs"][0]), # [action_horizon, ed]
                value=float(result["values"][0]),
                instruction=instruction,
            )
            chunks.append(chunk_transition)

            # Execute actions in the environment, matching the eval path
            # (OpenPIPolicy._adapt_action_chunk):
            # 1. Denormalize Pi0 outputs from z-score space to physical
            #    space (joint velocities + gripper).
            # 2. Pass through the action adapter which uses a learned
            #    dynamics model + Franka Panda FK to convert joint
            #    velocities → Cartesian positions (xyz + euler + gripper).
            # 3. Step env with adapted actions and full state updates
            #    (joint_position, gripper_position, cartesian_position).
            raw_actions = np.asarray(result["actions"][0])  # [action_horizon, 32]
            denormed = self._denormalize_actions(raw_actions)  # [action_horizon, 32]

            # Get current joint/gripper state for the action adapter
            joint_position, gripper_position = self._extract_joint_state(info)

            # Adapt through the action adapter (joint vel → Cartesian).
            # The adapter's dynamics model expects exactly action_num (15)
            # timesteps, but Pi0's action_horizon may be larger (e.g. 50).
            # Truncate to match, same as the eval path.
            adapter = self._get_action_adapter()
            denormed_chunk = denormed[: adapter.action_num]
            adapted = adapter.adapt(
                joint_position, gripper_position, denormed_chunk
            )

            # Optionally subsample to match world model temporal resolution
            if self._policy_skip_step > 1:
                indices = list(range(
                    0, adapted.env_actions.shape[0], self._policy_skip_step
                ))
                if self._num_action_steps is not None:
                    indices = indices[: self._num_action_steps]
                adapted = AdaptedActionChunk(
                    env_actions=adapted.env_actions[indices],
                    joint_positions=adapted.joint_positions[indices],
                    gripper_positions=adapted.gripper_positions[indices],
                )

            chunk_size = self.env.scheduler.chunk_size
            num_steps = min(chunk_size, adapted.env_actions.shape[0])

            for step_idx in range(num_steps):
                cartesian_action = adapted.env_actions[step_idx]
                next_joint = adapted.joint_positions[step_idx]
                next_gripper = adapted.gripper_positions[step_idx]
                action_dict = {
                    "env_action": cartesian_action,
                    "state_update": {
                        "robot": {
                            "state_representation": "cartesian_position_with_gripper",
                            "state": cartesian_action,
                            "cartesian_position": cartesian_action[:6],
                            "joint_position": next_joint,
                            "joint_positions": next_joint,
                            "gripper_position": next_gripper,
                        }
                    },
                }
                info = self.env.step(action_dict)

                if info["did_rollout"] and info["predicted_frames"]:
                    for frame in info["predicted_frames"]:
                        frame_np = np.asarray(frame)
                        if frame_np.dtype != np.uint8:
                            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                        all_frames.append(frame_np)

        all_frames_arr = (
            np.stack(all_frames, axis=0) if all_frames
            else np.zeros((1, 224, 224, 3), dtype=np.uint8)
        )

        return {
            "chunks": chunks,
            "all_frames": all_frames_arr,
            "instruction": instruction,
        }

    def _build_observation(
        self, env_info: dict[str, Any], initialization: Any
    ) -> Any:
        """Convert environment observation to openpi Observation format.

        This is a minimal conversion — the observation dict from
        WorldModelEnv is adapted to the format expected by Pi0.
        """
        from openpi.models import model as _model

        obs = env_info["observation"]
        state = env_info["state"]

        # Extract image views.
        # The world model returns a single frame with all camera views
        # stacked vertically (e.g. 576x320 for 3 views at 192x320).
        # Split it back into individual views along the height axis
        # using the configured view_order.
        if isinstance(obs, dict) and "views" in obs:
            views = obs["views"]
        elif isinstance(obs, dict):
            views = obs
        else:
            obs_arr = np.asarray(obs)
            wm_config = getattr(self.env.world_model, "config", None)
            view_order = getattr(wm_config, "view_order", None)
            n_views = len(view_order) if view_order else 1
            if obs_arr.ndim == 3 and n_views > 1:
                H = obs_arr.shape[0]
                view_h = H // n_views
                views = {
                    name: obs_arr[i * view_h : (i + 1) * view_h, :, :]
                    for i, name in enumerate(view_order)
                }
            else:
                views = {"base_0_rgb": obs_arr}

        # Prepare images as float32 [-1, 1], matching the eval path
        # (OpenPIPolicy._prepare_image → DroidInputs → Observation.from_dict).
        # Critical: use resize_with_pad (aspect-ratio-preserving + black padding)
        # instead of stretching to 224x224, which distorts the image.
        from openpi.shared import image_tools as _image_tools

        images = {}
        image_masks = {}
        for key_name, target_key in [
            ("exterior_left", "base_0_rgb"),
            ("wrist", "left_wrist_0_rgb"),
        ]:
            img = views.get(key_name, views.get(target_key))
            if img is None:
                # Use first available view or a blank image
                img = next(iter(views.values())) if views else np.zeros(
                    (224, 224, 3), dtype=np.uint8
                )
            # If the observation stores a file path, load the image from disk.
            if isinstance(img, str):
                from PIL import Image
                img = np.asarray(Image.open(img).convert("RGB"))
            img = np.asarray(img)
            # Keep as uint8 for now — convert to [-1, 1] after resize_with_pad
            # to match the eval path (resize in pixel space, then normalize).
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 3:
                img = img[np.newaxis]  # add batch dim
            # Resize using pad-to-square (preserves aspect ratio) to match eval.
            if img.shape[-3:-1] != (224, 224):
                img = _image_tools.resize_with_pad(jnp.asarray(img), 224, 224)
            # Convert to float32 [-1, 1]
            if img.dtype == np.uint8 or img.dtype == jnp.uint8:
                img = jnp.asarray(img, dtype=jnp.float32) / 255.0 * 2.0 - 1.0
            else:
                img = jnp.asarray(img, dtype=jnp.float32)
            images[target_key] = img
            image_masks[target_key] = jnp.ones((img.shape[0],), dtype=jnp.bool_)

        # Add a third image view (right wrist) — use zeros with mask=False
        # to match the eval path (DroidInputs uses np.zeros + np.False_).
        right_key = "right_wrist_0_rgb"
        if right_key not in images:
            img_shape = images["left_wrist_0_rgb"].shape
            # Zeros in [-1, 1] space means -1.0 (black)
            images[right_key] = jnp.full(img_shape, -1.0, dtype=jnp.float32)
            image_masks[right_key] = jnp.zeros((img_shape[0],), dtype=jnp.bool_)

        # Extract state vector — always use joint_position + gripper_position
        # to match the eval path (OpenPIPolicy._build_state_inputs).
        # Never fall back to robot["state"] which may hold Cartesian
        # positions after the first env step, breaking the policy's
        # expected input format.
        if isinstance(state, dict):
            robot = state.get("robot", {})
            joint_pos = robot.get(
                "joint_position", robot.get("joint_positions")
            )
            gripper_pos = robot.get("gripper_position")
            if joint_pos is not None:
                jp = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
                if gripper_pos is not None:
                    gp = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
                    state_vec_np = np.concatenate([jp, gp])
                else:
                    state_vec_np = jp
            else:
                state_vec_np = np.zeros(
                    self.policy.config.action_dim, dtype=np.float32
                )
        else:
            state_vec_np = (
                np.asarray(state, dtype=np.float32).reshape(-1)
                if state is not None
                else np.zeros(self.policy.config.action_dim, dtype=np.float32)
            )
        # Normalize state to match the eval path.
        # The eval pipeline applies Normalize(norm_stats) which z-score
        # normalizes the state BEFORE padding and model ingestion.
        state_vec_np = self._normalize_state(state_vec_np)

        state_vec = jnp.asarray(state_vec_np.reshape(1, -1))
        # Pad state to action_dim to match the eval path.
        # The eval pipeline applies pad_to_dim(state, action_dim) via
        # a data transform (transforms.py:PadToModelActionDim) before
        # the state reaches embed_suffix → state_proj(Linear(action_dim, ...)).
        if state_vec.shape[-1] < self.policy.config.action_dim:
            pad = jnp.zeros(
                (1, self.policy.config.action_dim - state_vec.shape[-1])
            )
            state_vec = jnp.concatenate([state_vec, pad], axis=-1)
        elif state_vec.shape[-1] > self.policy.config.action_dim:
            state_vec = state_vec[:, :self.policy.config.action_dim]

        # Build tokenized prompt
        instruction = getattr(initialization, "instruction", "") or ""
        tokenized_prompt, tokenized_prompt_mask = self._tokenize_prompt(
            instruction
        )

        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state_vec,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def _tokenize_prompt(
        self, instruction: str
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Tokenize a language instruction for Pi0.

        Uses openpi's PaligemmaTokenizer which returns (tokens, mask) arrays
        already padded to max_token_len.

        Returns (tokens, mask) each shaped [1, max_token_len].
        """
        max_len = self.policy.config.max_token_len

        try:
            from openpi.models.tokenizer import PaligemmaTokenizer
            tokenizer = PaligemmaTokenizer(max_len=max_len)
            tokens, mask = tokenizer.tokenize(instruction)
            # tokenizer returns numpy arrays of shape (max_len,)
            tokens = np.asarray(tokens, dtype=np.int32)
            mask = np.asarray(mask, dtype=bool)
        except Exception:
            # Fallback: zeros with mask=False (the model will rely on images)
            tokens = np.zeros(max_len, dtype=np.int32)
            mask = np.zeros(max_len, dtype=bool)

        return (
            jnp.asarray(tokens, dtype=jnp.int32).reshape(1, -1),
            jnp.asarray(mask, dtype=jnp.bool_).reshape(1, -1),
        )

    @staticmethod
    def _observation_to_numpy(observation) -> dict[str, Any]:
        """Convert an Observation to a dict of numpy arrays for storage.

        Squeezes the batch dimension (B=1) so that stacking in
        ``RolloutBuffer.get_batch`` produces [N, ...] arrays.
        """
        return {
            "image": {
                k: np.asarray(v).squeeze(0) for k, v in observation.images.items()
            },
            "image_mask": {
                k: np.asarray(v).squeeze(0) for k, v in observation.image_masks.items()
            },
            "state": np.asarray(observation.state).squeeze(0),
            "tokenized_prompt": (
                np.asarray(observation.tokenized_prompt).squeeze(0)
                if observation.tokenized_prompt is not None
                else None
            ),
            "tokenized_prompt_mask": (
                np.asarray(observation.tokenized_prompt_mask).squeeze(0)
                if observation.tokenized_prompt_mask is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Phase 2: Reward scoring
    # ------------------------------------------------------------------

    def _score_episodes(
        self,
        buffer: RolloutBuffer,
        episode_frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score episodes with Robometer and assign per-chunk rewards.

        Returns the raw Robometer results for use in video saving.
        """
        results = self.reward_scorer.score_episodes(episode_frames)

        for ep_idx, result in enumerate(results):
            if "error" in result:
                logger.warning(
                    "Episode %s scoring failed: %s",
                    result.get("id", ep_idx),
                    result["error"],
                )
                ep_start = (
                    buffer.episode_boundaries[ep_idx - 1] if ep_idx > 0 else 0
                )
                ep_end = buffer.episode_boundaries[ep_idx]
                n_chunks = ep_end - ep_start
                buffer.set_rewards(ep_idx, [0.0] * n_chunks)
                continue

            progress = result.get("per_frame_progress", [])
            ep_start = (
                buffer.episode_boundaries[ep_idx - 1] if ep_idx > 0 else 0
            )
            ep_end = buffer.episode_boundaries[ep_idx]
            n_chunks = ep_end - ep_start

            chunk_rewards = frames_to_chunk_rewards(progress, n_chunks)
            buffer.set_rewards(ep_idx, chunk_rewards)

        return results

    # ------------------------------------------------------------------
    # Phase 2.5: Save videos with reward labels
    # ------------------------------------------------------------------

    def _save_all_videos(
        self,
        iteration: int,
        buffer: RolloutBuffer,
        episode_frames: list[dict[str, Any]],
        reward_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Save all episode videos and return info for wandb logging.

        Returns a list of dicts for wandb video logging.
        """
        wandb_videos = []

        for ep_idx, (ep_frame_data, result) in enumerate(
            zip(episode_frames, reward_results)
        ):
            frames = ep_frame_data["frames"]
            instruction = ep_frame_data.get("instruction", "")
            progress = result.get("per_frame_progress", [])

            # Get chunk rewards for this episode
            ep_start = (
                buffer.episode_boundaries[ep_idx - 1] if ep_idx > 0 else 0
            )
            ep_end = buffer.episode_boundaries[ep_idx]
            chunk_rewards = [
                buffer.transitions[i].reward
                for i in range(ep_start, ep_end)
            ]
            total_reward = sum(chunk_rewards)

            # Save video + JSON sidecar to disk
            self._save_episode_video(
                iteration=iteration,
                ep_idx=ep_idx,
                frames=frames,
                instruction=instruction,
                chunk_rewards=chunk_rewards,
                per_frame_progress=progress,
            )

            # Collect for wandb
            wandb_videos.append({
                "id": f"iter{iteration:04d}_ep{ep_idx:03d}",
                "frames": frames,
                "reward": total_reward,
                "instruction": instruction,
            })

        return wandb_videos

    @staticmethod
    def _compute_episode_rewards(buffer: RolloutBuffer) -> list[float]:
        """Compute total reward per episode from the buffer."""
        episode_rewards = []
        episode_starts = [0] + buffer.episode_boundaries[:-1]
        episode_ends = buffer.episode_boundaries

        for start, end in zip(episode_starts, episode_ends):
            ep_reward = sum(
                buffer.transitions[i].reward for i in range(start, end)
            )
            episode_rewards.append(ep_reward)

        return episode_rewards if episode_rewards else [0.0]
