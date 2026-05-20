import logging
from typing import Any, Callable, Dict, List, Optional

from openworld.datasets.initialization import Initialization
from openworld.datasets.initialization_dataset import InitializationDataset
from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.base_policy import Policy
from openworld.utils.video import render_observation_frame
from openworld.utils.video import save_rollout_video

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs policy rollouts inside a world-model environment and collects results."""

    def __init__(
        self,
        env: WorldModelEnv,
        policy: Policy,
        frame_scorer: Optional[Callable[[Any, str], float]] = None,
    ):
        self.env = env
        self.policy = policy
        # Optional callable (frame, instruction) -> float in [0, 1]. When set, it
        # is queried after every world-model interaction on the latest frame so a
        # policy that succeeds mid-rollout (and then drifts away) is still scored.
        self.frame_scorer = frame_scorer

    def run_episode(
        self,
        initialization: Initialization,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run a single episode from the given initialization.

        Returns:
            Dict with keys: ``frames``, ``metadata``, etc. When a ``frame_scorer``
            is configured, also ``vlm_scores`` (one per interaction) and
            ``vlm_score_max``.
        """
        info = self.env.reset(initialization)
        self.policy.reset(instruction=initialization.instruction)

        all_frames: List[Any] = [render_observation_frame(info["observation"])]
        vlm_scores: List[float] = []

        for step in range(max_steps):
            obs = self.env.get_current_observation()
            state = self.env.get_current_state()

            action = self.policy.act(
                observation=obs,
                state=state,
                instruction=initialization.instruction,
            )

            step_info = self.env.step(action)

            if step_info["did_rollout"]:
                predicted = step_info["predicted_frames"]
                all_frames.extend(predicted)

                # Judge the latest frame of this interaction (in case the policy
                # reached the goal here even if it later drifts away).
                if self.frame_scorer is not None and predicted:
                    score = self.frame_scorer(predicted[-1], initialization.instruction)
                    vlm_scores.append(float(score))
                    logger.info(
                        "%s step %d: VLM frame score %.3f", initialization.id, step, score
                    )

        result: Dict[str, Any] = {
            "initialization_id": initialization.id,
            "instruction": initialization.instruction,
            "frames": all_frames,
            "num_steps": max_steps,
            "metadata": dict(initialization.metadata or {}),
        }
        if self.frame_scorer is not None:
            result["vlm_scores"] = vlm_scores
            result["vlm_score_max"] = max(vlm_scores) if vlm_scores else None
            result["metadata"]["vlm_scores"] = vlm_scores
            result["metadata"]["vlm_score_max"] = result["vlm_score_max"]
        return result

    def run_dataset(
        self,
        dataset: InitializationDataset,
        max_steps: int = 50,
        video_dir: Optional[str] = None,
        video_fps: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run episodes for every initialization in the dataset.

        Args:
            dataset: The initialization dataset to iterate over.
            max_steps: Maximum number of environment steps per episode.
            video_dir: If provided, save rollout videos to this directory.
            video_fps: Frames per second for saved videos.

        Returns:
            List of per-episode result dicts.
        """
        results: List[Dict[str, Any]] = []

        for init in dataset:
            episode_result = self.run_episode(init, max_steps=max_steps)
            results.append(episode_result)

            if video_dir and episode_result["frames"]:
                save_rollout_video(
                    frames=episode_result["frames"],
                    output_path=f"{video_dir}/{init.id}.mp4",
                    fps=video_fps,
                )
                logger.info(
                    "Saved video for %s (%d frames)",
                    init.id,
                    len(episode_result["frames"]),
                )

        return results
