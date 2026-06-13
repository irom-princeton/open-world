import logging
from typing import Any, Dict, List, Optional

from openworld.datasets.initialization import Initialization
from openworld.datasets.initialization_dataset import InitializationDataset
from openworld.envs.world_model_env import WorldModelEnv
from openworld.policies.base_policy import Policy
from openworld.utils.video import render_observation_frame
from openworld.utils.video import save_rollout_video
from openworld.utils.video import save_rollout_videos_per_view

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs policy rollouts inside a world-model environment and collects results."""

    def __init__(
        self,
        env: WorldModelEnv,
        policy: Policy,
    ):
        self.env = env
        self.policy = policy

    def run_episode(
        self,
        initialization: Initialization,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run a single episode from the given initialization.

        Returns:
            Dict with keys: ``frames``, ``metadata``, etc.
        """
        info = self.env.reset(initialization)
        self.policy.reset(instruction=initialization.instruction)

        # Match the initial observation layout to the WM's stacked-view
        # convention so frame 0 of the saved video aligns with the predicted
        # frames that follow.
        wm_cfg = getattr(getattr(self.env, "world_model", None), "config", None)
        wm_view_order = getattr(wm_cfg, "view_order", None)
        render_kwargs = {"view_order": tuple(wm_view_order)} if wm_view_order else {}
        all_frames: List[Any] = [
            render_observation_frame(info["observation"], **render_kwargs)
        ]

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
                all_frames.extend(step_info["predicted_frames"])

        return {
            "initialization_id": initialization.id,
            "instruction": initialization.instruction,
            "frames": all_frames,
            "num_steps": max_steps,
            "metadata": initialization.metadata,
        }

    def run_dataset(
        self,
        dataset: InitializationDataset,
        max_steps: int = 50,
        video_dir: Optional[str] = None,
        video_fps: int = 5,
        per_view: bool = False,
        view_order: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """Run episodes for every initialization in the dataset.

        Args:
            dataset: The initialization dataset to iterate over.
            max_steps: Maximum number of environment steps per episode.
            video_dir: If provided, save rollout videos to this directory.
            video_fps: Frames per second for saved videos.
            per_view: If True, split each (vertically-stacked) frame into one
                video per camera view instead of a single combined video.
            view_order: View names ordered top-to-bottom in the stacked frame.
                Required when ``per_view`` is True; defaults to the world
                model's configured ``view_order`` if available.

        Returns:
            List of per-episode result dicts.
        """
        results: List[Dict[str, Any]] = []

        if per_view and not view_order:
            wm_cfg = getattr(getattr(self.env, "world_model", None), "config", None)
            view_order = getattr(wm_cfg, "view_order", None)
            if not view_order:
                raise ValueError(
                    "per_view=True requires view_order (none found on the world model)."
                )

        for init in dataset:
            episode_result = self.run_episode(init, max_steps=max_steps)
            results.append(episode_result)

            if video_dir and episode_result["frames"]:
                if per_view:
                    written = save_rollout_videos_per_view(
                        frames=episode_result["frames"],
                        output_dir=f"{video_dir}/{init.id}",
                        view_order=tuple(view_order),
                        fps=video_fps,
                    )
                    episode_result["view_video_paths"] = written
                    logger.info(
                        "Saved %d per-view videos for %s (%d frames each)",
                        len(written),
                        init.id,
                        len(episode_result["frames"]),
                    )
                else:
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
