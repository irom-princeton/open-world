"""Redteam: an autonomous closed loop for discovering robot-policy failure modes.

Stages (each run as an isolated subprocess by ``bash_scripts/redteam.sh``):
  A. ``generate_tasks``  -- LLM proposes scene-edit tasks from the score history.
  B. ``generate_scenes`` -- nanobanana + multiview turn each task into a 3-view
     world-model initialization suite.
  C. ``scripts/generate_videos.py`` -- policy rollout inside the world model.
  D. ``scripts/score_videos_robometer.py`` -- score the rollouts.
  E. ``update_history`` -- merge scores back into ``history.json``.

``config.py`` holds shared config loading and small CLI helpers for the
orchestrator. Heavy stages are intentionally not imported here so that
``python -m openworld.redteam.config`` stays light.
"""
