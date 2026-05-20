"""Stage E: merge rollout scores back into the redteam history.

Reads the iteration's ``tasks.json`` (stage A) and ``rewards.json`` (stage D),
appends one record per task to ``history.json``, and writes it atomically so a
crash never corrupts earlier iterations.

The scalar score the LLM sees is the last value of the averaged
``per_frame_progress`` -- matching ``run_evaluation.py::_print_reward_summary``.
Cases that stage B never produced, or that scoring dropped, are still recorded
(with ``score: null``) so the LLM sees every attempt.

Run as:
    python -m openworld.redteam.update_history --tasks tasks.json \\
        --rewards rewards.json --history history.json --iter I --suite SUITE_DIR

Seed an empty history with:
    python -m openworld.redteam.update_history --init --config CFG --history history.json
"""

import argparse
import json
import os
from pathlib import Path

from openworld.redteam.config import load_redteam_config


def _score_from_episode(ep):
    """Map a rewards.json episode entry to (score, status).

    For the VLM frame judge, the scalar is the max over per-interaction scores
    (a policy that succeeds mid-rollout still counts). For robometer it is the
    last value of the averaged per-frame progress.
    """
    if ep is None:
        return None, "unscored"
    if "error" in ep:
        return None, "error"
    if ep.get("vlm_score_max") is not None:
        return float(ep["vlm_score_max"]), "scored"
    if "vlm_scores" in ep:
        scores = ep["vlm_scores"]
        return (float(max(scores)) if scores else 0.0), "scored"
    progress = ep.get("per_frame_progress", [])
    if not progress:
        return 0.0, "scored"
    return float(progress[-1]), "scored"


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Redteam stage E: update score history")
    p.add_argument("--history", required=True, help="history.json path")
    p.add_argument("--init", action="store_true",
                   help="Seed an empty history.json (requires --config)")
    p.add_argument("--config", default=None, help="Redteam YAML config (for --init)")
    p.add_argument("--tasks", default=None, help="tasks.json from stage A")
    p.add_argument("--rewards", default=None, help="rewards.json from stage D")
    p.add_argument("--iter", type=int, default=None, help="Iteration index")
    p.add_argument("--suite", default=None, help="Iteration suite directory")
    args = p.parse_args()

    if args.init:
        if not args.config:
            p.error("--init requires --config")
        if Path(args.history).exists():
            print(f"[update_history] {args.history} already exists; leaving as-is")
            return
        cfg = load_redteam_config(args.config)
        _atomic_write(
            args.history,
            {"meta_objective": cfg["redteam"]["meta_objective"], "records": []},
        )
        print(f"[update_history] seeded {args.history}")
        return

    if not (args.tasks and args.rewards and args.iter is not None and args.suite):
        p.error("merge mode requires --tasks, --rewards, --iter, and --suite")

    history_path = Path(args.history)
    history = (
        json.loads(history_path.read_text())
        if history_path.exists()
        else {"meta_objective": "", "records": []}
    )
    history.setdefault("records", [])

    tasks_data = json.loads(Path(args.tasks).read_text())
    iter_idx = tasks_data.get("iter", args.iter)
    tasks = tasks_data.get("tasks", [])

    rewards_path = Path(args.rewards)
    rewards = (
        json.loads(rewards_path.read_text()).get("episodes", [])
        if rewards_path.exists()
        else []
    )
    rewards_by_id = {ep.get("id"): ep for ep in rewards}

    suite_dir = Path(args.suite)
    for j, task in enumerate(tasks):
        case_id = f"iter{iter_idx}_task{j}"
        ep = rewards_by_id.get(case_id)
        if ep is not None:
            score, status = _score_from_episode(ep)
        elif not (suite_dir / case_id).is_dir():
            # Stage B never produced this case.
            score, status = None, "skipped"
        else:
            # The case existed but the rollout/scoring stage dropped it.
            score, status = None, "unscored"
        history["records"].append({
            "iter": iter_idx,
            "case_id": case_id,
            "task_prompt": task.get("task_prompt", ""),
            "robot_instruction": task.get("robot_instruction", ""),
            "failure_mode": task.get("failure_mode", ""),
            "based_on": task.get("based_on", ""),
            "score": score,
            "status": status,
        })

    _atomic_write(history_path, history)
    scored = sum(
        1 for r in history["records"]
        if r["iter"] == iter_idx and r["status"] == "scored"
    )
    print(
        f"[update_history] iter {iter_idx}: added {len(tasks)} records "
        f"({scored} scored) -> {history_path}"
    )


if __name__ == "__main__":
    main()
