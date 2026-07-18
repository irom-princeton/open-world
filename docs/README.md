# OpenWorld docs

The two core pipelines and how they connect:

```
            ┌─────────────────────────────┐         ┌──────────────────────────────┐
 language + │  SCENE GENERATION           │ suite   │  POLICY EVALUATION           │ rollout
 image      │  instruction+image → suite  │ ──────► │  policy ⇄ world model (loop) │ videos
            │  → SCENEGEN.md              │         │  → EVAL.md                   │ (+reward)
            └─────────────────────────────┘         └──────────────────────────────┘
```

Scene generation produces **Initialization suites**; policy evaluation runs a
policy closed-loop inside a world model **over those suites**. The suite is the
handoff format between them (a dir of per-case `initialization.yaml` + per-view
PNGs, pointed to by an eval config's `dataset_path`).

## Pipelines

| Doc | Pipeline | Entry points |
|-----|----------|--------------|
| **[SCENEGEN.md](SCENEGEN.md)** | Build test-case suites from an instruction + image | `scripts/generate_test_case.py` (single case) · `scripts/scenegen/` (batch authoring) · `openworld/scenegen/` · `configs/scenegen/` |
| **[EVAL.md](EVAL.md)** | Run pi0.5 closed-loop in a world model (ctrl-world / AR / weaver) over a suite | `bash_scripts/eval_wm.sbatch` → `scripts/run_evaluation.py` → `scripts/generate_videos.py` · `configs/evaluation/` |

## Quick start

```bash
# 1. (one time) set up the eval stack: submodules + venvs + checkpoint symlinks
bash bash_scripts/setup_eval_env.sh

# 2. (optional) author a test suite from an instruction + image
GOOGLE_API_KEY=... uv run python scripts/generate_test_case.py \
    --instruction "put the carrot in the bowl" --init-image <img> --name carrot --num-cases 3

# 3. run a policy eval over a suite (ar | weaver | ctrlworld)
sbatch bash_scripts/eval_wm.sbatch --wm ar
```

## Other references

| Doc | Topic |
|-----|-------|
| [MODELS.md](MODELS.md) | World-model × function support matrix |
| [TRAIN_POLICY.md](TRAIN_POLICY.md) | Training / loading policy checkpoints |
| [LIBERO.md](LIBERO.md) | LIBERO benchmark specifics |
| [world_model_training/](world_model_training/) | AR world-model training (self-forcing, etc.) |
