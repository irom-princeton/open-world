# Autoregressive (Wan / Cosmos) World Model Training


## Stages

| Stage | What | Entrypoint |
|---|---|---|
| **1 — student-init** | block-causal mid-training; inits the generator | `train_midtrain` (`stage_is_causal=True`) |
| **2 — teacher** | bidirectional mid-training; inits the teacher + critic | `train_midtrain` (`stage_is_causal=False`) |
| **3 — self-forcing / DMD** | few-step distillation on own rollouts (loads 1 + 2) | `train_self_forcing` |

> Action conditioning ablations:
[ACTION_COND_EXPERIMENTS.md](../ACTION_COND_EXPERIMENTS.md).

## How to run


**Setup**
```bash
uv sync --extra autoregressive
python -m openworld.autoregressive.train_self_forcing --smoke   # sanity check
```
**Weights + data**  (offline-loading gotcha: [AUTOREGRESSIVE.md](../AUTOREGRESSIVE.md))
```bash
bash bash_scripts/download_weights.sh
sbatch bash_scripts/data_process/preprocess_latents.sh
```
**Train**  (stages 1 + 2 in parallel, then 3)
```bash
sbatch bash_scripts/training/train_student_aligned.sh   # stage 1 (student-init)
sbatch bash_scripts/training/train_teacher_aligned.sh   # stage 2 (teacher)
sbatch bash_scripts/training/train_wan.sh               # stage 3 (self-forcing); or train_cosmos.sh
```