# `bash_scripts/` — AR world-model launchers

All shell / sbatch wrappers for the autoregressive world-model pipeline, grouped
by stage. The Python entrypoints they call live in `scripts/` and
`openworld/autoregressive/`; these scripts are the thin, cluster-aware launchers.

Submit from the **repo root** so relative paths (`scripts/…`, `configs/…`,
`external/…`) resolve.

```
bash_scripts/
  _env.sh                       shared env (sourced): offline HF, cd, slurm_outputs, node info
  ar_gpu.slurm                  generic GPU runner for ad-hoc commands
  download_weights.sh           login-node weight download (NOT sbatch)
  data_process/
    preprocess_latents.sh       stage 1: raw DROID RGB+actions -> 16-ch Wan-VAE latents
    validate_data.sh            stage 1 check: latents -> one real Wan forward
  training/
    train_wan.sh                stage 2: self-forcing / DMD distillation (Wan backbone)
    train_cosmos.sh             stage 2: same, Cosmos-Predict2 backbone
    smoke.sh                    unit tests + weightless smoke (REAL=1 adds real-weights smoke)
  inference/
    replay_wan.sh               stage 3: open-loop trajectory replay (Wan)
    replay_cosmos.sh            stage 3: open-loop trajectory replay (Cosmos)
```

## Typical run

```bash
# 0. one-time, on the login node (has internet):
bash bash_scripts/download_weights.sh                       # Wan transformer + VAE -> external/
# Cosmos too, if needed:
bash bash_scripts/download_weights.sh nvidia/Cosmos-Predict2-2B-Video2World

# 1. preprocess raw DROID -> latents (GPU):
sbatch bash_scripts/data_process/preprocess_latents.sh
sbatch bash_scripts/data_process/validate_data.sh

# 2. distill:
sbatch bash_scripts/training/train_wan.sh                   # or training/train_cosmos.sh

# 3. replay a trained checkpoint:
CKPT=checkpoints/ar_wm/ar_wan_droid/checkpoint-50000.pt \
  sbatch bash_scripts/inference/replay_wan.sh

# monitor any job:
squeue -j <jobid>   ·   tail -f slurm_outputs/<name>_<jobid>.out
```

## Overriding defaults

Every launcher reads its knobs from environment variables (sensible defaults
baked in) so you rarely edit the files. Set them inline before `sbatch`:

| script | useful env vars (defaults) |
|---|---|
| `data_process/preprocess_latents.sh` | `ROOT`, `OUT` (`data/droid_ar_latents`), `SPLITS` (`train val`), `NUM_VIEWS` (3), `LIMIT`, `VAE_DIR` |
| `data_process/validate_data.sh` | `LATENT_ROOT` (`data/droid_ar_latents`) |
| `training/train_wan.sh` / `train_cosmos.sh` | `CONFIG`, `ACCELERATE` |
| `training/smoke.sh` | `REAL` (0) |
| `inference/replay_wan.sh` / `replay_cosmos.sh` | `CKPT`, `CONFIG`, `LATENT_ROOT`, `SPLIT` (`val`), `HISTORY_BLOCKS` (1), `NUM_EPISODES`, `EPISODE_ID`, `MAX_BLOCKS`, `OUTPUT_DIR`, `SEPARATE` |

`PY` (default `.venv/bin/python`) and `HF_HOME` are honored by all of them via
`_env.sh`. SBATCH resource directives (`--time`, `--mem`, `--gres`) are at the top
of each file — edit there for longer runs or multi-GPU.
