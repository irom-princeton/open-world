#!/bin/bash
#SBATCH --job-name=ar_wm
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/ar_%j.out
#SBATCH --error=slurm_outputs/ar_%j.out
#
# Batch (non-interactive) GPU runner for the autoregressive world model — the
# `sbatch` analog of the `get_h200` salloc alias, so an agent can submit and
# monitor the log file without holding an interactive shell.
#
# Usage:
#   mkdir -p slurm_outputs
#   sbatch scripts/ar_gpu.slurm <command...>
# e.g.
#   sbatch scripts/ar_gpu.slurm .venv/bin/python -m pytest openworld/autoregressive/tests -q
#   sbatch scripts/ar_gpu.slurm .venv/bin/python -m openworld.autoregressive.train_self_forcing --smoke
#   sbatch scripts/ar_gpu.slurm accelerate launch -m openworld.autoregressive.train_self_forcing \
#                                  --config configs/training/ar_wan_1_3b.py
#
# Monitor:  squeue -j <jobid>        (status)
#           tail -f slurm_outputs/ar_<jobid>.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  gpu=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Default command if none given: unit tests + Wan/Cosmos wiring on GPU.
if [ "$#" -eq 0 ]; then
  set -- .venv/bin/python -m pytest openworld/autoregressive/tests -q
fi

echo "+ $*"
exec "$@"
