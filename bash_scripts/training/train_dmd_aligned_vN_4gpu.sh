#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#
# Generic 4-GPU DMD distillation launcher for the v4..v10 hyperparameter sweep.
# Submit with the config path as the first arg and an output subdir name, e.g.:
#
#   sbatch --job-name=dmd_aligned_v4_4gpu \
#          --output=slurm_outputs/ar_dmd_aligned_v4_4gpu/%j.out \
#          --error=slurm_outputs/ar_dmd_aligned_v4_4gpu/%j.out \
#          bash_scripts/training/train_dmd_aligned_vN_4gpu.sh \
#          configs/training/ar_wan_droid_aligned_v4_4gpu.py
#
# Same inits / teacher-resolution / guards as train_dmd_aligned_v3_4gpu.sh, but
# the config (and thus the tuned hyperparameters + output tag) is an argument.

set -eo pipefail

CONFIG="${1:?usage: sbatch ... train_dmd_aligned_vN_4gpu.sh <config.py>}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"

GPUS="${GPUS:-4}"
# unique master port per job so concurrent runs never collide
MASTER_PORT="${MASTER_PORT:-$((29520 + SLURM_JOB_ID % 1000))}"

# output dir = tag from the config (recomputed from its own get_args)
TAG="$(.venv/bin/python -c "from importlib import import_module; import sys; m=import_module('${CONFIG%.py}'.replace('/','.')); print(m.get_args().tag)")"
mkdir -p "slurm_outputs/ar_${TAG#ar_wan_dmd_}" "checkpoints/ar_wm/${TAG}"

# --- Resolve + pin the teacher checkpoint (config's own resolver). -----------
DMD_TEACHER_CKPT="$(.venv/bin/python -c 'from configs.training.ar_wan_droid_aligned import _latest_teacher_ckpt; print(_latest_teacher_ckpt())')"
export DMD_TEACHER_CKPT
echo "CONFIG: ${CONFIG}"
echo "TAG: ${TAG}"
echo "DMD student checkpoint: checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-40000.pt"
echo "DMD teacher checkpoint: ${DMD_TEACHER_CKPT}"

# --- Safety guard: refuse a teacher that never advanced past the resume start.
MIN_TEACHER_STEP=13000
TEACHER_STEP="$(echo "$DMD_TEACHER_CKPT" | sed -E 's#.*checkpoint-([0-9]+).*#\1#')"
echo "DMD teacher step: ${TEACHER_STEP} (must exceed ${MIN_TEACHER_STEP})"
if [ -z "$TEACHER_STEP" ] || [ "$TEACHER_STEP" -le "$MIN_TEACHER_STEP" ]; then
    echo "ERROR: teacher did not advance past step ${MIN_TEACHER_STEP}; not launching." >&2
    exit 1
fi

# --- Guard: fresh start; refuse to silently resume a stale state bundle.
if [ -f "checkpoints/ar_wm/${TAG}/training_state.pt" ]; then
    echo "NOTE: checkpoints/ar_wm/${TAG}/training_state.pt exists -- this run will" \
         "RESUME it. Remove it for a fresh start." >&2
fi

set -x
exec "${TORCHRUN:-.venv/bin/torchrun}" --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
    -m openworld.autoregressive.train_self_forcing \
    --config "$CONFIG"
