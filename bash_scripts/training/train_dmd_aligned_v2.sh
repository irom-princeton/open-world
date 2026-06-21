#!/bin/bash
#SBATCH --job-name=dmd_aligned_v2
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_outputs/ar_dmd_aligned_v2/%j.out
#SBATCH --error=slurm_outputs/ar_dmd_aligned_v2/%j.out
#
# Stage L0 self-forcing / DMD distillation, aligned -- CLEAN RESTART (v2).
# Same as train_dmd_aligned.sh but:
#   * config = ar_wan_droid_aligned_v2.py  (tag ar_wan_dmd_aligned_v2 -> NEW
#     output dir, so it does NOT resume the diverged v1 training_state.pt)
#   * 8h wall-clock
# Generator <- ar_wan_studentinit_aligned/checkpoint-40000.pt; teacher + critic
# backbone <- latest checkpoints/ar_wm/ar_wan_teacher_aligned/ (resolved at start).
# Carries the post-v1 fixes: whole-clip scoring + mean-reduced losses.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"
mkdir -p slurm_outputs/ar_dmd_aligned_v2

CONFIG=configs/training/ar_wan_droid_aligned_v2.py
GPUS="${GPUS:-8}"

# --- Resolve + pin the teacher checkpoint (config's own resolver). -----------
DMD_TEACHER_CKPT="$(.venv/bin/python -c 'from configs.training.ar_wan_droid_aligned import _latest_teacher_ckpt; print(_latest_teacher_ckpt())')"
export DMD_TEACHER_CKPT
echo "DMD student checkpoint: checkpoints/ar_wm/ar_wan_studentinit_aligned/checkpoint-40000.pt"
echo "DMD teacher checkpoint: ${DMD_TEACHER_CKPT}"

# --- Safety guard: refuse a teacher that never advanced past the resume start.
MIN_TEACHER_STEP=13000
TEACHER_STEP="$(echo "$DMD_TEACHER_CKPT" | sed -E 's#.*checkpoint-([0-9]+).*#\1#')"
echo "DMD teacher step: ${TEACHER_STEP} (must exceed ${MIN_TEACHER_STEP})"
if [ -z "$TEACHER_STEP" ] || [ "$TEACHER_STEP" -le "$MIN_TEACHER_STEP" ]; then
    echo "ERROR: teacher did not advance past step ${MIN_TEACHER_STEP}; the resume" \
         "likely failed. Not launching DMD." >&2
    exit 1
fi

# --- Guard: this is a clean restart; the v2 output dir must not already carry a
#     resume bundle (otherwise we'd silently continue an earlier v2 attempt).
if [ -f "checkpoints/ar_wm/ar_wan_dmd_aligned_v2/training_state.pt" ]; then
    echo "NOTE: checkpoints/ar_wm/ar_wan_dmd_aligned_v2/training_state.pt exists" \
         "-- this run will RESUME that v2 state (not v1). Remove it for a fresh start." >&2
fi

set -x
exec "${TORCHRUN:-.venv/bin/torchrun}" --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29517}" \
    -m openworld.autoregressive.train_self_forcing \
    --config "$CONFIG"
