#!/bin/bash
#SBATCH --job-name=dmd_aligned_v3
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/ar_dmd_aligned_v3/%j.out
#SBATCH --error=slurm_outputs/ar_dmd_aligned_v3/%j.out
#
# Stage L0 self-forcing / DMD distillation, aligned -- v3 (clean restart).
# Same inits as v2 (aligned student-init 40k + latest aligned teacher) but with
# the post-v2 fixes:
#   * config = ar_wan_droid_aligned_v3.py (tag ar_wan_dmd_aligned_v3 -> NEW dir)
#   * gen learning_rate 2e-6 -> 1e-6 (generator was outpacing the critic and
#     eroding per-view motion routing in this 8-GPU under-batched setup)
#   * bug-A cond-alignment fix (in distill/self_forcing.py)
#   * checkpoints every 200 steps; full instrumentation logged every 20
#   * 4h wall-clock

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"
mkdir -p slurm_outputs/ar_dmd_aligned_v3

CONFIG=configs/training/ar_wan_droid_aligned_v3.py
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

# --- Guard: clean restart; the v3 output dir must not already carry a resume
#     bundle (otherwise we'd silently continue an earlier v3 attempt).
if [ -f "checkpoints/ar_wm/ar_wan_dmd_aligned_v3/training_state.pt" ]; then
    echo "NOTE: checkpoints/ar_wm/ar_wan_dmd_aligned_v3/training_state.pt exists" \
         "-- this run will RESUME that v3 state. Remove it for a fresh start." >&2
fi

set -x
exec "${TORCHRUN:-.venv/bin/torchrun}" --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29518}" \
    -m openworld.autoregressive.train_self_forcing \
    --config "$CONFIG"
