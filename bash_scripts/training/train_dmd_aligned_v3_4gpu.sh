#!/bin/bash
#SBATCH --job-name=dmd_aligned_v3_4gpu
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_outputs/ar_dmd_aligned_v3_4gpu/%j.out
#SBATCH --error=slurm_outputs/ar_dmd_aligned_v3_4gpu/%j.out
#
# Smaller-scale 4-GPU / 6h TEST of the v3 DMD distillation run.
# Runs ALONGSIDE the queued 8-GPU v3 job -- it writes to its OWN tag/output dir
# (ar_wan_dmd_aligned_v3_4gpu) via configs/training/ar_wan_droid_aligned_v3_4gpu.py
# and uses a distinct master_port, so the two jobs never share checkpoints.
# Half the GPUs -> half the global batch vs the 8-GPU run; this is a feasibility
# / smoke test, not an apples-to-apples reproduction.

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"
mkdir -p slurm_outputs/ar_dmd_aligned_v3_4gpu

CONFIG=configs/training/ar_wan_droid_aligned_v3_4gpu.py
GPUS="${GPUS:-4}"

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

# --- Guard: clean restart; the 4gpu output dir must not already carry a resume
#     bundle (otherwise we'd silently continue an earlier 4gpu attempt).
if [ -f "checkpoints/ar_wm/ar_wan_dmd_aligned_v3_4gpu/training_state.pt" ]; then
    echo "NOTE: checkpoints/ar_wm/ar_wan_dmd_aligned_v3_4gpu/training_state.pt exists" \
         "-- this run will RESUME that 4gpu state. Remove it for a fresh start." >&2
fi

set -x
exec "${TORCHRUN:-.venv/bin/torchrun}" --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29519}" \
    -m openworld.autoregressive.train_self_forcing \
    --config "$CONFIG"
