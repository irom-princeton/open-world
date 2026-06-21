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
#SBATCH --output=slurm_outputs/ar_gpu/%j.out
#SBATCH --error=slurm_outputs/ar_gpu/%j.out
#
# Generic (non-interactive) GPU runner for ad-hoc commands — the `sbatch` analog
# of an interactive salloc. The per-stage launchers in
# bash_scripts/{data_process,training,inference}/ are preferred for the standard
# pipeline; use this for one-offs (tests, smoke, a quick python -c).
#
# Usage:
#   sbatch bash_scripts/ar_gpu.slurm <command...>
# e.g.
#   sbatch bash_scripts/ar_gpu.slurm .venv/bin/python -m pytest openworld/autoregressive/tests -q
#   sbatch bash_scripts/ar_gpu.slurm .venv/bin/python -m openworld.autoregressive.train_self_forcing --smoke
#
# Monitor:  squeue -j <jobid>   ·   tail -f slurm_outputs/ar_<jobid>.out

source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"

# Default command if none given: unit tests + Wan/Cosmos wiring on GPU.
if [ "$#" -eq 0 ]; then
  set -- "$PY" -m pytest openworld/autoregressive/tests -q
fi

echo "+ $*"
exec "$@"
