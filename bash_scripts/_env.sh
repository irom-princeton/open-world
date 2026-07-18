#!/bin/bash
# Shared environment for the AR world-model sbatch launchers. Source this *after*
# the `#SBATCH` header (those directives must be the first lines of the file):
#
#     source "${SLURM_SUBMIT_DIR:-$(pwd)}/bash_scripts/_env.sh"
#
# It cds to the submit dir (so relative paths like `scripts/…` and `configs/…`
# resolve), makes the slurm log dir, forces offline HF loading (compute nodes
# have no internet — weights come from the local `external/` dir), and prints the
# node/GPU for the log.
#
# NOTE: deliberately `-eo pipefail` WITHOUT `-u` (nounset). On the compute nodes
# the startup bashrc / module / conda init references unbound variables (e.g.
# $PS1, $CONDA_SHLVL, $LMOD_*), which `set -u` would turn into a fatal error
# before our command runs. We keep -e (fail fast on a bad exit code) and
# pipefail (propagate failures through pipes); optional env vars are guarded with
# ${VAR:-default} so -u would buy little anyway.
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs

# Compute nodes are offline. The Wan/Cosmos pipeline loads all weights from the
# local `external/` dir (bash_scripts/download_weights.sh); HF_HOME is defaulted
# to a codebase-local cache so everything is self-contained. Override with
# `export HF_HOME=...` before sbatch if a backbone resolves a bare HF repo id.
export HF_HOME="${HF_HOME:-$(pwd)/external/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# wandb must not phone home on an offline node (its init blocks ~90s then errors,
# killing training before step 1). "offline" logs locally for an optional later
# `wandb sync`; set WANDB_MODE=disabled to skip logging entirely.
export WANDB_MODE="${WANDB_MODE:-offline}"

PY="${PY:-.venv/bin/python}"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  gpu=${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
