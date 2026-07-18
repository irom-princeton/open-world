#!/bin/bash
# ============================================================================
# One-shot setup for the self-contained pi0.5 policy-eval stack (3 world models).
# ============================================================================
# Run this from a node WITH internet (a Della login node) — the compute nodes
# are offline. It is idempotent; existing venvs are kept unless --force.
#
#   bash bash_scripts/setup_eval_env.sh [--submodules] [--venv-eval]
#                                       [--venv-weaver] [--links] [--force]
#
# With no flags it runs everything except the heavy .venv-weaver build (pass
# --venv-weaver to also build that, or --force to rebuild existing venvs).
#
# Pieces:
#   submodules   external/openpi (tenny-yinyijun/openpi) + external/WEAVER
#                (arnavkj1995/WEAVER), pinned to the validated commits.
#   venv-eval    .venv-eval : torch + jax/openpi (ar, ctrlworld, weaver server).
#   venv-weaver  .venv-weaver : torch-2.7 / diffusers-0.35 for WEAVER, built from
#                external/WEAVER/pyproject.toml. Heavy (flash-attn compile).
#   links        cluster-specific checkpoint/data symlinks (weights stay external):
#                  external/{stable-video-diffusion-img2vid,clip-vit-base-patch32}
#                  checkpoints/{action_adapter/model2_15_9.pth,weaver,ctrlworld}
#                  data/benchmark/0617_generated
# ----------------------------------------------------------------------------
set -eo pipefail
OW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$OW"

DO_SUB=0; DO_EVAL=0; DO_WEAVER=0; DO_LINKS=0; FORCE=0; ANY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --submodules)  DO_SUB=1; ANY=1 ;;
    --venv-eval)   DO_EVAL=1; ANY=1 ;;
    --venv-weaver) DO_WEAVER=1; ANY=1 ;;
    --links)       DO_LINKS=1; ANY=1 ;;
    --force)       FORCE=1 ;;
    *) echo "[setup] unknown arg: $1" >&2; exit 2 ;;
  esac; shift
done
if [ "$ANY" -eq 0 ]; then DO_SUB=1; DO_EVAL=1; DO_LINKS=1; fi   # default: all but weaver

# Source locations for the external weights/data (override via env if they move).
OPENWORLD_REPO="${OPENWORLD_REPO:-/scratch/gpfs/AM43/yy4041/open-world}"
WEAVER_EXT="${WEAVER_EXT:-/scratch/gpfs/AM43/yy4041/WEAVER}"

# ---- submodules -------------------------------------------------------------
if [ "$DO_SUB" -eq 1 ]; then
  echo "[setup] initializing submodules (external/openpi, external/WEAVER)..."
  # Non-recursive: the WM adapters only need the top-level packages. (WEAVER's own
  # third_party/* submodules are for its data-gen/reward pipeline, not WM eval.)
  git submodule update --init external/openpi external/WEAVER
fi

# ---- .venv-eval (torch + jax/openpi) ----------------------------------------
if [ "$DO_EVAL" -eq 1 ]; then
  if [ -x "$OW/.venv-eval/bin/python" ] && [ "$FORCE" -eq 0 ]; then
    echo "[setup] .venv-eval exists (use --force to rebuild); skipping."
  else
    echo "[setup] building .venv-eval (uv sync --extra policy-openpi)..."
    UV_PROJECT_ENVIRONMENT="$OW/.venv-eval" uv sync --extra policy-openpi
  fi
fi

# ---- .venv-weaver (torch-2.7 / diffusers-0.35) ------------------------------
if [ "$DO_WEAVER" -eq 1 ]; then
  if [ ! -f "$OW/external/WEAVER/pyproject.toml" ]; then
    echo "[setup] external/WEAVER not initialized — run with --submodules first." >&2; exit 1
  fi
  if [ -x "$OW/.venv-weaver/bin/python" ] && [ "$FORCE" -eq 0 ]; then
    echo "[setup] .venv-weaver exists (use --force to rebuild); skipping."
  else
    echo "[setup] building .venv-weaver from external/WEAVER (heavy: torch-2.7 + flash-attn)..."
    ( cd "$OW/external/WEAVER" && UV_PROJECT_ENVIRONMENT="$OW/.venv-weaver" uv sync )
    echo "[setup] NOTE: if the model needs flash-attn, add it: "
    echo "        ( cd external/WEAVER && UV_PROJECT_ENVIRONMENT=$OW/.venv-weaver uv sync --extra flash-attn )"
  fi
fi

# ---- checkpoint/data symlinks (weights stay external) -----------------------
if [ "$DO_LINKS" -eq 1 ]; then
  echo "[setup] creating checkpoint/data symlinks (sources stay external)..."
  mkdir -p external checkpoints/action_adapter data/benchmark
  link() {  # link <target> <linkname>  (skip with a warning if target missing)
    if [ -e "$1" ]; then ln -sfn "$1" "$2"; echo "  $2 -> $1";
    else echo "  WARN: source missing, skipped: $1" >&2; fi
  }
  link "$OPENWORLD_REPO/external/stable-video-diffusion-img2vid" external/stable-video-diffusion-img2vid
  link "$OPENWORLD_REPO/external/clip-vit-base-patch32"          external/clip-vit-base-patch32
  link "$OPENWORLD_REPO/checkpoints/action_adapter/model2_15_9.pth" checkpoints/action_adapter/model2_15_9.pth
  link "$OPENWORLD_REPO/checkpoints/wm/ctrlworld/ctrlworld"      checkpoints/ctrlworld
  link "$WEAVER_EXT/checkpoints/WEAVER"                          checkpoints/weaver
  link "$OPENWORLD_REPO/data/benchmark/0617_generated"          data/benchmark/0617_generated
fi

echo "[setup] done. Launch an eval with:  sbatch bash_scripts/eval_wm.sbatch --wm {ctrlworld|ar|weaver}"
