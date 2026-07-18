#!/usr/bin/env bash
# Batch suite authoring — "add object" mode (multiview).
# ============================================================================
# From an empty-table base view, generate one Initialization case per object by
# editing the wrist view (nanobanana) + completing the two side views (FLUX.2),
# via scripts/generate_test_case.py. This is the recipe used to author init_0..7
# of the 0617_generated suite; the case list below is that recipe — edit it (or
# the env paths) to author your own suite.
#
# Prereqs (see docs/SCENEGEN.md): GOOGLE_API_KEY, the diffusers fork + multiview
# checkpoint, a GPU. Override any path via env, e.g.  OUT=/my/suite bash ...
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# --- configurable paths (defaults reproduce the 0617 suite) -----------------
BASEDIR="${BASEDIR:-/scratch/gpfs/AM43/yy4041/open-world/data/benchmark/0617_generated/_base_no_mug}"
SIDEDIR="${SIDEDIR:-data/_scenegen_sides/0617_vid15_640}"
FORK="${FORK:-/scratch/gpfs/AM43/yy4041/diffusers}"          # diffusers fork (FLUX.2-klein)
OUT="${OUT:-/scratch/gpfs/AM43/yy4041/open-world/data/benchmark/0617_generated}"
TPL="${TPL:-configs/scenegen/template_vid15.yaml}"
LOGDIR="${LOGDIR:-slurm_outputs/scenegen_add_object}"
mkdir -p "$LOGDIR"

run() {  # idx  instruction  scene_edit  seed
  .venv/bin/python scripts/generate_test_case.py \
    --instruction "$2" --scene-edit "$3" \
    --init-image "$BASEDIR/wrist.png" --out-suite "$OUT" \
    --num-cases 1 --start-index "$1" --seed "$4" --guardrail-backend template \
    --python-exec "$FORK/.venv/bin/python" --diffusers-dir "$FORK" \
    --multiview-script "$FORK/examples/inference/multiview_droid_with_nanobanana.py" \
    --checkpoint-path "$FORK/checkpoints/multiview_droid_v0" \
    --side-cond "$SIDEDIR/side1.png" --side-cond "$SIDEDIR/side2.png" \
    --template-init "$TPL" --scene "vid15_multiview" \
    > "$LOGDIR/case_$1.log" 2>&1
  echo "init_$1 done: $(tail -1 "$LOGDIR/case_$1.log")"
}

# --- the 0617 recipe: (a) add a mug variant, (b) add a non-mug object -------
run 0 "put the red mug in the white container"     "add a red mug on the table"                      0
run 1 "put the white mug in the white container"   "add a white mug on the table"                    1
run 2 "put the glass mug in the white container"   "add a clear transparent glass mug on the table"  2
run 3 "put the rainbow mug in the white container" "add a rainbow colored mug on the table"          3
run 4 "put the carrot in the white container"      "add a carrot on the table"                       4
run 5 "put the lemon in the white container"       "add a lemon on the table"                        5
run 6 "put the teddy bear in the white container"  "add a small stuffed teddy bear on the table"     6
run 7 "put the scissors in the white container"    "add a pair of scissors on the table"             7
echo "add-object suite done."
