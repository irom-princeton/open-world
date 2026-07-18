#!/usr/bin/env python
"""Build a world-model test-case suite from a language instruction + an image.

Pipeline (see openworld/scenegen/): a *guardrail* rewrites your plain instruction
into a nanobanana-ready edit prompt, the bundled diffusers pipeline edits the
wrist view and completes the two side views, and the three views are assembled
into an Initialization suite that scripts/run_evaluation.py can roll out.

Prerequisites:
  - GOOGLE_API_KEY exported (nanobanana + the gemini guardrail backend).
  - the diffusers fork at external/diffusers (see external/README.md).
  - the multiview checkpoint at checkpoints/multiview_droid_v0
    (bash external/download_models.sh), plus FLUX.2-klein-4B cached / reachable.
  - a GPU for the multiview stage; `uv sync --extra scenegen` for google-genai.

Example:
    GOOGLE_API_KEY=... uv run python scripts/generate_test_case.py \\
        --instruction "put the carrot in the bowl" \\
        --init-image external/diffusers/assets/droid/wrist.jpg \\
        --name carrot_in_bowl \\
        --num-cases 3
    # -> writes data/initializations/carrot_in_bowl/

Then roll out + score (write a config whose dataset_path points at the suite):
    uv run python scripts/run_evaluation.py --config configs/evaluation/<your>.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/generate_test_case.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openworld.scenegen.guardrail import DEFAULT_GEMINI_MODEL
from openworld.scenegen.pipeline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DIFFUSERS_DIR,
    DEFAULT_MULTIVIEW_SCRIPT,
    DEFAULT_TEMPLATE_INIT,
    generate_test_cases,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a test-case suite from a language instruction + image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core inputs.
    p.add_argument("--instruction", required=True,
                   help="Plain-language task command; stored as the case instruction "
                        "and (unless --scene-edit is given) drives the scene edit.")
    p.add_argument("--init-image", required=True,
                   help="Initial wrist-camera image to edit (the 'already provided' image).")
    p.add_argument("--name", default=None,
                   help="Suite name; output goes to data/initializations/<name>. "
                        "Required unless --out-suite is given.")
    p.add_argument("--out-suite", default=None,
                   help="Explicit output suite directory (overrides --name).")
    p.add_argument("--num-cases", type=int, default=1,
                   help="How many cases (init_<i>) to generate from this instruction.")
    p.add_argument("--scene-edit", default=None,
                   help="Override what nanobanana edits (defaults to --instruction). "
                        "Use when the visual scene differs from the policy command.")

    # Guardrail.
    p.add_argument("--guardrail-backend", choices=["gemini", "template"], default="gemini",
                   help="'gemini' rewrites the instruction with an LLM; 'template' wraps "
                        "it deterministically (no network).")
    p.add_argument("--guardrail-model", default=DEFAULT_GEMINI_MODEL,
                   help="Gemini text model for the guardrail rewrite.")

    # Multiview / diffusers fork.
    p.add_argument("--diffusers-dir", default=str(DEFAULT_DIFFUSERS_DIR),
                   help="Path to the diffusers fork (its src/ is put on PYTHONPATH).")
    p.add_argument("--multiview-script", default=str(DEFAULT_MULTIVIEW_SCRIPT),
                   help="multiview_droid_with_nanobanana.py inside the fork.")
    p.add_argument("--python-exec", default=None,
                   help="Interpreter for the multiview subprocess "
                        "(default: this process's python). Point at the fork's venv "
                        "if you keep diffusers separately installed.")
    p.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT),
                   help="multiview_droid_v0 checkpoint directory.")
    p.add_argument("--side-cond", action="append", default=None,
                   help="Side conditioning image; repeat for each side. "
                        "Defaults to the fork's assets/droid/side{1,2}.jpg.")
    p.add_argument("--num-inference-steps", type=int, default=50,
                   help="FLUX denoising steps (lower = faster, higher = better).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; case i uses seed+i.")

    # Suite assembly.
    p.add_argument("--template-init", default=str(DEFAULT_TEMPLATE_INIT),
                   help="initialization.yaml whose initial_state is cloned into every case.")
    p.add_argument("--width", type=int, default=320, help="Suite image width (world-model res).")
    p.add_argument("--height", type=int, default=192, help="Suite image height (world-model res).")
    p.add_argument("--scene", default="scenegen", help="metadata.scene tag.")
    p.add_argument("--task-type", default="manipulation", help="metadata.task_type tag.")
    p.add_argument("--start-index", type=int, default=0,
                   help="First case index (use to append to an existing suite).")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the intermediate full-res edits under <suite>/_raw/.")
    args = p.parse_args()

    generate_test_cases(
        instruction=args.instruction,
        init_image=args.init_image,
        name=args.name,
        out_suite=args.out_suite,
        num_cases=args.num_cases,
        scene_edit=args.scene_edit,
        guardrail_backend=args.guardrail_backend,
        guardrail_model=args.guardrail_model,
        multiview_script=args.multiview_script,
        diffusers_dir=args.diffusers_dir,
        python_exec=args.python_exec,
        checkpoint_path=args.checkpoint_path,
        side_cond=args.side_cond,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        template_init=args.template_init,
        target_size=(args.width, args.height),
        scene=args.scene,
        task_type=args.task_type,
        start_index=args.start_index,
        keep_raw=args.keep_raw,
    )


if __name__ == "__main__":
    main()
