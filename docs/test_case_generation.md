# Test-Case Generation from Language + Image

`openworld.scenegen` turns **one language instruction + one initial image** into a
ready-to-roll-out **Initialization suite** (the handoff format consumed by
`scripts/run_evaluation.py`; see `docs/tmp_AGENT_GUIDE.md` §3). It is the
*directed* counterpart to the autonomous `openworld.redteam` loop — you supply
the instruction and image instead of an LLM proposing them.

```
instruction + wrist image
   ▼
[1] guardrail   rewrite instruction → nanobanana-ready edit prompt  (openworld/scenegen/guardrail.py)
[2] nanobanana  edit the wrist view            (Gemini 2.5 Flash Image)
[3] multiview   complete the 2 side views      (FLUX.2-klein; stages 2+3 = one subprocess)
[4] assembly    resize + write initialization.yaml per case         (openworld/scenegen/pipeline.py)
   ▼
data/initializations/<name>/init_*/{wrist,exterior_left,exterior_right}.png + initialization.yaml
```

## The guardrail

nanobanana is phrasing-sensitive: a bare `"put the carrot in the bowl"` often
shifts the camera, moves the gripper, or occludes objects. The guardrail rewrites
the instruction into the top-down wrist-cam edit format nanobanana wants. Two
backends (`--guardrail-backend`):

- **`gemini`** (default): `gemini-2.5-flash` rewrites it (see
  `guardrail.SYSTEM_PROMPT`); needs `GOOGLE_API_KEY`, falls back to `template`
  on any error.
- **`template`**: deterministic wrapper — no network, fully reproducible.

Reusable on its own:

```python
from openworld.scenegen.guardrail import build_edit_prompt
build_edit_prompt("add a red apple to the left of a blue plate")
```

## Prerequisites

| Need | How |
|------|-----|
| `google-genai` | `uv sync --extra scenegen` |
| **`GOOGLE_API_KEY`** | `export GOOGLE_API_KEY=<key>` (https://aistudio.google.com/apikey); needs network |
| diffusers fork | `git clone https://github.com/tenny-yinyijun/diffusers external/diffusers` (see `external/README.md`) |
| multiview checkpoint | `bash external/download_models.sh` → `checkpoints/multiview_droid_v0` |
| FLUX.2-klein-4B | auto-downloads on first run; for offline GPU nodes pre-cache it and set `HF_HUB_OFFLINE=1` |
| GPU | the multiview stage loads ~8 GB |

> **Branch note.** `autoregressive` ships PyPI `diffusers==0.34.0` (no
> `Flux2KleinPipeline`), so the pipeline puts `external/diffusers/src` on the
> subprocess `PYTHONPATH` to import the fork. If the fork lives in its own venv,
> pass `--python-exec /path/to/venv/bin/python`.

## Usage

```bash
GOOGLE_API_KEY=... uv run python scripts/generate_test_case.py \
    --instruction "put the carrot in the bowl" \
    --init-image external/diffusers/assets/droid/wrist.jpg \
    --name carrot_in_bowl \
    --num-cases 3
# -> data/initializations/carrot_in_bowl/
```

`--name` places the suite under `data/initializations/`; pass `--out-suite PATH`
for an explicit location instead. (`bash_scripts/generate_test_case.sh
"<instruction>" <image> <name> <n>` wraps this.) Then point an eval config's
`dataset_path` at the suite and run `scripts/run_evaluation.py` (see
`docs/tmp_AGENT_GUIDE.md` §4).

Useful flags: `--scene-edit` (decouple what's drawn from the policy command),
`--num-cases`/`--seed` (case `i` uses `seed+i`), `--guardrail-backend template`
(offline), `--side-cond` (override side images), `--start-index` (append to a
suite), `--keep-raw`, `--template-init`.

## Output

```
data/initializations/<name>/
├── scenegen_manifest.json     # provenance: instruction, edit_prompt, seeds, cases
└── init_*/                    # wrist.png + exterior_{left,right}.png (320×192) + initialization.yaml
```

Each `initialization.yaml` clones `initial_state` from `--template-init`, sets
`instruction`, and records the guardrail's prompt under `metadata.edit_prompt`.
`initial_observation` is omitted — `InitializationDataset` infers it from the PNGs.
