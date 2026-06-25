# Scene Generation — building test-case suites from language + image

`openworld.scenegen` turns **one language instruction + one initial image** into a
ready-to-roll-out **Initialization suite** — the format consumed by the policy-eval
pipeline (`scripts/run_evaluation.py`; see [EVAL.md](EVAL.md)). It is the *directed*
counterpart to the autonomous red-team loop: you supply the instruction and image.

```
instruction + wrist image
   ▼
[1] guardrail   rewrite instruction → nanobanana-ready edit prompt   (openworld/scenegen/guardrail.py)
[2] nanobanana  edit the wrist view                                  (Gemini 2.5 Flash Image)
[3] multiview   complete the 2 side views                            (FLUX.2-klein; stages 2+3 = one subprocess)
[4] assembly    resize + write initialization.yaml per case          (openworld/scenegen/pipeline.py)
   ▼
<suite>/init_*/{wrist,exterior_left,exterior_right}.png + initialization.yaml
```

The output suite is what a policy-eval config's `dataset_path` points at — so
**scene generation produces the benchmark that [policy evaluation](EVAL.md) runs on.**

---

## Layout

| Piece | Path |
|-------|------|
| package (guardrail + pipeline + nanobanana) | `openworld/scenegen/` |
| **suite-from-spec CLI** (nanobanana all-views) | `scripts/scenegen/build_suite.py` |
| **single-case CLI** (multiview add-object) | `scripts/generate_test_case.py` |
| **batch suite-authoring tools** | `scripts/scenegen/` |
| base view sets (`tri`, `irom`) | `assets/<base>/` |
| configs (state template, multiview template, suite specs) | `configs/scenegen/` |

---

## The guardrail

nanobanana is phrasing-sensitive: a bare `"put the carrot in the bowl"` often
shifts the camera, moves the gripper, or occludes objects. The guardrail rewrites
the instruction into the top-down wrist-cam edit format nanobanana wants. Two
backends (`--guardrail-backend`):

- **`gemini`** (default): `gemini-2.5-flash` rewrites it (`guardrail.SYSTEM_PROMPT`);
  needs `GOOGLE_API_KEY`, falls back to `template` on any error.
- **`template`**: deterministic wrapper — no network, fully reproducible.

Reusable on its own: `from openworld.scenegen.guardrail import build_edit_prompt`.

## Prerequisites

| Need | How |
|------|-----|
| `google-genai` | `uv sync --extra scenegen` |
| **`GOOGLE_API_KEY`** | `export GOOGLE_API_KEY=<key>` (https://aistudio.google.com/apikey); needs network |
| diffusers fork | `git clone https://github.com/tenny-yinyijun/diffusers external/diffusers` (FLUX.2-klein lives only there) |
| multiview checkpoint | `bash external/download_models.sh` → `checkpoints/multiview_droid_v0` |
| FLUX.2-klein-4B | auto-downloads first run; for offline GPU nodes pre-cache + `HF_HUB_OFFLINE=1` |
| GPU | the multiview stage loads ~8 GB |

> **Branch note.** `autoregressive` ships PyPI `diffusers==0.34.0` (no
> `Flux2KleinPipeline`), so the pipeline prepends `external/diffusers/src` to the
> subprocess `PYTHONPATH` to import the fork. If the fork lives in its own venv,
> pass `--python-exec /path/to/venv/bin/python`.

---

## 0. Suite from a spec — `scripts/scenegen/build_suite.py`  (recommended)

The most usable path: pick a **base** view set and list the **edits** you want in
one YAML file. Each edit nanobanana-edits all three views of the base and becomes
one `init_<i>` case under `data/benchmark/<name>/`. **No GPU, no FLUX** — just
`GOOGLE_API_KEY` (`uv sync --extra scenegen`). This is the nanobanana all-views
mode (background / lighting / material / content edits); use the multiview
add-object path (§1–2) only when you need to *introduce a new object*.

```bash
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example.yaml
# -> data/benchmark/example_suite/init_*/{wrist,exterior_left,exterior_right}.png + initialization.yaml
```

**Bases** live in [`assets/`](../assets/README.md) — each is three views plus a
`template.yaml` (robot start state + default `scene` tag):

| Base | Source views |
|------|--------------|
| `tri`  | `data/benchmark/0617_generated/_base_original/` |
| `irom` | `open-world/data/benchmark/irom_carrot_pnp/init_2/` |

**Spec** (`configs/scenegen/suites/example.yaml`): `base` (a name under `assets/`
or a path to three views), `name` (→ `data/benchmark/<name>`), an optional shared
`keep` clause appended to every edit prompt, and the `edits` list. Each edit sets
`prompt` (the nanobanana edit), `instruction` (the policy command), an optional
`label`, an optional per-edit `keep` override, and an optional `views` subset.

```yaml
base: tri
name: my_suite
keep: >
  Keep the robot arm, camera viewpoint, framing and all object positions exactly
  the same; do not move, remove, add, warp, or recolor any object.
edits:
  - label: green_tabletop
    instruction: put the mug in the white container
    prompt: Change ONLY the wooden tabletop to a solid matte green tabletop.
```

CLI flags `--base`, `--name`/`--out-dir`, `--start-index` override the spec (use
`--start-index` to append to an existing suite). Library entry point:
`from openworld.scenegen import build_suite_from_spec`.

---

## 1. Single case — `scripts/generate_test_case.py`

```bash
GOOGLE_API_KEY=... uv run python scripts/generate_test_case.py \
    --instruction "put the carrot in the bowl" \
    --init-image external/diffusers/assets/droid/wrist.jpg \
    --name carrot_in_bowl \
    --num-cases 3
# -> data/initializations/carrot_in_bowl/
```

`--name` places the suite under `data/initializations/`; pass `--out-suite PATH`
for an explicit location. Then point an eval config's `dataset_path` at the suite
and run it (see [EVAL.md](EVAL.md)).

Useful flags: `--scene-edit` (decouple what's drawn from the policy command),
`--num-cases`/`--seed` (case `i` uses `seed+i`), `--guardrail-backend template`
(offline), `--side-cond` (override side images), `--start-index` (append to a
suite), `--keep-raw`, `--template-init`.

## 2. Batch suite authoring — `scripts/scenegen/`

These tools compose `generate_test_case.py` + nanobanana into full multi-case
suites. They are the **recipes used to author the `0617_generated` suite**; the
case/theme lists inside are that recipe — copy and edit them (or override paths
via env) for a new suite. All paths default to the 0617 setup and are env-overridable.

| Script | Mode | What it does |
|--------|------|--------------|
| `remove_object.py` | object removal | nanobanana-erase a named object from each view → an empty-table **base** (`--object`, `--src-dir`, `--dst-dir`) |
| `make_suite_add_object.sh` | add object (multiview) | from the empty base, generate one case per object (mug variants, carrot, lemon, …) via `generate_test_case.py` |
| `make_suite_background_edit.py` | background / lighting | nanobanana-edit all 3 views with a theme (green tabletop, forest, restaurant, red light), objects fixed |
| `nanobanana_edit.py` | helper | reusable upscale→edit→downscale single-image nanobanana edit (library + CLI) |

```bash
# empty-table base:
GOOGLE_API_KEY=... python scripts/scenegen/remove_object.py \
    --object "green mug" --src-dir <suite>/_base_original --dst-dir <suite>/_base_no_mug
# add-object cases (needs GPU + diffusers fork):
GOOGLE_API_KEY=... OUT=<suite> bash scripts/scenegen/make_suite_add_object.sh
# background/lighting cases:
GOOGLE_API_KEY=... SUITE=<suite> python scripts/scenegen/make_suite_background_edit.py
```

---

## Output

```
<suite>/
├── scenegen_manifest.json     # provenance (single-case CLI): instruction, edit_prompt, seeds, cases
└── init_*/                    # wrist.png + exterior_{left,right}.png (320×192) + initialization.yaml
```

Each `initialization.yaml` clones `initial_state` from the template
(`configs/scenegen/template_*.yaml`), sets `instruction`, and records the
guardrail's prompt under `metadata.edit_prompt`. `initial_observation` is omitted
— `InitializationDataset` infers it from the PNGs.

**Next:** run a policy in a world model over the suite → [EVAL.md](EVAL.md).
