# Handoff: make the 3-world-model pi0.5 policy-eval self-contained + unify the launcher

> This file is a context handoff for a **fresh Claude Code agent**. The previous
> session ran out of `/tmp` disk (the Bash tool could not write its output-capture
> file under `/tmp/claude-<pid>/...`, so no shell command could run). The user is
> relaunching with `TMPDIR` pointed at scratch. Read this top-to-bottom, then do
> the **TASK** at the end. Verify every path with `ls`/`Read` before acting on it —
> some paths below are from memory and may have drifted.

Repo: `/scratch/gpfs/AM43/yy4041/open-world-autoregressive` (branch `autoregressive`).
User: yy4041@princeton.edu (Tenny). Cluster: Della/Slurm (partition `ailab`, qos
`ailab`, account `am43`, `--gres=gpu:h200:1`). Cluster gotcha: sbatch scripts use
`set -eo pipefail` (NOT `-euo` — nounset trips on the compute-node bashrc).

Also read the persistent memory note `memory/policy-eval-3wm.md` (path:
`/home/yy4041/.claude/projects/-scratch-gpfs-AM43-yy4041-open-world-autoregressive/memory/policy-eval-3wm.md`)
and, if deeper detail is needed, the prior transcript:
`/home/yy4041/.claude/projects/-scratch-gpfs-AM43-yy4041-open-world-autoregressive/be5130af-8309-4daa-a4e5-0f3612ac208a.jsonl`

---

## 0. Current status (what already works — do NOT redo)

We run **pi0.5** (openpi `pi05_droid`) closed-loop inside **three world models** over a
benchmark of initializations, producing rollout videos (reward scoring is separate;
`reward_model: dummy`). All three currently produce **12 clean videos** on the
`0617_generated` suite (12 inits, `init_0`..`init_11`):

- **ctrl-world** (vidwm/SVD) — always fine.
- **AR slow student** (Wan block-causal DiT) — fixed via the "rate + persistent action buffer" fix.
- **weaver fast** — fixed via the "latent-state stepper" rewrite.

Outputs consolidated at `/scratch/gpfs/AM43/yy4041/open-world/outputs/0617_pi05/{ctrlworld,ar,weaver}/`
(note: under the **/open-world** repo path — that's where videos get viewed), with
buggy backups alongside (`ar_v1_ratebug`, `weaver_v1_roundtrip`, `ar_OLD_buggy`).

**The generation pipeline is correct and validated. The new task is purely about
code organization + launch UX — not about fixing rollout quality.**

---

## 1. How a policy-eval run actually executes

`scripts/run_evaluation.py` (reads an eval YAML) → spawns `scripts/generate_videos.py`
as a subprocess → builds a `WorldModelEnv` + `Evaluator` (`openworld/runners/evaluator.py`).
The policy (pi0.5) drives a `WorldModel` closed-loop over an `InitializationDataset`
(a dir of per-case `initialization.yaml` + per-view PNGs).

`WorldModel` base interface (`openworld/world_models/base_world_model.py`):
- `load_checkpoint(path)`
- `rollout(state, observation, action_chunk, instruction) -> {"frames": [...], "next_state": {...}}`
- Frames are height-stacked multiview `(V*H, W, 3)` uint8; the env feeds `frames[-1]`
  back as the next observation. `world_model.config.view_order` tells the Evaluator how
  to stack the initial obs to match predicted frames.

Registry plumbing (already committed, key enablers):
- `openworld/world_models/registry.py` — **lazy** imports; base registry only
  `{"dummy": DummyWorldModel}`; `_LAZY` dict with `_register_vidwm/_register_ar/_register_weaver`;
  `build_world_model(name)` calls `_LAZY[name]()`. This lets the driver import in venvs
  that lack jax/vidwm.
- `openworld/world_models/__init__.py` — `VidWMWorldModel`/`VidWMConfig` lazy via `__getattr__`.
- `openworld/runners/__init__.py` — `RLFineTuneRunner` (jax) lazy via `__getattr__` so
  Evaluator imports without jax.
- `openworld/policies/registry.py` — `build_policy` skips the jax/flax requirement when
  `server_url` is set (websocket mode for openpi).

---

## 2. The three world models — venvs, adapters, configs

**They live in incompatible venvs; there is no single unified stack.** This is the crux
of the self-containment task.

### (a) ctrl-world  (the "easy" one)
- Model `vidwm` (SVD). Historically run from the **/open-world** repo
  (`/scratch/gpfs/AM43/yy4041/open-world`), because that's where SVD/CLIP/openpi/robometer
  externals + benchmark data have lived.
- Config knobs: `flow_map_type: flow_matching`, `num_inference_steps: 50`,
  ckpt `checkpoints/wm/ctrlworld/ctrlworld/checkpoint-10000.pt`.
- Launcher historically: `/open-world` `bash_scripts/eval_tri_0609.sbatch` (+ `--gres=gpu:h200:1`).
- Config in THIS repo: `configs/evaluation/0617_ctrlworld_pi05.yaml` (verify it exists; it
  may currently live under /open-world).

### (b) AR slow student  (adapter already in this repo)
- Adapter: `openworld/world_models/ar_world_model.py` → `ARWanWorldModel`. (Read it; it's
  short and self-documenting.)
- Wraps `openworld.autoregressive.model.ARWorldModel` (block-causal DiT + KV cache),
  Wan VAE (4× temporal downsample), `frames_per_block=2`, `num_history_blocks=2`,
  3 cams height-stacked, `view_order=("exterior_right","exterior_left","wrist")`.
- "Slow" = `studentinit_aligned/checkpoint-40000.pt` (pre-distill; run ~32 flow-matching steps).
- Config: `config_path="configs/training/ar_wan_studentinit_droid_aligned.py"`,
  `stats_root="data/droid_ar_latents"` (has `stats.json`),
  `vae_dir="external/Wan2.1-T2V-1.3B-Diffusers"`.
- Eval config: `configs/evaluation/0617_ar_pi05.yaml` (`scheduler.chunk_size=8` — the rate fix).
- Runs in an **isolated `.venv-eval`** (autoregressive + policy-openpi extras), built with
  `uv` via `UV_PROJECT_ENVIRONMENT=.venv-eval`.
- External deps currently reached via **symlinks** from /open-world into this repo:
  `external/openpi`, `checkpoints/action_adapter/model2_15_9.pth`. **These cross-repo
  symlinks are exactly what the self-containment task should resolve** (vendor or copy,
  leaving only checkpoints external).

### (c) weaver fast  (adapter in this repo, but imports a sibling repo)
- Adapter: `openworld/world_models/weaver_world_model.py` → `WeaverWorldModel`. (Read it.)
- **PROBLEM TO FIX**: it does `sys.path.insert(0, "/scratch/gpfs/AM43/yy4041/WEAVER")`
  and `os.chdir(weaver_repo)`, importing `from weaver.generate_views import ...` and
  `from weaver.utils.tools import ...` from a **separate repo outside this one**. The
  default `weaver_repo="/scratch/gpfs/AM43/yy4041/WEAVER"` and
  `norm_stats_path=".../WEAVER/checkpoints/WEAVER/norm_stats_relabel.json"`.
- Runs in **`/scratch/gpfs/AM43/yy4041/WEAVER/.venv`** (torch 2.7 / diffusers 0.35 — cannot
  share this repo's venv).
- Knobs: `val_steps=4` (fast few-step distilled sampler), `horizon=8`, `bootstrap=5`,
  views `["wrist_left","exterior_1_left"]` mapped to dataset `("wrist","exterior_left")`,
  8-D joint+gripper deltas, `n_history=2`, `n_memory_frames=6`, `t_memory=5`.
- pi0.5 runs **out-of-process** as an openpi **websocket server** in `.venv-eval`, sharing
  the same H200 (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`);
  weaver connects via `ws://127.0.0.1:8123`.
- Eval config: `configs/evaluation/0617_weaver_pi05.yaml` (`chunk_size=5`, `horizon=8`,
  `bootstrap=5`, `server_url ws://127.0.0.1:8123`, action_adapter ON).
- Launcher: `bash_scripts/eval_weaver_0617.sbatch` — launches the pi05 server in `.venv-eval`
  background (PYTHONPATH includes `$OW/external/openpi/src` and
  `$OW/external/openpi/packages/openpi-client/src`), waits for the port via `/dev/tcp`,
  then runs `run_evaluation.py` in the weaver venv with `PYTHONPATH=$OW`.
  Server cmd: `serve_policy.py --env DROID05 --port $PORT policy:checkpoint
  --policy.config pi05_droid --policy.dir $PI_DIR`.

### Action sourcing (the user's design choice — keep it)
Derive each WM's actions from the env's **absolute robot state**, normalized per-WM:
- AR uses 7-D cartesian state directly (`state["_robot_state_history"]`, `stats.json` percentiles).
- weaver derives 8-D joint+gripper **deltas** from the joint state the cartesian
  action-adapter (`model2_15_9.pth`) keeps advanced — keep `action_adapter_checkpoint_path` set.

### Closed-loop gotchas (already fixed — do NOT regress them)
- **AR**: (1) persistent append-only **per-frame action buffer** — never re-align past
  frames' conditioning each step; (2) match the latent rate — Wan VAE is 4× temporal,
  fpb=2 → use `scheduler.chunk_size=8`.
- **weaver**: do NOT call `generate_videos_full` per env-step (decode→re-encode RGB =
  lossy VAE round-trip + memory reset → collapse). Carry weaver's predicted **latents**
  (`encode_obs` dict) + `memory_tokens` across steps; call `generate_latent_rollouts`
  per step under `model.ema.use_ema_weights()`; decode only for output. Weaver WM
  generation is task-agnostic (instructions only feed its reward/critic heads).

---

## 3. THE TASK (what the user actually wants now)

Two parts:

### Part 1 — Make all rollout-generation code self-contained under this repo
Everything needed to GENERATE the rollouts must live under
`/scratch/gpfs/AM43/yy4041/open-world-autoregressive`. **Checkpoints may stay external**
(do not copy multi-GB weights), but **code and repo references must not point outside**.

Concretely:
- **weaver**: install/vendor the WEAVER package as a dependency **under `external/`**
  (e.g. `external/WEAVER` or `external/weaver`) instead of `sys.path`-inserting
  `/scratch/gpfs/AM43/yy4041/WEAVER`. Update `weaver_world_model.py`'s default
  `weaver_repo`/import path accordingly. Decide between: (a) `git clone`/copy the WEAVER
  source into `external/` and `pip install -e` it into the weaver venv, or (b) add it as a
  path/VCS dependency. Prefer whatever matches how `external/openpi` and
  `external/Wan2.1-T2V-1.3B-Diffusers` are already vendored — **inspect `external/` first**
  to stay consistent. The weaver venv itself can stay where it is, or be rebuilt under the
  repo; confirm with the user if rebuilding is expensive.
- **AR**: replace the cross-repo symlinks (`external/openpi`,
  `checkpoints/action_adapter/model2_15_9.pth` pointing into /open-world) so the repo is
  self-sufficient. `external/openpi` should be a real vendored copy (or a submodule);
  `model2_15_9.pth` is a checkpoint so a symlink/external path is acceptable, but make the
  config reference explicit and documented.
- **ctrl-world**: it has historically run from the /open-world repo. Bring whatever is
  needed to launch a ctrl-world eval **from this repo** (config + sbatch + any vidwm code
  path), OR — if vidwm genuinely cannot be vendored cleanly — document precisely the minimal
  external dependency and confine it to checkpoints/data only. Check what `vidwm` import
  actually requires (it's lazy-registered in `registry.py`); the SVD/CLIP weights are
  checkpoints (external OK), but the *code* should be reachable in-repo.
- Verify the `0617_generated` benchmark data location and make the configs reference it by
  a stable in-repo or clearly-documented path (don't hardcode /open-world if avoidable).
- After changes, **re-validate** with a 2-init smoke at FULL duration (25 s) per WM before
  declaring done (short smokes miss back-half drift). Eyeball the wrist band ordering.

### Part 2 — Unify + simplify the launch UX (abstract into configs)
Make launching a policy-eval job easy and consistent across the three world models, with
**minimal arguments**. Design goals:
- One consistent entry point / sbatch template where a run is specified by: **world-model
  name + benchmark suite + checkpoint** (+ optional overrides). Everything else
  (venv selection, server launch for weaver, PYTHONPATH, step counts, view order, action
  adapter) should be abstracted into per-WM config so the user doesn't re-specify it.
- Keep the design **consistent with the existing infrastructure** (the eval YAML +
  `run_evaluation.py` → `generate_videos.py` flow, the registry pattern). Don't invent a
  parallel system; extend what's there.
- Consider a single `bash_scripts/eval_wm.sbatch` that takes `--wm {ctrlworld,ar,weaver}`
  and a benchmark/config name, selects the right venv + (for weaver) spins up the pi05
  websocket server, and dispatches. The per-WM specifics live in config, not in the user's
  command line.
- **Update the docs**: `docs/EVAL.md` (read it first — it's the canonical eval doc the user
  pointed to) should get a clear "policy eval under different world models" section
  documenting the unified launcher, the per-WM configs, and the one-liner to launch each.

### Suggested order of work
1. `ls external/`, read `docs/EVAL.md`, read the two adapter files, read the existing
   sbatch scripts in `bash_scripts/` and the eval YAMLs in `configs/evaluation/` — build an
   accurate picture before changing anything. (Confirm the WEAVER repo layout at
   `/scratch/gpfs/AM43/yy4041/WEAVER` and how its venv was built.)
2. Decide the vendoring approach per dependency (match existing `external/` convention).
3. Vendor weaver → `external/`; fix the adapter import. Vendor/replace AR's cross-repo
   symlinks. Address ctrl-world's external dependency.
4. Design + implement the unified `bash_scripts/eval_wm.sbatch` (or equivalent) + the
   per-WM config layering.
5. Smoke-test each WM (2 inits, full duration). Fix regressions.
6. Rewrite the EVAL.md policy-eval section.
7. Update `memory/policy-eval-3wm.md` to reflect the new self-contained layout + launcher.

### Constraints / preferences
- Don't break the validated rollout quality (respect the gotchas in §2).
- Don't copy large checkpoints; symlink or reference them and document the source.
- The temp `scripts/_tmp_*` files in the repo root are throwaway debug scripts — safe to
  ignore or remove (confirm before deleting anything you didn't create).
- Use `AskUserQuestion` for genuine forks (e.g. "vendor weaver as a copy vs. git submodule",
  "rebuild the weaver venv under the repo or leave it in place") — but pick sensible
  defaults and keep momentum where the choice is obvious.

---

## 4. Files to know (verify each before trusting it)

- `openworld/world_models/ar_world_model.py` — AR adapter (self-contained except VAE/ckpt paths).
- `openworld/world_models/weaver_world_model.py` — weaver adapter (imports the external WEAVER repo — FIX THIS).
- `openworld/world_models/registry.py`, `__init__.py` — lazy registry (already done).
- `openworld/runners/__init__.py`, `openworld/policies/registry.py` — lazy/websocket enablers (already done).
- `scripts/run_evaluation.py`, `scripts/generate_videos.py`, `openworld/runners/evaluator.py` — the eval driver.
- `bash_scripts/eval_weaver_0617.sbatch` — weaver launcher (server + venv selection); template for the unified script.
- `configs/evaluation/0617_*_pi05.yaml` (+ `_smoke`/`_smoke_long`/`_smoke_rate` variants) — eval configs.
- `configs/training/ar_wan_studentinit_droid_aligned.py` — AR model config (referenced by the adapter).
- `data/droid_ar_latents/stats.json` — AR action-normalization stats.
- `docs/EVAL.md` — canonical eval doc to update.
- External (verify): `external/openpi`, `external/Wan2.1-T2V-1.3B-Diffusers`, WEAVER repo at `/scratch/gpfs/AM43/yy4041/WEAVER`.

## 5. The /tmp problem (why this handoff exists)
`/tmp` filled up; the Bash tool writes its output-capture file under
`/tmp/claude-<pid>/.../tasks/*.output`, so **every** shell command failed with `ENOSPC`
before the command even ran. Fix before/while relaunching: free `/tmp` (old `/tmp/claude-*`
session dirs are safe to delete) and/or `export TMPDIR=/scratch/gpfs/AM43/yy4041/tmp`
(mkdir it first) and start Claude Code from the repo dir. Confirm `df -h /tmp` (or the new
TMPDIR) shows free space before doing real work.
