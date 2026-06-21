# Running Policy Evaluation in OpenWorld

Policy evaluation drives a policy (e.g. pi0.5 / openpi) **closed-loop** inside a
world model over a suite of initializations, producing rollout videos (and,
optionally, reward annotations). It is a two-phase pipeline:

```
run_evaluation.py  (reads an eval YAML)
   └─ spawns generate_videos.py  →  WorldModelEnv + Evaluator
                                     (policy ⇄ world model, closed loop)
   └─ (optional) scores the videos with a reward model
```

A run is fully specified by an eval YAML under `configs/evaluation/`:
world model + checkpoint, policy + checkpoint, reward model, the benchmark
suite (`dataset_path`), and the output dir (`video_dir`).

---

## Policy eval under the three world models (pi0.5)

We run **pi0.5** (`pi05_droid`) closed-loop inside three world models. All
rollout-generation **code is self-contained in this repo**; only large weights
and benchmark data stay external (reached through symlinks).

| WM | name | code | venv | pi0.5 |
|----|------|------|------|-------|
| **Ctrl-World** (SVD) | `vidwm` | `external/vidwm` (vendored pkg) | `.venv-eval` | in-process |
| **AR slow student** (Wan block-causal) | `ar_wan` | `openworld/autoregressive` | `.venv-eval` | in-process |
| **weaver fast** (few-step) | `weaver` | `external/WEAVER` (submodule) | `.venv-weaver` | out-of-process websocket server |

Per-WM specifics — which venv, whether to spin up the pi0.5 websocket server,
PYTHONPATH, diffusion step counts, view order, action adapter — are abstracted
into the per-WM eval YAML and the launcher's runtime profile. You only choose
the **world model + benchmark + checkpoint**.

### One-time setup (on a login node, which has internet)

```bash
bash bash_scripts/setup_eval_env.sh            # submodules + .venv-eval + symlinks
bash bash_scripts/setup_eval_env.sh --venv-weaver   # also build the weaver venv (heavy)
```

This initializes the `external/openpi` and `external/WEAVER` submodules, builds
`.venv-eval` (`uv sync --extra policy-openpi`), and creates the checkpoint/data
symlinks (weights stay external). `--venv-weaver` additionally builds the
torch-2.7 weaver venv from `external/WEAVER/pyproject.toml`.

### Launch a run — one unified entry point

```bash
sbatch bash_scripts/eval_wm.sbatch --wm ctrlworld          # Ctrl-World
sbatch bash_scripts/eval_wm.sbatch --wm ar                 # AR slow student
sbatch bash_scripts/eval_wm.sbatch --wm weaver             # weaver fast (auto-starts pi0.5 server)
```

The eval YAML is resolved as `configs/evaluation/<benchmark>_<wm>_pi05.yaml`
(`--benchmark` defaults to `0617`). The launcher selects the right venv, and for
`weaver` it launches the pi0.5 `serve_policy.py` websocket server in `.venv-eval`
on the shared H200 and waits for it before starting the rollout.

Optional overrides (forwarded to `run_evaluation.py`, no YAML edit needed):

```bash
sbatch bash_scripts/eval_wm.sbatch --wm ar --checkpoint <ckpt.pt>
sbatch bash_scripts/eval_wm.sbatch --wm weaver --dataset data/benchmark/0617_smoke --duration 10
sbatch bash_scripts/eval_wm.sbatch --wm ctrlworld --config configs/evaluation/0617_ctrlworld_pi05.yaml
```

`--config <path>` · `--benchmark <name>` · `--checkpoint <path>` ·
`--dataset <path>` · `--video-dir <path>` · `--duration <sec>` · `--port <n>`

Outputs land under `video_dir/videos/` (per the YAML, e.g.
`outputs/0617_pi05/{ctrlworld,ar,weaver}/videos/`).

### Where the external bits live (weights + data, not code)

| symlink (in-repo) | source |
|-------------------|--------|
| `external/openpi`, `external/WEAVER` | git submodules (code, tracked) |
| `external/vidwm` | vendored source (code, tracked) |
| `external/stable-video-diffusion-img2vid`, `external/clip-vit-base-patch32` | SVD / CLIP weights |
| `checkpoints/action_adapter/model2_15_9.pth` | cartesian action adapter |
| `checkpoints/ctrlworld`, `checkpoints/weaver` | WM checkpoint dirs |
| `data/benchmark/0617_generated` | benchmark suite (12 inits) |

The pi0.5 policy checkpoint (`pi05_droid`) is an openpi asset under
`~/.cache/openpi/...` referenced by absolute path in the eval YAMLs.

### Validating a new adapter or change

Re-validate with a **2-init smoke at full duration (25 s)** per WM — short smokes
miss back-half drift — and eyeball the stacked wrist/exterior band ordering. e.g.:

```bash
sbatch bash_scripts/eval_wm.sbatch --wm weaver --dataset data/benchmark/0617_smoke
```

---

## Building a test suite

Each test case provides initial observations (left / right / wrist views), the
initial robot state + gripper, and an optional instruction. Place suites under
`data/benchmark/<name>` (or `data/evaluation_suites/<name>`) and point the eval
config's `dataset_path` at it. See [SCENEGEN.md](SCENEGEN.md)
for the scenegen pipeline that builds initialization suites from an instruction +
image.

## Reward scoring (optional)

`reward_model: dummy` produces videos only (the default for the above). To score
with Robometer:

```bash
git clone https://github.com/robometer/robometer.git external/robometer
uv sync --extra reward-robometer
# then set reward_model.name: robometer in the eval YAML
```

💡 `num_inference_steps` controls diffusion denoising steps (lower = faster,
higher = better quality). `duration` controls rollout length in seconds.
