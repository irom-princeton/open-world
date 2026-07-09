# 🕹️ Teleoperation (SpaceMouse → AR world model)

Drive a **4-step distilled** autoregressive world-model checkpoint live with a 3D
SpaceMouse and watch it dream the robot's video in your browser. This is the
interactive cousin of open-loop [trajectory replay](AUTOREGRESSIVE.md): instead of
feeding a recorded action sequence, *you* steer the end-effector and the model
generates the consequences block-by-block, keeping its KV-cache warm so each step
is cheap.

## The cluster split

The model needs a GPU, but the SpaceMouse is plugged into your laptop/desktop. So
the two halves run on different machines and talk over an SSH tunnel — only HTTP
crosses it, so this works for any GPU + local-machine combination:

```
laptop/desktop                          GPU node (compute)
┌──────────────────────────┐            ┌────────────────────────────────────┐
│ SpaceMouse (USB)          │            │ scripts/interactive_ar.py           │
│   │  robosuite reader     │            │   ARWorldModel + Wan VAE            │
│   ▼                       │            │   InteractiveRoller (4-step deploy) │
│ scripts/spacemouse_client │  HTTP      │   MJPEG stream + /action endpoint   │
│   │  POST /action ────────┼──tunnel────┼─► drives the dreamed rollout        │
│ browser  ◄── MJPEG /stream┼────────────┼─◄ live video                        │
└──────────────────────────┘            └────────────────────────────────────┘
                    ssh -N -L 8000:<gpu-node>:8000 <you>@<login>
```

- **GPU node** runs `scripts/interactive_ar.py` (loads the checkpoint, runs the
  few-step roller, serves the video + a continuous `/action` endpoint).
- **Laptop** runs `scripts/spacemouse_client.py` — the *same* SpaceMouse reader as
  robocasa's [`demo_teleop.py`](https://github.com/robocasa) (`robosuite.devices.SpaceMouse`),
  so the recorded states/actions are accurate by construction. It integrates the
  6-DOF deltas + gripper into a normalized pose and POSTs it through the tunnel.
- **Browser** (`http://localhost:8000`) shows the MJPEG stream of the dreamed video.

## 0. Laptop setup (one-time)

The laptop only runs the SpaceMouse *client*, so it needs a small dedicated env —
**not** the full project `.venv`. Build one with `robosuite` + `hidapi`:

```bash
cd open-world
uv venv .venv-teleop --python 3.11
uv pip install --python .venv-teleop/bin/python robosuite hidapi
```

Confirm the SpaceMouse is visible (vendor `0x256f`); this prints its product id:

```bash
.venv-teleop/bin/python -c "import hid; print([hex(d['product_id']) for d in hid.enumerate(0x256f, 0)])"
# e.g. ['0xc62e']  -> 3Dconnexion SpaceMouse Wireless, matches robosuite's defaults
```

**USB permissions (Linux).** hidapi needs rw on the device node. This laptop is
already set up via `/etc/udev/rules.d/99-spacemouse.rules` (grants the `plugdev`
group access to `256f:*`) and the user is in `plugdev`. On a fresh machine, create
that rule, add yourself to `plugdev`, then replug:

```bash
sudo tee /etc/udev/rules.d/99-spacemouse.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG plugdev "$USER"   # log out/in for group change to take effect
```

The `hidapi` pip wheel bundles the native library, so no `apt install libhidapi*`
is needed. Notes: the `robosuite>=1.5` device class is driven env-lessly by the
client (it supplies a tiny stub env); the SpaceMouse Wireless (`256f:c62e`) matches
robosuite's default macros, so you don't need `--vendor-id`/`--product-id`.

## Quick start — toy 2D sanity check (no GPU)

Simulating teleoperation control with a simple 2D environment to test latency: no GPU/checkpoint required.

### 1. (Compute Node) launch environment

```bash
cd /path/to/open-world
uv run python scripts/marker_env.py --port 8000
#   --width/--height  canvas size       --fps  render/stream rate (default 20)
#   --trail N         marker trail len   --action-dim  pose size (default 7)
#   --sim-latency MS  fake per-frame compute to mimic a heavier model (e.g. 20; see §4)
```

### 2. (Local) launch SSH tunnel

```bash
ssh -N -L 8000:<node>:8000 <you>@<cluster-login>
# e.g. ssh -N -L 8000:localhost:8000 <user>@<gpu-host>
```


### 3. (Local) start teleoperation

```bash
# laptop, teleop env from step 0:
cd open-world
.venv-teleop/bin/python scripts/spacemouse_client.py --url http://localhost:8000 --latency
```

Open `http://localhost:8000/` and start driving the marker with the space mouse. The interface will print out latency for each interaction for reference.

## Prerequisites

- A **4-step distilled** student checkpoint (`*.pt`) and its training config. The
  4-step *deployment* schedule (`--distilled`) only looks right on a **distilled**
  checkpoint; on a non-distilled / mid-training (studentinit) backbone it produces
  a blurry colour-wash — sample those with the many-step preview schedule instead
  (omit `--distilled`).
- A way to **prime** the world (the "first frame" the model starts dreaming from).
  Two options:
  - **Bundled example inits (no latents to download).** A couple of still
    initializations ship in [`assets/teleop_inits/`](../assets/teleop_inits) — per-view
    PNGs + an `initialization.yaml` (robot pose + instruction) and the action-norm
    `stats.json`. The server VAE-encodes a still on demand and repeats it across the
    history block, so a **fresh clone can teleop with nothing downloaded but the
    checkpoint**. This is the default `--benchmark-root`; see the latent-free quick
    start below.
  - **Preprocessed seed episodes** (the richer path): the Wan-VAE latents used for
    replay / training (see [AUTOREGRESSIVE.md](AUTOREGRESSIVE.md)), passed via
    `--latent-root`. The model is primed from one ground-truth episode's first
    frames, then you drive from there.
- On the **laptop**: a Python env with `robosuite` + `hidapi`, and the SpaceMouse
  plugged in. The client only imports `robosuite` for the device — everything else
  is Python stdlib (no torch, no GPU). See [Laptop setup](#0-laptop-setup-one-time)
  above for the exact commands.

## 1. Launch the server on the GPU node

### Latent-free quick start (bundled example inits, nothing to download)

The fastest way to see teleop working: prime from one of the example stills in
`assets/teleop_inits/` (the default `--benchmark-root`). No `--latent-root`, no
preprocessed `.pt` episodes — only the distilled checkpoint:

```bash
cd /path/to/open-world
uv run python scripts/interactive_ar.py \
    --config configs/inference/ar_wan_droid_3view_cartesian.py \
    --checkpoint /path/to/<your_distilled_3view_cartesian>.pt \
    --distilled --bf16 --static-cache --max-kv-blocks 8 --compile \
    --port 8000
# pick the config whose num_cams / action_space match the checkpoint (see below)
```

The browser shows an **"initialization (benchmark suite)"** dropdown — pick an init
(e.g. `init_0`) and hit *reseed* to prime the world from that still, then drive.

Each init ships **all three views** (`exterior_left.png`, `exterior_right.png`,
`wrist.png`), so the same inits work for **both 2- and 3-view** cartesian checkpoints
— just swap `--config` between `ar_wan_droid_2view_cartesian.py` and
`ar_wan_droid_3view_cartesian.py` (the loader subsets views by the config's
`num_cams`, keeping the wrist + the first side views). They are **cartesian** (7-dim),
so pair them with a `*_cartesian` config + checkpoint, **not** `jointpos`. Their
`stats.json` (action normalization) is bundled too and loaded automatically when no
`--latent-root` stats are present. To prime from your *own* stills, point
`--benchmark-root` at a dir of `init_*/` subdirs, each with the three view PNGs + an
`initialization.yaml`.

### Recommended defaults (lowest felt latency)

This is the config to use for live teleop. It is the validated low-latency path —
bf16 + a fused, bounded, fixed-shape attention cache — with **synchronous decode**
(decode stays in the critical path on purpose; see the rationale below).

```bash
# on an interactive GPU node (this cluster's compute nodes are offline -> salloc)
cd /path/to/open-world
uv run python scripts/interactive_ar.py \
    --config configs/inference/ar_wan_droid_3view_cartesian.py \
    --checkpoint /path/to/<your_distilled_3view_cartesian>.pt \
    --latent-root /path/to/<preprocessed_latents> \
    --distilled \
    --bf16 --static-cache --max-kv-blocks 8 --compile \
    --measure-latency \
    --record-dir runs/teleop/$(date +%Y%m%d_%H%M%S) \
    --port 8000
#   add --rotate-wrist if your wrist camera is mounted inverted (display-only)
```

Pick the `configs/inference/ar_wan_droid_{2,3}view_{cartesian,jointpos}.py` config
whose `num_cams` / `action_space` **match how the checkpoint was trained** (see
[configs/inference/README.md](../configs/inference/README.md)). `cartesian` (7-dim)
and `joint_pos` (8-dim) checkpoints are not interchangeable.

**What each flag does, and why these are the defaults:**

| flag | why |
|---|---|
| `--distilled` | few-step deployment schedule (`denoising_step_list`, e.g. `(1000,750,500,250)`). **Required** on a distilled checkpoint; on a non-distilled backbone it's a blurry colour-wash (omit it for the many-step preview schedule). |
| `--bf16` | pure-bf16 deploy (params + decode VAE), drops autocast. ~1.3×. |
| `--static-cache` | fixed-shape ring-buffer KV cache. Correctness-neutral; prerequisite for `--compile`. |
| `--max-kv-blocks 8` | **bounds the attention window.** Without it the KV cache is *unbounded* and the forward time grows with session length (climbs from ~1.3 s into multiple seconds). 8 is the validated window. |
| `--compile` | `torch.compile` fusion on the block loop (~1.4× on the forward). The one-time ~15 s compile is **warmed up automatically at startup** (see below), so it never stalls your first move. |
| `--measure-latency` | logs per-block `a2e / qwait / step / fwd` and exposes `a2e_ms` in `/state`. Drop it once you're happy. |
| `--record-dir` | writes the session (`actions_norm/raw.npy`, `teleop.mp4`, `meta.json`) on Ctrl-C. |

**Do NOT use these for normal teleop:**

- **`--decode-device cuda:1`** — offloads the VAE decode to a 2nd GPU. It's a
  *throughput* win but a *latency* loss: in overlapped mode `step()` returns the
  *previous* block's frames, adding a full block of action→pixel lag. Generation
  already outruns world-time, so for teleop **feel** keep decode synchronous (omit
  this). Use it only for headless throughput/recording runs.
- **`--continuous`** (and `--roll-when-idle`) — keeps the world dreaming forever at
  the held pose. It never pauses when idle, so a **reset/reseed can't visibly hold**
  (the world resets then immediately resumes drifting). Omit it: the world then
  pauses on the last frame when idle and snaps cleanly on reset.

**Startup warmup (automatic with `--compile`).** After seeding, the server paints
the clean initial frame, runs a few discarded warmup `step()`s to compile the
denoise+commit graphs, then re-primes to the pristine initial frame — so the ~15 s
compile is paid at boot, not on your first SpaceMouse move. You'll see
`warming compiled graphs ... / warmup done in ~15s` in the log; `/state` reports
`status: warming` meanwhile.

**Expected latency** (3-view, sync decode, distilled 4-step): forward ~0.45–0.5 s,
full block (forward + 3-view VAE decode) ~0.7–0.8 s, so **action→pixels ≈ 0.7 s**
with an empty queue (spikes higher when a pose lands mid-block). The biggest further
lever is 4→2 step distillation (halves the forward). Going 3-view → 2-view also cuts
both forward and decode if the quality is acceptable.

Other knobs: `--seed-episode <id>` picks the priming clip, `--fps` the playback rate,
`--host`/`--port` the bind address. The server prints the exact tunnel command for
its node.

## 2. Open the SSH tunnel (from your laptop)

```bash
ssh -N -L 8000:<gpu-node>:8000 <you>@<cluster-login>
# then open http://localhost:8000 in your browser to watch the stream
```

This is the identical tunnel shown in the toy-env quick start above (§2).

## 3. Drive with the SpaceMouse (from your laptop)

```bash
# laptop, in the teleop env from step 0, SpaceMouse plugged in
cd open-world
.venv-teleop/bin/python scripts/spacemouse_client.py --url http://localhost:8000
```

Translate/rotate the SpaceMouse to move the end-effector. The gripper is **latching
by default**: each grasp-button *press* toggles it open ↔ closed and it stays there
(one click to close, another to open — no need to hold). Pass `--gripper-momentary`
for the old hold-to-close behaviour (release = open), and `--invert-gripper` to swap
that mapping. Tune feel with `--pos-gain` / `--rot-gain` (normalized units per tick),
`--rate` (POST Hz), and robosuite's `--pos-sensitivity` / `--rot-sensitivity`. Pass
`--vendor-id` / `--product-id` if your SpaceMouse model isn't the robosuite/robocasa
default. The browser keyboard D-pad (W/A/S/D · O/L · K/J) still works as a no-hardware
fallback.

## 4. Recorded trajectory (`--record-dir`)

On Ctrl-C the server writes, into the directory you passed:

| file | contents |
|---|---|
| `actions_norm.npy` | `[num_blocks, action_dim]` normalized EEF poses, one per generated AR block |
| `actions_raw.npy`  | the same, de-normalized to raw units via the run's percentile stats |
| `teleop.mp4`       | the dreamed video (`frames_per_block × 4` RGB frames per block) |
| `meta.json`        | config, checkpoint, seed episode, schedule, action space, fps, counts |

Actions are logged **per AR block**; the cadence vs. RGB frames is recorded in
`meta.json` (`frames_per_block`, `rgb_frames_per_block`).

## Smoke test (no GPU / no hardware)

Verify the whole laptop↔server path before wiring real hardware:

1. **Client protocol, hardware-free** — `--device dummy` synthesizes a slow circle
   so you can confirm POSTs land and integrate correctly, even against a stub:
   ```bash
   .venv-teleop/bin/python scripts/spacemouse_client.py --device dummy --url http://localhost:8000 --verbose
   ```
2. **Repo smoke test** — `openworld/autoregressive/tests/test_teleop_protocol.py` asserts the `/action`
   contract (start-pose sync, clipping to [−1, 1], delta integration) against a
   stub server, and boots the real `marker_env.py` to confirm the model-free
   path produces frames and tracks the marker:
   ```bash
   uv run python -m pytest openworld/autoregressive/tests/test_teleop_protocol.py -q
   ```
3. **Server plumbing** — `scripts/interactive_ar.py` boots without `--checkpoint`
   (untrained backbone; video is noise but the build→roll→decode→serve→`/action`
   path is exercised end to end).

## Troubleshooting

- **`/action` returns 409 "not seeded yet"** — the server is still priming; it seeds
  on startup and the client adopts the pose automatically once `/state` reports one.
- **`could not reach .../state`** — the tunnel is down or the server isn't running;
  re-check the `ssh -N -L` command the server printed.
- **SpaceMouse won't open** — install `hidapi`, check USB permissions (Linux often
  needs a udev rule), or pass the right `--vendor-id`/`--product-id`. Use
  `--device dummy` to confirm everything else works first.
- **Blurry colour-wash** — you used `--distilled` on a non-distilled checkpoint;
  drop `--distilled` (preview schedule) or point at a distilled `.pt`.
