# Bundled teleop initializations (latent-free seeds)

Example **still initializations** for live teleoperation
(`scripts/interactive_ar.py`), so you can drive the AR world model from a fresh
clone **without downloading any preprocessed latents** — only the model checkpoint.
See [`docs/TELEOPERATION.md`](../../docs/TELEOPERATION.md) for the full guide.

This directory is the default `--benchmark-root`. Each `init_*/` is one still scene:

```
init_0/
  exterior_left.png      # left side camera
  exterior_right.png     # right side camera
  wrist.png              # wrist camera
  initialization.yaml    # robot initial pose (cartesian, 7-dim) + instruction
stats.json               # action-normalization percentiles (cartesian)
```

At seed time the server VAE-encodes the three views into a single latent frame and
**repeats it across the model's history block** (there's no recorded clip — just the
still), using the YAML's `initial_state.robot.state` as the (constant) seed pose.
That's why no `.pt` latents are needed.

**These inits are cartesian (7-dim).** Use a `*_cartesian` config + checkpoint; the
bundled `stats.json` is loaded automatically when `--latent-root` has no stats.

**Add your own:** drop a new `init_<name>/` here (the three PNGs + an
`initialization.yaml` with `initial_state.robot.state` and `instruction`), or point
`--benchmark-root` at any directory of such subdirs.
