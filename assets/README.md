# assets/ — base initialization views for scene generation

Each subdirectory is a **base**: a fresh, unedited initialization that scene-edit
suites are built on top of. A base holds the three world-model views plus a
`template.yaml` with the robot start state and a default `scene` tag.

```
assets/<base>/
├── wrist.png            # 320×192 top-down wrist camera
├── exterior_left.png    # 320×192
├── exterior_right.png   # 320×192
└── template.yaml        # initial_state (robot pose) + default scene tag
```

| Base | Views from | Used by |
|------|------------|---------|
| `tri`  | `data/benchmark/0617_generated/_base_original/` | the `0617_generated` suite |
| `irom` | `open-world/data/benchmark/irom_carrot_pnp/init_2/` | irom-princeton DROID setup |

## Use

Reference a base by name (`tri` / `irom`) in a suite spec and build with
nanobanana all-views edits — see
[`configs/scenegen/suites/example.yaml`](../configs/scenegen/suites/example.yaml)
and [`docs/SCENEGEN.md`](../docs/SCENEGEN.md):

```bash
GOOGLE_API_KEY=... python scripts/scenegen/build_suite.py \
    --spec configs/scenegen/suites/example.yaml
# -> data/benchmark/<name>/init_*/
```

## Adding a base

Drop the three `*.png` views into a new `assets/<name>/`, add a `template.yaml`
with an `initial_state` block (copy one from an existing
`initialization.yaml`) and a `scene:` tag, then reference `<name>` from a spec.
